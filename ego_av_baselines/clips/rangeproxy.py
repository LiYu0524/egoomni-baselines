"""Local HTTP range server that makes one ffmpeg read fast behind a per-connection-throttled proxy.

ffmpeg reads a remote MP4 over a single connection; behind the cluster proxy that connection is capped at ~10-20 KB/s.
This server sits between ffmpeg and the upstream (a presigned URL): every byte range ffmpeg asks for is split into
fixed blocks that are fetched over many parallel upstream connections, with bounded read-ahead and an LRU block cache
(the three EgoToM context lengths of one clip reuse the same blocks).

    rp = RangeProxy(workers=48); rp.start()
    local_url = rp.register(upstream_url, size)   # -> http://127.0.0.1:<port>/<token>
"""
import collections, http.server, os, socketserver, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
import requests

BLOCK = int(os.environ.get("RP_BLOCK_MB", "1")) << 20   # larger blocks amortise per-request latency on far-away buckets


class _Blocks:
    def __init__(self, workers, cache_bytes, read_ahead, log):
        self.pool = ThreadPoolExecutor(workers)
        self.cache = collections.OrderedDict()          # (token, idx) -> Future[bytes]
        self.cap = max(8, cache_bytes // BLOCK)
        self.ahead = read_ahead
        self.lock = threading.Lock()
        self.tls = threading.local()
        self.log = log
        self.fetched = 0
        self.seq = {}                                   # token -> (last block index read, sequential streak)

    def _sess(self):
        if not hasattr(self.tls, "s"):
            self.tls.s = requests.Session()
        return self.tls.s

    def _fetch(self, url, a, b):
        err = None
        for attempt in range(30):
            try:
                r = self._sess().get(url, headers={"Range": f"bytes={a}-{b}", "Accept-Encoding": "identity"},
                                     timeout=(30, 120))
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    with self.lock:
                        self.fetched += len(r.content)
                    return r.content
                err = f"HTTP {r.status_code} len {len(r.content)}"
            except Exception as e:
                err = str(e)[:150]
            time.sleep(min(30, 2 ** min(attempt, 5)))
        self.log(f"block {a}-{b} failed 30x: {err}")
        raise IOError(err)

    def get(self, token, url, size, idx):
        """Future for block idx; schedules read-ahead that grows only while the reader stays sequential
        (ffmpeg's header/index probes and seeks then cost a block or two instead of a full read-ahead window)."""
        nblocks = (size + BLOCK - 1) // BLOCK
        with self.lock:
            last, streak = self.seq.get(token, (-2, 0))
            streak = streak + 1 if idx == last + 1 else (streak if idx == last else 0)
            self.seq[token] = (idx, streak)
            ahead = min(self.ahead, 1 + 2 * streak)
            for j in range(idx, min(nblocks, idx + 1 + ahead)):
                k = (token, j)
                if k in self.cache:
                    self.cache.move_to_end(k)
                    continue
                a = j * BLOCK
                self.cache[k] = self.pool.submit(self._fetch, url, a, min(size, a + BLOCK) - 1)
            while len(self.cache) > self.cap:
                old_k, old_f = self.cache.popitem(last=False)
                old_f.cancel()
            return self.cache[(token, idx)]

    def drop(self, token):
        with self.lock:
            self.seq.pop(token, None)
            for k in [k for k in self.cache if k[0] == token]:
                self.cache.pop(k).cancel()


class RangeProxy:
    def __init__(self, workers=48, cache_bytes=3 << 30, read_ahead=16, log=print):
        self.blocks = _Blocks(workers, cache_bytes, read_ahead, log)
        self.files = {}   # token -> (url, size)
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _range(self, size):
                rng = self.headers.get("Range")
                if not rng or not rng.startswith("bytes="):
                    return 0, size - 1, False
                a, _, b = rng[6:].split(",")[0].strip().partition("-")
                if a == "":                                # suffix range: last N bytes
                    return max(0, size - int(b)), size - 1, True
                return int(a), min(int(b) if b else size - 1, size - 1), True

            def do_HEAD(self):
                self._serve(body=False)

            def do_GET(self):
                self._serve(body=True)

            def _serve(self, body):
                tok = self.path.strip("/")
                if tok not in outer.files:
                    self.send_error(404)
                    return
                url, size = outer.files[tok]
                a, b, partial = self._range(size)
                if a >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(b - a + 1))
                if partial:
                    self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
                self.end_headers()
                if not body:
                    return
                pos = a
                try:
                    while pos <= b:
                        i = pos // BLOCK
                        data = outer.blocks.get(tok, url, size, i).result()
                        off = pos - i * BLOCK
                        chunk = data[off: off + (b - pos + 1)]
                        self.wfile.write(chunk)
                        pos += len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass                                   # ffmpeg closed the stream after seeking elsewhere

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = S(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    @staticmethod
    def remote_size(url):
        """Object size via a 1-byte GET (presigned URLs are signed for GET only, so HEAD returns 403)."""
        for attempt in range(10):
            try:
                r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=(30, 60))
                if r.status_code == 206:
                    return int(r.headers["Content-Range"].split("/")[-1])
                err = f"HTTP {r.status_code}"
            except Exception as e:
                err = str(e)[:150]
            time.sleep(2 ** min(attempt, 5))
        raise IOError(f"size of {url[:80]}: {err}")

    def register(self, url, size=None):
        size = size or self.remote_size(url)
        tok = uuid.uuid4().hex
        self.files[tok] = (url, size)
        return f"http://127.0.0.1:{self.port}/{tok}", tok

    def release(self, tok):
        self.files.pop(tok, None)
        self.blocks.drop(tok)
