#!/usr/bin/env bash
# H rjob capability probe: FUSE device, gpfs mount, object-storage reachability, container disk.
H=/mnt/shared-storage-user/ai4good1-share/liyu
L=$H/tools/h/logs/probe_$(hostname).log; mkdir -p $(dirname $L); exec > >(tee -a $L) 2>&1
echo "[probe] $(date -Is) host=$(hostname) uid=$(id -u)"
grep PRETTY /etc/os-release; echo "nproc=$(nproc)"; free -g | head -2
df -h / /dev/shm /tmp 2>/dev/null | tail -3
echo "--- /dev/fuse: $(ls -la /dev/fuse 2>&1)"
echo "--- CapEff: $(grep CapEff /proc/self/status)"
echo "--- gpfs: $(ls $H | head -5 | tr '\n' ' ')"
python3 - <<'PY'
import urllib.request, time
for u in ("http://hdd2.h.pjlab.org.cn:8060", "https://pypi.org/simple/pip/", "https://github.com"):
    t = time.time()
    try:
        urllib.request.urlopen(u, timeout=8); print(u, "200", round(time.time()-t, 3))
    except Exception as e:
        print(u, type(e).__name__, getattr(e, "code", ""), str(e)[:80])
PY
echo "--- geesefs: $($H/tools/geesefs --version 2>&1 | head -1)"
echo "--- fusermount: $(which fusermount fusermount3 2>/dev/null)"
echo "--- nvidia: $(ls /dev/nvidia* 2>/dev/null | head -3 | tr '\n' ' ')"
echo "[probe] $(date -Is) done"
