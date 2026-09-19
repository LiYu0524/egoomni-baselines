import sys, time, copy, numpy as np, torch
sys.path.insert(0, "/ai4good1-shared/liyu/egoOmni_baselines/repos/video-SALMONN-2/video_SALMONN2_plus")
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from torchcodec.decoders import VideoDecoder
clip = sys.argv[1]; MD = "/ai4good1-shared/liyu/egoOmni_baselines/models/video-SALMONN2_plus_7B_full"
proc = Qwen2VLImageProcessorFast.from_pretrained(MD)
for nthreads in (1, 8):
    t = time.time(); dec = VideoDecoder(clip, num_ffmpeg_threads=nthreads); total = dec.metadata.num_frames; fps = dec.metadata.average_fps
    L = total / fps; target = min(max(round(L / 0.1), 16), 768); idx = np.unique(np.linspace(0, total - 1, target, dtype=int))
    fb = dec.get_frames_at(indices=idx.tolist()); frames = fb.data; print(f"decode threads={nthreads}: {len(idx)} frames {tuple(frames.shape)} in {time.time()-t:.1f}s")
def prep(video, device):
    p = copy.deepcopy(proc); new_pixel = 61250
    if len(idx) < 768: new_pixel = 0.95 * 768 / len(idx) * new_pixel
    p.max_pixels = new_pixel; p.min_pixels = 4 * 28 * 28; p.size["longest_edge"] = p.max_pixels; p.size["shortest_edge"] = p.min_pixels
    t = time.time(); out = p.preprocess(images=None, videos=video, return_tensors="pt", device=device); torch.cuda.synchronize(); dt = time.time() - t
    return out["pixel_values_videos"], out["video_grid_thw"], dt
for nt in (128, 16):
    torch.set_num_threads(nt); pv_cpu, thw, dt = prep(frames.numpy(), None); print(f"cpu preprocess threads={nt}: {dt:.1f}s  pixel_values {tuple(pv_cpu.shape)} grid {thw.tolist()}")
pv_gpu, thw2, dt = prep(frames.cuda(), "cuda"); print(f"gpu preprocess: {dt:.2f}s (incl. first-call warmup) grid {thw2.tolist()}")
pv_gpu, thw2, dt = prep(frames.cuda(), "cuda"); print(f"gpu preprocess (warm): {dt:.2f}s")
d = (pv_gpu.float().cpu() - pv_cpu.float()).abs(); print(f"gpu vs cpu pixel_values: max abs diff {d.max():.4g}, mean {d.mean():.2e}, cpu value range [{pv_cpu.min():.2f},{pv_cpu.max():.2f}]")
