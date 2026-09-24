#!/usr/bin/env python3
"""One-off: (1) let the four make_config.py take ADAPTER = model=DIR (a full-weight model instead of base + LoRA);
(2) create the egoOmni instance eval/fft_epoch2 from eval/base (same protocol, other model). Files are replaced atomically."""
import os

L = "/mnt/shared-storage-user/ai4good1-share/liyu"
BASE_MODEL = "/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train"
FFT_MODEL = "/ai4good1-shared/liyu/egoavu/models/groo_ckpt_fft_epoch2"


def rewrite(path, subs, dst=None):
    s = open(path).read()
    for a, b in subs:
        assert s.count(a) == 1, (path, a, s.count(a))
        s = s.replace(a, b)
    dst = dst or path
    open(dst + ".tmp", "w").write(s)
    os.chmod(dst + ".tmp", os.stat(path).st_mode)
    os.replace(dst + ".tmp", dst)
    print("wrote", dst)


OVERRIDE = (f'\nMODEL = "{BASE_MODEL}"\n'
            'if adapter.startswith("model="):   # a full-weight model (e.g. a full fine-tune) instead of base + LoRA adapter\n'
            '    MODEL, adapter = adapter[len("model="):], "none"')
for f, argv in (("egoavu/bench_infer/tools/make_config.py", "tag, adapter, dataset, out = sys.argv[1:5]"),
                ("egocross_eval_h/make_config.py", "tag, adapter, dataset, fps, out = sys.argv[1:6]"),
                ("egoschema_eval_h/make_config.py", "tag, adapter, dataset, out = sys.argv[1:5]"),
                ("egotaskqa_eval_h/make_config.py", "tag, adapter, dataset, out = sys.argv[1:5]")):
    s = open(f"{L}/{f}").read()
    usage = [("ADAPTER_DIR|none DATASET", "ADAPTER_DIR|none|model=DIR DATASET")] if "ADAPTER_DIR|none DATASET" in s else \
            [("ADAPTER|none DATASET", "ADAPTER|none|model=DIR DATASET")]
    rewrite(f"{L}/{f}", usage + [(argv, argv + OVERRIDE),
                                 (f'"model_name_or_path: {BASE_MODEL}"', 'f"model_name_or_path: {MODEL}"')])

E = f"{L}/egoOmni_baselines/eval"
os.makedirs(f"{E}/fft_epoch2", exist_ok=True)
R_OLD, R_NEW = "/ai4good1-shared/liyu/egoOmni_baselines/eval/base", "/ai4good1-shared/liyu/egoOmni_baselines/eval/fft_epoch2"
rewrite(f"{E}/base/build_data.py", [(f'R = "{R_OLD}"', f'R = "{R_NEW}"'),
                                    ("for the EgoAVU r100k LoRA on the egoOmni bench", "for groo ckpt_fft_epoch2 (full fine-tune) on the egoOmni bench")],
        f"{E}/fft_epoch2/build_data.py")
rewrite(f"{E}/base/collect.py", [(f'TAG, MODEL = "qwen25omni7b_base", "{BASE_MODEL}"', f'TAG, MODEL = "ckpt_fft_epoch2", "{FFT_MODEL}"'),
                                 ("eval/preds/qwen25omni7b_base/", "eval/preds/ckpt_fft_epoch2/")],
        f"{E}/fft_epoch2/collect.py")
rewrite(f"{E}/base/make_config.py", [(f'R = "{R_OLD}"', f'R = "{R_NEW}"'), (f'"model_name_or_path: {BASE_MODEL}"', f'"model_name_or_path: {FFT_MODEL}"')],
        f"{E}/fft_epoch2/make_config.py")
s = open(f"{E}/base/job.sh").read()
n = s.count("[base]")
s = s.replace(f"R={R_OLD}", f"R={R_NEW}").replace("[base]", "[fft_epoch2]").replace("/tmp/base_cache", "/tmp/fft_epoch2_cache") \
     .replace("EgoAVU r100k LoRA on egoOmni", "groo ckpt_fft_epoch2 (full fine-tune) on egoOmni")
assert "eval/base" not in s and "base_cache" not in s and "[base]" not in s and s.count("[fft_epoch2]") == n
open(f"{E}/fft_epoch2/job.sh.tmp", "w").write(s); os.chmod(f"{E}/fft_epoch2/job.sh.tmp", 0o755)
os.replace(f"{E}/fft_epoch2/job.sh.tmp", f"{E}/fft_epoch2/job.sh"); print("wrote", f"{E}/fft_epoch2/job.sh")
if not os.path.lexists(f"{E}/fft_epoch2/audio16k"):
    os.symlink("../r100k/audio16k", f"{E}/fft_epoch2/audio16k")
