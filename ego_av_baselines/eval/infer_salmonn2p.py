#!/usr/bin/env python3
"""video-SALMONN2+ 7B adapter — settings copied from video_SALMONN2_plus/inference.py (the README's evaluation setting):
video_max_frames 768, video_min_frames 16, base_interval 0.1, max_pixels = video_max_frame_pixels = 61250,
Qwen2VLImageProcessorFast + WhisperFeatureExtractor, the repo's own dataset _get_item(), use_audio=True, bf16,
greedy decoding. Liger kernels are applied as in the script when liger_kernel is installed (numerically equivalent).
A clip without an audio track is retried with use_audio=False (recorded). A benchmark system prompt (EgoToM) is put
before the question in the human turn.
usage: infer_salmonn2p.py MODEL_DIR REQUESTS OUT_DIR SHARD NUM_SHARDS      env SALMONN_REPO = .../video_SALMONN2_plus
"""
import os, sys
import torch

MODEL_DIR, REQ, OUT, SHARD, NSH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_infer as C  # noqa: E402

REPO = os.environ.get("SALMONN_REPO", "/mnt/shared-storage-user/ai4good1-share/liyu/repos_bench/video-SALMONN-2/video_SALMONN2_plus")
sys.path.insert(0, REPO)
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus  # noqa: E402
from qwenvl.data.dataset import make_supervised_data_module  # noqa: E402
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast  # noqa: E402
from qwenvl.train.argument import DataArguments  # noqa: E402
from transformers import AutoTokenizer, WhisperFeatureExtractor  # noqa: E402

try:
    from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from qwenvl.model import modeling_qwen2_5_vl as _m
    _m.apply_multimodal_rotary_pos_emb, _m.Qwen2RMSNorm, _m.Qwen2MLP = liger_multimodal_rotary_pos_emb, LigerRMSNorm, LigerSwiGLUMLP
    LIGER = True
except ImportError:
    LIGER = False

da = DataArguments()
da.video_max_frames, da.video_min_frames, da.base_interval = 768, 16, 0.1
da.max_pixels = da.video_max_frame_pixels = 61250
da.run_test = True
da.image_processor = Qwen2VLImageProcessorFast.from_pretrained(MODEL_DIR)
da.audio_processor = WhisperFeatureExtractor(feature_size=da.feature_size, sampling_rate=da.sampling_rate,
                                             hop_length=da.hop_length, chunk_length=da.chunk_length)
da.model_type = "qwen2.5vl"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, model_max_length=131072, padding_side="right", use_fast=False)
try:
    import flash_attn  # noqa: F401
    ATTN = "flash_attention_2"
except ImportError:
    ATTN = "sdpa"
model = video_SALMONN2_plus.from_pretrained(MODEL_DIR, attn_implementation=ATTN, torch_dtype=torch.bfloat16, device_map="cpu")
model.cuda().eval()
test_data = make_supervised_data_module(tokenizer=tokenizer, data_args=da)["train_dataset"]


def prepare(inputs):
    for k in ("video", "image", "prompt", "ref", "audio", "use_audio", "should_use"):
        inputs.pop(k, None)
    return {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}


def infer(r):
    text = f"{r['system']}\n\n{r['user']}" if r.get("system") else r["user"]
    extra = {"attn": ATTN, "liger": LIGER}
    for use_audio in (True, False):
        try:
            item = {"video": C.media(r["video"]), "use_audio": use_audio,
                    "conversations": [{"from": "human", "value": f"<video>\n{text}"}, {"from": "gpt", "value": ""}]}
            inputs = prepare(test_data._get_item(item))
            break
        except Exception as e:
            if not use_audio:
                raise
            extra["audio_error"] = str(e)[:200]
    extra["audio"] = use_audio
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=r["max_new_tokens"], do_sample=False)
    pred = tokenizer.decode(out[0, len(inputs["input_ids"][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return pred.strip(), extra


C.run("salmonn2p", REQ, OUT, SHARD, NSH, infer)
