#!/usr/bin/env python3
"""Native Qwen2.5-Omni inference that reproduces the colleague's EgoToM kit (egotom_infer_2fps_audio.py) step by step,
generalized from EgoToM's test_questions.json to any benchmark's request file.
Kept as in the kit: Qwen2_5OmniForConditionalGeneration in bf16 with flash_attention_2, talker disabled; a LoRA is merged into
the thinker weights with every adapter tensor verified (same asserts); LastTokenHead; Qwen2_5OmniProcessor(use_fast=False) +
qwen_omni_utils.process_mm_info with decord; video min_pixels 224*224 and max_pixels 156800; the video's own audio track via
use_audio_in_video whenever ffprobe finds an audio stream; greedy decoding, 256 new tokens, seed 42; per-shard resumable result
files over rows[SHARD::NUM_SHARDS] (questions that errored are retried on the next run).
Generalized: 2 fps is requested as fps=2.0 (for the kit's 30 s EgoToM clips that is exactly its nframes=60); a video given as a
list of frame files keeps the benchmark's own frame rate (sample_fps); optional video_start / video_end window; optional
per-request max_new_tokens. The env has transformers 4.54, where the kit's dtype= keyword is torch_dtype=.
Request fields: sample_id, system, user, video (path or list of frame paths), [fps | sample_fps], [video_start, video_end],
[max_new_tokens]; every other field is copied into the result row.
usage: nat_infer.py REQUESTS_JSONL OUT_DIR MODEL_DIR ADAPTER_DIR|none SHARD NUM_SHARDS"""
import gc, json, os, subprocess, sys, time, traceback
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

REQ, OUT, BASE, ADAPTER = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
SHARD, NUM_SHARDS = int(sys.argv[5]), int(sys.argv[6])
VARIANT = 'lora' if ADAPTER != 'none' else 'full'
LIMIT = int(os.environ.get('LIMIT', '0'))
ATTN = os.environ.get('ATTN', 'flash_attention_2')
FFPROBE = os.environ.get('FFPROBE', 'ffprobe')
MAX_PIXELS, MIN_PIXELS = 156800, 224 * 224
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get('NAT_THREADS', '4')))   # the kit: 4 (one process per GPU)
torch.manual_seed(42)


class LastTokenHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__(); self.head = head

    def forward(self, x):
        return self.head(x[:, -1:, :] if x.ndim == 3 else x)


def video_element(row):
    ele = {'type': 'video', 'video': row['video'], 'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS}
    if isinstance(row['video'], list):
        ele['sample_fps'] = row.get('sample_fps', 2.0)
    else:
        ele['fps'] = row.get('fps', 2.0)
        for k in ('video_start', 'video_end'):
            if row.get(k) is not None:
                ele[k] = row[k]
    return ele


def has_audio_stream(path):
    probe = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'json', path],
                           capture_output=True, text=True, check=True)
    return bool(json.loads(probe.stdout).get('streams'))


def main():
    rows = [json.loads(l) for l in open(REQ)][SHARD::NUM_SHARDS]
    if LIMIT:
        rows = rows[:LIMIT]
    path = OUT / f'result_{SHARD}.jsonl'
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                if r['status'] == 'ok':
                    done.add(r['sample_id'])
            except (ValueError, KeyError):
                pass
    rows = [r for r in rows if r['sample_id'] not in done]
    if not rows:
        print('SHARD_COMPLETE errors=0 (nothing pending)', flush=True)
        return
    (OUT / f'config_{SHARD}.json').write_text(json.dumps(dict(
        model=BASE, adapter=None if ADAPTER == 'none' else ADAPTER, attention=ATTN, torch=torch.__version__,
        transformers=transformers.__version__, qwen_omni_utils=version('qwen-omni-utils'), target_fps=2, min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS, use_audio_in_video='when ffprobe finds an audio stream', max_new_tokens=256, do_sample=False, seed=42,
        shard=SHARD, num_shards=NUM_SHARDS, requests=str(REQ), video_backend=os.environ.get('FORCE_QWENVL_VIDEO_READER'),
        protocol='colleague EgoToM kit egotom_infer_2fps_audio.py, generalized'), indent=2))
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
                                                                attn_implementation=ATTN, low_cpu_mem_usage=True)
    model.disable_talker()
    if VARIANT == 'lora':
        from safetensors.torch import load_file
        cfg = json.loads(Path(ADAPTER + '/adapter_config.json').read_text())
        assert cfg['bias'] == 'none' and not cfg.get('use_dora') and not cfg.get('use_rslora')
        assert not cfg.get('rank_pattern') and not cfg.get('alpha_pattern') and not cfg.get('modules_to_save') and not cfg.get('fan_in_fan_out')
        expected = load_file(ADAPTER + '/adapter_model.safetensors')
        used = set()
        with torch.no_grad():
            for k, a in expected.items():
                if not k.endswith('.lora_A.weight'):
                    continue
                prefix = k.removesuffix('.lora_A.weight')
                bkey = prefix + '.lora_B.weight'
                b = expected[bkey]
                module = model.thinker.get_submodule(prefix.removeprefix('base_model.model.'))
                assert isinstance(module, torch.nn.Linear)
                delta = (b.float() @ a.float()) * (cfg['lora_alpha'] / cfg['r'])
                assert delta.shape == module.weight.shape and torch.isfinite(delta).all()
                module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
                used.update([k, bkey])
        assert used == set(expected), 'Unapplied adapter tensors'
        print(f'LORA_VERIFIED_AND_MERGED tensors={len(used)} modules={len(used) // 2}', flush=True)
    model.thinker.lm_head = LastTokenHead(model.thinker.lm_head)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(BASE, use_fast=False)
    print(f'READY variant={VARIANT} shard={SHARD}/{NUM_SHARDS} attention={model.thinker.config._attn_implementation} '
          f'pending={len(rows)}', flush=True)
    failures = 0
    for row in rows:
        result = dict(row, variant=VARIANT, attention=ATTN)
        start = time.time()
        try:
            frames = isinstance(row['video'], list)
            for v in (row['video'] if frames else [row['video']]):
                assert Path(v).is_file(), str(v)
            conversation = [{'role': 'system', 'content': [{'type': 'text', 'text': row['system']}]},
                            {'role': 'user', 'content': [video_element(row), {'type': 'text', 'text': row['user']}]}]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            use_audio = False if frames else has_audio_stream(row['video'])
            result['audio_status'] = 'frames' if frames else ('present' if use_audio else 'absent_in_source')
            result['use_audio_in_video'] = use_audio
            audios, images, videos, vkwargs = process_mm_info(conversation, use_audio_in_video=use_audio, return_video_kwargs=True)
            if use_audio and (audios is None or len(audios) == 0):
                raise RuntimeError('Audio input missing despite audio stream')
            result['audio_samples'] = [len(a) for a in audios] if audios is not None else []
            fps = vkwargs['fps'][0]
            inputs = processor(text=prompt, audio=audios, images=images, videos=videos, fps=fps, return_tensors='pt', padding=True,
                               use_audio_in_video=use_audio).to(model.device)
            max_new = int(row.get('max_new_tokens', 256))
            with torch.inference_mode():
                generated = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False,
                                           thinker_max_new_tokens=max_new, thinker_do_sample=False)
            if hasattr(generated, 'sequences'):
                generated = generated.sequences
            new = generated[:, inputs.input_ids.shape[1]:]
            pred = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            if not pred:
                raise RuntimeError('Empty prediction')
            grid = inputs.video_grid_thw.tolist()
            result.update(status='ok', pred=pred, nframes=int(grid[0][0]) * 2, effective_fps=fps, max_pixels=MAX_PIXELS,
                          input_tokens=inputs.input_ids.shape[1], new_tokens=int(new.shape[1]), hit_max_new_tokens=int(new.shape[1]) >= max_new,
                          video_grid_thw=grid, peak_memory_bytes=torch.cuda.max_memory_allocated())
            del inputs, generated, videos, audios
        except Exception as exc:
            failures += 1
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        result['seconds'] = round(time.time() - start, 2)
        with path.open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"RESULT {row['sample_id']} {result['status']} {result['seconds']}s", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(f'SHARD_COMPLETE errors={failures}', flush=True)
    if failures:
        raise RuntimeError(f'{failures} questions failed; see result file')


if __name__ == '__main__':
    main()
