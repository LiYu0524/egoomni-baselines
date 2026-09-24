#!/usr/bin/env python3
"""Drop-in replacement for `llamafactory-cli train <predict config>` on a LLaMAFactory sharegpt dataset, running the colleague's
EgoToM-kit protocol instead (same inference code path as tools/h/nat_infer.py): Qwen2_5OmniForConditionalGeneration bf16 +
flash_attention_2, talker disabled, LoRA merged with every tensor verified, LastTokenHead, Qwen2_5OmniProcessor(use_fast=False) +
qwen_omni_utils.process_mm_info (decord), video at 2 fps with min_pixels 224*224 / max_pixels 156800, greedy decoding, seed 42.
Per row: the first user turn's <video>[<audio>] placeholders become the video element (window from the '#t=a,b' fragment →
video_start/video_end) and, when the row has an audio file, that clip's soundtrack (16 kHz FLAC extracted from the same clip,
same window) is fed as audio-in-video exactly as the kit feeds a video's own track (librosa at 16 kHz, use_audio_in_video=True);
later turns are plain text; the row's last assistant message is the label, not part of the input.
Clips decord cannot decode (egoOmni's source clips) are read by a PyAV reader with the identical frame selection instead of
qwen_omni_utils' torchvision fallback, which loads the whole clip into RAM.
Long videos: at most MAX_FRAMES frames (default 360 = 3 min at 2 fps, the longest EgoSchema input), evenly spaced — the kit
itself only ever sees 30 s clips.
Writes OUT_DIR/generated_predictions.jsonl in row order (prompt = the chat text with unexpanded media placeholders, predict,
label, n_video_tokens, n_audio_tokens) + predict_results.json once every row succeeded; progress is resumable from
OUT_DIR/nat_rows.jsonl.   usage: nat_predict_lf.py DATASET_JSONL OUT_DIR MODEL_DIR ADAPTER_DIR|none MAX_NEW_TOKENS [MAX_FRAMES]"""
import gc, json, os, sys, time, traceback
from pathlib import Path

import av
import librosa
import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from qwen_omni_utils.v2_5 import vision_process as VP

DATA, OUT, BASE, ADAPTER, MAX_NEW = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
MAX_FRAMES = int(sys.argv[6]) if len(sys.argv) > 6 else 360
ATTN = os.environ.get('ATTN', 'flash_attention_2')
SYSTEM = "You are a helpful assistant."   # Qwen2.5-Omni chat-template default = what the LLaMAFactory runs received
MAX_PIXELS, MIN_PIXELS = 156800, 224 * 224
# 'SRC_PREFIX=>DST_PREFIX;...': read the video from a proxy tree instead (egoOmni: 2 fps H.264 proxies, see prep_egoomni_proxy2fps.py)
VIDEO_MAP = [tuple(x.split('=>', 1)) for x in os.environ.get('NAT_VIDEO_PREFIX_MAP', '').split(';') if '=>' in x]
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get('NAT_THREADS', '4')))
torch.manual_seed(42)


FALLBACK = {'used': False}


def _read_video_pyav(ele):
    """Stand-in for qwen_omni_utils' fallback reader, which it uses when decord cannot decode a clip (egoOmni's source clips):
    the SAME frame selection as its decord reader (calculate_video_frame_range + smart_nframes + linspace over the window),
    decoded with PyAV keeping only the selected frames. Its own fallback, torchvision.io.read_video, holds every frame of the
    clip at full resolution in RAM (~17 GB for a 90 s 1080p clip) and got the 4-process egoOmni jobs OOM-killed."""
    FALLBACK['used'] = True
    path = ele['video']
    with av.open(path) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate or vs.guessed_rate)
        total = vs.frames or sum(1 for pkt in c.demux(vs) if pkt.pts is not None)
    start, end, n_range = VP.calculate_video_frame_range(ele, total, fps)
    nframes = VP.smart_nframes(ele, total_frames=n_range, video_fps=fps)
    idx = torch.linspace(start, end, nframes).round().long().tolist()
    want, got = set(idx), {}
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = 'AUTO'
        for i, fr in enumerate(c.decode(vs)):
            if i in want:
                got[i] = fr.to_ndarray(format='rgb24')
            if i >= end:
                break
    assert got, f'no frames decoded from {path}'
    keys = sorted(got)
    pick = lambda i: got[i] if i in got else got[max([k for k in keys if k <= i] or keys[:1])]   # stream shorter than its header
    video = torch.from_numpy(np.stack([pick(i) for i in idx])).permute(0, 3, 1, 2)
    return video, dict(fps=fps, frames_indices=idx, total_num_frames=n_range, video_backend='pyav'), nframes / max(n_range, 1e-6) * fps


VP.VIDEO_READER_BACKENDS['torchvision'] = _read_video_pyav   # the key qwen_omni_utils.fetch_video falls back to when decord fails


class LastTokenHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__(); self.head = head

    def forward(self, x):
        return self.head(x[:, -1:, :] if x.ndim == 3 else x)


def split_frag(p):
    if '#t=' not in p:
        return p, None, None
    path, frag = p.split('#t=')
    a, b = frag.split(',')
    return path, float(a), float(b)


def load_model():
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
                                                                attn_implementation=ATTN, low_cpu_mem_usage=True)
    model.disable_talker()
    if ADAPTER != 'none':
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
    return model


def conversation_for(row):
    msgs = row['messages']
    assert [m['role'] for m in msgs] == ['user', 'assistant'] * (len(msgs) // 2), 'rows must alternate user/assistant'
    first = msgs[0]['content']
    assert first.startswith('<video>'), 'first turn must start with the video'
    first = first[len('<video>'):]
    with_audio = first.startswith('<audio>')
    if with_audio:
        first = first[len('<audio>'):]
    assert '<video>' not in first and '<audio>' not in first and all('<video>' not in m['content'] and '<audio>' not in m['content'] for m in msgs[1:])
    vpath, vs, ve = split_frag(row['videos'][0])
    for src, dst in VIDEO_MAP:
        if vpath.startswith(src):
            vpath = dst + vpath[len(src):]
            break
    ele = {'type': 'video', 'video': vpath, 'fps': 2.0, 'max_frames': MAX_FRAMES, 'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS}
    widened = False
    if vs is not None:
        if ve - vs < 1.0:   # qwen_omni_utils needs >= 2 frames; 2 EgoAVU-Bench windows are 0.125 s / 0.94 s -> 1 s around their centre
            c = (vs + ve) / 2; vs = max(0.0, c - 0.5); ve = vs + 1.0; widened = True
        ele.update(video_start=vs, video_end=ve)
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [ele, {'type': 'text', 'text': first}]}]
    for m in msgs[1:-1]:
        conv.append({'role': m['role'], 'content': [{'type': 'text', 'text': m['content']}]})
    audio = None
    if with_audio:
        apath, a0, a1 = split_frag(row['audios'][0])
        audio = librosa.load(apath, sr=16000, offset=a0 or 0.0, duration=None if a1 is None else a1 - (a0 or 0.0))[0]
    return conv, audio, msgs[-1]['content'], widened


def main():
    rows = [json.loads(l) for l in open(DATA)]
    part = OUT / 'nat_rows.jsonl'
    done = {}
    if part.exists():
        for line in part.read_text().splitlines():
            try:
                r = json.loads(line)
                if r['status'] == 'ok':
                    done[r['i']] = r
            except (ValueError, KeyError):
                pass
    todo = [i for i in range(len(rows)) if i not in done]
    t_start = time.time()
    if todo:
        model = load_model()
        processor = Qwen2_5OmniProcessor.from_pretrained(BASE, use_fast=False)
        vid_id, aud_id = processor.tokenizer.convert_tokens_to_ids('<|VIDEO|>'), processor.tokenizer.convert_tokens_to_ids('<|AUDIO|>')
        print(f'READY {DATA.name} pending={len(todo)}/{len(rows)} adapter={ADAPTER} max_new={MAX_NEW} max_frames={MAX_FRAMES}', flush=True)
        failures = 0
        for i in todo:
            t0, rec = time.time(), {'i': i}
            try:
                conv, audio, label, widened = conversation_for(rows[i])
                prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                FALLBACK['used'] = False
                _, images, videos, vkwargs = process_mm_info(conv, use_audio_in_video=False, return_video_kwargs=True)
                use_audio = audio is not None
                audios = [audio] if use_audio else None
                fps = vkwargs['fps'][0]
                inputs = processor(text=prompt, audio=audios, images=images, videos=videos, fps=fps, return_tensors='pt', padding=True,
                                   use_audio_in_video=use_audio).to(model.device)
                with torch.inference_mode():
                    gen = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False,
                                         thinker_max_new_tokens=MAX_NEW, thinker_do_sample=False)
                if hasattr(gen, 'sequences'):
                    gen = gen.sequences
                new = gen[:, inputs.input_ids.shape[1]:]
                pred = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                ids = inputs.input_ids[0]
                rec.update(status='ok', prompt=prompt, predict=pred, label=label, n_video_tokens=int((ids == vid_id).sum()),
                           n_audio_tokens=int((ids == aud_id).sum()), nframes=int(inputs.video_grid_thw[0][0]) * 2, effective_fps=fps,
                           input_tokens=int(ids.shape[0]), new_tokens=int(new.shape[1]), hit_max_new_tokens=int(new.shape[1]) >= MAX_NEW,
                           video_window_widened_to_1s=widened, video_decoder='pyav (decord failed)' if FALLBACK['used'] else 'decord',
                           peak_memory_bytes=torch.cuda.max_memory_allocated())
                del inputs, gen, videos, audios
            except Exception as exc:
                failures += 1
                rec.update(status='error', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            rec['seconds'] = round(time.time() - t0, 2)
            with part.open('a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if rec['status'] == 'ok':
                done[i] = rec
            print(f"ROW {i} {rec['status']} {rec['seconds']}s", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        if failures:
            raise RuntimeError(f'{failures} rows failed; rerun resumes them')
    with open(OUT / 'generated_predictions.jsonl.tmp', 'w') as f:
        for i in range(len(rows)):
            r = done[i]
            f.write(json.dumps({k: r[k] for k in ('prompt', 'predict', 'label', 'n_video_tokens', 'n_audio_tokens')}, ensure_ascii=False) + '\n')
    os.replace(OUT / 'generated_predictions.jsonl.tmp', OUT / 'generated_predictions.jsonl')
    json.dump({'predict_runtime': round(sum(done[i]['seconds'] for i in done), 1), 'predict_samples': len(rows),
               'protocol': 'colleague EgoToM kit (nat_predict_lf.py)', 'max_new_tokens': MAX_NEW, 'max_frames': MAX_FRAMES,
               'wall_seconds_this_run': round(time.time() - t_start, 1)}, open(OUT / 'predict_results.json', 'w'), indent=2)
    print(f'DONE {DATA.name}: {len(rows)} rows', flush=True)


if __name__ == '__main__':
    main()
