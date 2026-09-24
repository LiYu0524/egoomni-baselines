# H-cluster pipeline (2026-09-23 – 25)

Scripts used to run the evaluations in this repo on the H cluster (rjob, private GPU nodes of the `safevlagent_gpu` group,
8× H200 per node, no internet on the nodes). No credentials are stored here: object storage is reached with a named AWS profile
from a credentials file outside the repo.

- `h_prelude.sh` — sourced by every job: maps the old P-cluster paths (`/ai4good1-shared/…`) onto the H gpfs volume, mounts the
  dataset buckets read-only with geesefs (`h_mount_s3`, `h_mount_egoomni`, `h_mount_prefix`), refuses to run if a pod sees more or
  fewer GPUs than it requested (`h_check_gpus`), and points `CUDA_HOME` at a CUDA 12.6 `nvcc` for LLaMA-Factory/DeepSpeed.
- `submit_h.sh` — one rjob per call (1-GPU jobs by default; **not** `--privileged`, which exposes every GPU of the node).
- `bench2_h.sh`, `egoomni2_h.sh`, `egocross_h.sh`, `egoschema_h.sh` — inference jobs, two bs=1 predict processes per H200.
- `eab_judge_h.sh`, `egoomni_judge_h.sh` — judges (vLLM; Qwen3-235B-A22B-Instruct-2507 TP4 / Qwen3-32B TP1) + scoring.
- `conductor_h.sh`, `egoschema_chain.sh` — chain the stages; failed or stuck (pod-sandbox error) jobs are resubmitted.
- `mirror_egoomni_hf_to_s3.py`, `copy_ego4d_full_sources.sh`, `prep_egoomni_media_h.sh`, `restore_egoavu_h.sh` — data movement
  and media prep (HF → object storage with sha256 checks and HTTP-range resume; restored_v3 re-cut; 16 kHz audio).
- Lessons: CPU-only jobs on GPU nodes are capped at 64 CPUs cumulative; vLLM 0.10.1 needs transformers 4.x (4.56.2 used);
  request only the per-GPU CPU / RAM share (8 CPU / 64 GB per GPU) — 1-GPU jobs asking 24 CPUs sat in the queue while GPUs were idle
  on nodes whose CPUs were taken.

**Added 2026-09-25**
- `convert_thinker_ckpt.py` (+ `fft_convert_after_download.sh`, `fft_epoch2_CONVERSION_REPORT.txt`) — thinker-only checkpoint →
  full-Omni thinker view, weights byte-identical; `fft_chain.sh` — the full fine-tune on all five benchmarks; `patch_fft.py` — adds
  `ADAPTER = model=<dir>` to the four `make_config.py` and clones the egoOmni instance.
- `egotaskqa_prep_h.sh`, `egotaskqa_h.sh`, `egotaskqa_score_h.sh`, `egotaskqa_chain.sh` — EgoTaskQA (see `../egotaskqa/`);
  `egoomni_complete_chain.sh` — egoOmni for the base and the colleague checkpoints + judging of the external baselines.
- **EgoToM-kit protocol re-test:** `nat_infer.py` (request-file runner: EgoSchema / EgoCross / EgoTaskQA; `nat_build.py`,
  `nat_to_lf.py`, `nat_status.py`, `nat_job_h.sh`, `nat_chain.sh`) and `nat_predict_lf.py` (drop-in for LLaMA-Factory predict on a
  sharegpt dataset: EgoAVU-Bench; `kit_job_h.sh`, `kit_chain.sh`, `kit_eab_rest_h.sh` + `split_kit_rest.py` / `finalize_kit_rest.py`
  for rebalancing a run's tail). qwen_omni_utils 0.0.9 + decord 0.6.0 are installed with `pip install --target` and put on
  `PYTHONPATH` over the LLaMA-Factory env (transformers 4.54: the kit's `dtype=` is `torch_dtype=` there). `make_kit_instances.py`,
  `prep_egoomni_proxy2fps*.py|sh` belong to the egoOmni part of the re-test, which was dropped (decord cannot decode the egoOmni
  source clips; `nat_predict_lf.py` keeps a PyAV fallback with the same frame selection).
- `eab_judge_h.sh --suffix SFX --shards K` — judge a run into its own `release_h/judge_h/results_hSFX` dirs; `summarize_all.py` —
  the three result tables of the top-level README.
