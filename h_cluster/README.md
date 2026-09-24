# H-cluster pipeline (2026-09-23/24)

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
- Lessons: CPU-only jobs on GPU nodes are capped at 64 CPUs cumulative; vLLM 0.10.1 needs transformers 4.x (4.56.2 used).
