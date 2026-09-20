#!/usr/bin/env bash
# usage: download_models.sh <group>    ms7b | ms72b | hf
#   ms*  -> ModelScope direct (~45MB/s, no proxy)
#   hf   -> huggingface.co via the institutional proxy (SALMONN LoRA adapters; not on ModelScope)
set -u
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   # env.sh sets offline for runtime; downloads must go online
DL=$EGO_ROOT/miniconda3/envs/dl/bin
mark_done() { touch "$1/.DOWNLOAD_COMPLETE"; }
retry() {  # retry <dest> <cmd...>
  local dest=$1; shift; local n=0
  [ -f "$dest/.DOWNLOAD_COMPLETE" ] && { echo "[$(date -Is)] skip $dest (complete)"; return 0; }
  mkdir -p "$dest"
  while :; do n=$((n+1)); echo "[$(date -Is)] $dest attempt=$n size=$(du -sh "$dest" | cut -f1)"
    "$@" && { echo "[$(date -Is)] $dest DONE"; mark_done "$dest"; return 0; }
    echo "[$(date -Is)] $dest failed rc=$?; retry in 15s" >&2
    find "$dest" -type f \( -name "*.incomplete" -o -name "*.lock" \) -delete 2>/dev/null; sleep 15
  done
}
ms() { local repo=$1; shift; local dest=$EGO_ROOT/models/${repo#*/}
       retry "$dest" "$DL/modelscope" download --model "$repo" --local_dir "$dest" "$@"; }
hfdl() { local repo=$1; local dest=$EGO_ROOT/models/${repo#*/}
        retry "$dest" env HF_ENDPOINT=https://huggingface.co "$DL/hf" download "$repo" --local-dir "$dest" --max-workers 8 --format quiet; }
case "$1" in
  ms7b)  ms DAMO-NLP-SG/VideoLLaMA2.1-7B-AV
         ms google/siglip-so400m-patch14-384
         ms Qwen/Qwen2.5-VL-7B-Instruct
         ms openai-mirror/whisper-large-v3 --exclude "*.fp32*" "*.msgpack" "*.bin" "*.h5" ;;
  ms72b) ms Qwen/Qwen2.5-VL-72B-Instruct ;;
  minicpmo) ms OpenBMB/MiniCPM-o-2_6 ;;
  hf)    source $EGO_ROOT/.proxy_env
         hfdl tsinghua-ee/video-SALMONN-2_plus_7B
         hfdl tsinghua-ee/video-SALMONN-2_plus_72B ;;
  hfalign7b)  source $EGO_ROOT/.proxy_env; hfdl tsinghua-ee/video_SALMONN2plus_7B_audioAlign ;;    # base the LoRA needs (trained q_tokens live here)
  hfalign72b) source $EGO_ROOT/.proxy_env; hfdl tsinghua-ee/video_SALMONN2plus_72B_audioAlign ;;
  hffull7b)   source $EGO_ROOT/.proxy_env; hfdl tsinghua-ee/video-SALMONN2_plus_7B_full ;;     # official merged release = eval model (audioAlign+LoRA differs in q_tokens)
  hffull72b)  source $EGO_ROOT/.proxy_env; hfdl tsinghua-ee/video-SALMONN2_plus_72B_full ;;
  *) echo "unknown group $1"; exit 1 ;;
esac
echo "[$(date -Is)] group $1 ALL DONE"
