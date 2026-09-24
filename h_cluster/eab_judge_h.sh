#!/usr/bin/env bash
# H rjob (4 GPU): official EgoAVU-Bench scoring of sharded LoRA runs — merge shards, assemble, official judge
# (Qwen3-235B-A22B-Instruct-2507 from gpfs2-shared-public, vLLM TP4), official captioning metrics, closed-ended accuracy,
# and a summary that also lists the earlier base / ckpt_sft / ckpt_epoch2 results.   usage: eab_judge_h.sh [--suffix SFX] [--shards K] TAG [TAG ...]
# --suffix SFX: judge only these TAGs into release_hSFX / judge_hSFX / results_hSFX (earlier H models are not re-judged; the summary
# also lists every model already judged in release_h / judge_h).
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
SFX=""; [ "${1:-}" = "--suffix" ] && { SFX=$2; shift 2; }
NSH=8; [ "${1:-}" = "--shards" ] && { NSH=$2; shift 2; }   # shard runs per tag (s{k}of{NSH})
exec > >(tee -a $LY/tools/h/logs/eab_judge_$(hostname).log) 2>&1
h_check_gpus 4 || exit 1
set -o pipefail
BI=/ai4good1-shared/liyu/egoavu/bench_infer; PVC=/ai4good1-shared/liyu/egoavu
[ -x /shared/egoavu/env/bin/python ] || tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /
EPY=/shared/egoavu/env/bin/python; JE=$LY/egoOmni_baselines/envs_h/judge; CE=$LY/egoOmni_baselines/envs_h/capeval
J235=$($JE/bin/python $LY/tools/h/pick_snapshot.py /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507) || exit 1
h_log "judge model: $J235"
REL=$BI/release_h$SFX; JD=$BI/judge_h$SFX; OUT=$BI/results_h$SFX; mkdir -p $REL $JD $OUT
for t in "$@"; do
  $EPY $LY/tools/h/merge_bench_shards.py $t $NSH && (cd /tmp && $EPY $BI/tools/collect_predictions.py $t egoavu_bench_eval | tail -c 400) || { h_log "merge/collect $t FAILED"; exit 1; }
done
(cd /tmp && $EPY $BI/tools/assemble_release.py $REL "$@" 2>&1 | grep -v Warning) || { h_log "assemble FAILED"; exit 1; }
h_log "official judge"
PATH=$JE/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  $JE/bin/python $BI/tools/judge_official.py $REL/predictions $JD $J235 4 > $BI/logs/judge_h$SFX.log 2>&1 || { h_log "judge FAILED (see logs/judge_h$SFX.log)"; exit 1; }
cat $JD/judge_scores.csv
h_log "captioning metrics"; cd /shared/egoavu/repo
for C in "Audio-Visual Segment Narration" "Audio-Visual Dense Narration"; do
  tag=$(echo "$C" | tr -cd "A-Za-z" | tr A-Z a-z)
  PATH=$CE/bin:$PATH JAVA_HOME=$CE $CE/bin/python evaluation/captioning_eval.py --json_dir $REL/predictions --output_csv $JD/caption_${tag}.csv --categories "$C" > $BI/logs/capeval_h${SFX}_${tag}.log 2>&1 || h_log "captioning $tag FAILED"
  cat $JD/caption_${tag}.csv
done
for t in "$@"; do python3 $BI/tools/closed_ended_acc.py $REL/predictions/$t.json > $JD/closed_ended_acc_$t.json; done
ARGS="base_qwen25omni7b=$BI/release_control/predictions/base_qwen25omni7b.json=$BI/judge_control ckpt_sft=$BI/release/predictions/ckpt_sft.json=$BI/judge ckpt_epoch2=$BI/release/predictions/ckpt_epoch2.json=$BI/judge"
[ -n "$SFX" ] && for p in $BI/release_h/predictions/*.json; do ARGS="$ARGS $(basename $p .json)=$p=$BI/judge_h"; done
for t in "$@"; do ARGS="$ARGS $t=$REL/predictions/$t.json=$JD"; done
$EPY $BI/tools/summarize.py $OUT $ARGS && cat $OUT/SUMMARY.md
h_log "EAB judge DONE"
