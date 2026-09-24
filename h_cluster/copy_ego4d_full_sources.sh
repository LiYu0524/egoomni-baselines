#!/usr/bin/env bash
# Stream the 13 full Ego4D source videos of the restored_v3 "full_source" items from the old bucket (h-oss, read-only use)
# into s3://safevlagent/liyu/egoOmni/ego4d_full/ (h-hdd2). Nothing is staged on disk.
L=/mnt/shared-storage-user/ai4good1-share/liyu
AWS=/root/miniconda3/bin/aws
while read v size; do
  dst=s3://safevlagent/liyu/egoOmni/ego4d_full/$v.mp4
  have=$(AWS_SHARED_CREDENTIALS_FILE=$L/.aws/credentials AWS_CONFIG_FILE=$L/.aws/config $AWS --profile h-hdd2 --endpoint-url http://hdd2.h.pjlab.org.cn:8060 s3api head-object --bucket safevlagent --key liyu/egoOmni/ego4d_full/$v.mp4 --query ContentLength --output text 2>/dev/null)
  [ "$have" = "$size" ] && { echo "[copy] skip $v"; continue; }
  $AWS --profile h-oss --endpoint-url http://hdd1.h.pjlab.org.cn:8060 s3 cp s3://ai4good-h-hdd-1/egoavu/raw/ego4d/v2_1/full_scale/$v.mp4 - \
  | AWS_SHARED_CREDENTIALS_FILE=$L/.aws/credentials AWS_CONFIG_FILE=$L/.aws/config $AWS --profile h-hdd2 --endpoint-url http://hdd2.h.pjlab.org.cn:8060 s3 cp - $dst --expected-size $size --only-show-errors
  got=$(AWS_SHARED_CREDENTIALS_FILE=$L/.aws/credentials AWS_CONFIG_FILE=$L/.aws/config $AWS --profile h-hdd2 --endpoint-url http://hdd2.h.pjlab.org.cn:8060 s3api head-object --bucket safevlagent --key liyu/egoOmni/ego4d_full/$v.mp4 --query ContentLength --output text 2>/dev/null)
  echo "[copy] $v src=$size dst=$got $([ "$got" = "$size" ] && echo OK || echo MISMATCH)"
done < /tmp/fs_sizes.txt
echo "[copy] $(date -Is) done"
