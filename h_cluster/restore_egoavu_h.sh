#!/usr/bin/env bash
# Restore s3://safevlagent/liyu/egoavu/ onto H gpfs EXCEPT the dataset media (clips/, proxy262k/, proxy/),
# which stay in object storage and are geesefs-mounted inside jobs (see tools/h/h_prelude.sh).
set -uo pipefail
L=/mnt/shared-storage-user/ai4good1-share/liyu; DST=$L/egoavu
export AWS_SHARED_CREDENTIALS_FILE=$L/.aws/credentials AWS_CONFIG_FILE=$L/.aws/config
AWS="/root/miniconda3/bin/aws --profile h-hdd2 --endpoint-url http://hdd2.h.pjlab.org.cn:8060"
/root/miniconda3/bin/aws configure set s3.max_concurrent_requests 32 --profile h-hdd2
/root/miniconda3/bin/aws configure set s3.multipart_chunksize 64MB --profile h-hdd2
mkdir -p $DST
echo "[restore] $(date -Is) start -> $DST"
$AWS s3 sync s3://safevlagent/liyu/egoavu/ $DST/ --only-show-errors \
     --exclude "clips/*" --exclude "proxy262k/*" --exclude "proxy/*"
rc=$?
echo "[restore] $(date -Is) sync exit=$rc"
# media dirs -> the in-job geesefs mount (dangling on openvla, resolve inside rjobs)
for d in clips proxy262k proxy; do [ -e $DST/$d ] || ln -s /mnt/egoavu_s3/$d $DST/$d; done
du -sh $DST 2>/dev/null
echo "[restore] $(date -Is) done"
