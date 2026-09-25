#!/bin/bash
# rjob entry: EgoToM clips (official step 1 for hdd1 sources + official step 2 for every clip). No internet needed.
LY=/mnt/shared-storage-user/ai4good1-share/liyu; D=$LY/tools/egoclips
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY PYTHONPATH PYTHONHOME
export AWS_SHARED_CREDENTIALS_FILE=$LY/.aws/credentials AWS_CONFIG_FILE=$LY/.aws/config EGOCLIP_TMP=/tmp
cd $D && exec $LY/tools/egoclip_env/bin/python cut_clips.py egotom --workers ${WORKERS:-12} --proxy-workers 48
