#!/bin/bash
# On the A100 (tnr-2): pull every paired teacher store and its verso outputs from the two A6000s into ~/teacher
# (same layout as the desk), then keep refreshing every 30 min. Instance-to-instance, no desk upload.
mkdir -p ~/teacher
while true; do
  for h in tnr-0 tnr-1; do
    rsync -a --exclude "*.log" --exclude "*.partial" $h:out/ ~/teacher/ 2>>~/a100_pull.err
  done
  echo "$(date) pulled: $(ls -d ~/teacher/boxes*_vraw/*.zarr 2>/dev/null | wc -l) raw verso stores, $(du -sh ~/teacher | cut -f1)"
  [ "${ONCE:-0}" = 1 ] && break
  sleep 1800
done
