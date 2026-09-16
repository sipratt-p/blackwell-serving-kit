#!/bin/bash
# free-gpus.sh — show what holds the GPUs; with --stop, gracefully stop KNOWN servers.
# NEVER kills unknown processes: another operator or job may own them. Unknowns are listed.
set -u
KNOWN_CONTAINERS="qwen38-flash-next qwen38fn-ablit-nvfp4 v4-flash glm53-tr3"
echo "== GPU state"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== GPU processes"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sort -u
echo "== Known model containers running"
docker ps --format '{{.Names}}' | grep -xE "$(echo $KNOWN_CONTAINERS | tr ' ' '|')" || echo "(none)"
if [ "${1:-}" = "--stop" ]; then
  for c in $KNOWN_CONTAINERS; do
    docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "stopping $c…"; docker stop "$c"; }
  done
  sleep 3
  echo "== after stop"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  LEFT=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | sort -u)
  [ -n "$LEFT" ] && { echo "⚠ processes still on GPU (NOT killed — check whose they are):"; echo "$LEFT"; }
  # headless-chrome GPU leak check: a debug-port Chrome pins a GPU context and wedges NCCL
  pgrep -af 'chrome.*headless' | head -3 && echo "⚠ headless chrome above may be leaking GPU — kill its tree if yours."
else
  echo; echo "run 'free-gpus.sh --stop' to gracefully stop the known model servers listed above"
fi
