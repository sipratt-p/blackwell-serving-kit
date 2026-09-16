#!/bin/bash
# Ornith-1.5-397B NVFP4 on 2x RTX PRO 6000 — vLLM TP2 + selective UVA expert-offload.
# Weights: 222GB NVFP4; 2×98GB VRAM → offload expert tensors to pinned RAM,
# zero-copy read only the ~11 active experts/layer/token (A~15B of 397B).
# NEEDS BOTH GPUs FREE (stop ds4-flash-dspark / ornith-35b first) + free RAM ≥ ~80GB.
# Tunables: OFFLOAD_GB (per-GPU GiB of experts to CPU; raise if OOM at load),
#           UTIL (gpu-memory-utilization), MAXLEN, PORT.
set -eu
OFFLOAD_GB=${OFFLOAD_GB:-60}   # 35 OOMed at init: each rank held ~94GB before load (2026-08-19)
UTIL=${UTIL:-0.88}
MAXLEN=${MAXLEN:-65536}
PORT=${PORT:-8011}

docker run -d --name ornith-397b --gpus all --shm-size 64g --network host --ipc=host \
  --ulimit memlock=-1:-1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_P2P_DISABLE=1 \
  -v "${MODEL_DIR:-${MODELS_ROOT:-/mnt/models}/Ornith-1.5-397B-NVFP4}":/model:ro \
  vllm/vllm-openai:latest \
  --model /model --served-model-name ornith-397b --port "$PORT" \
  --tensor-parallel-size 2 --max-model-len "$MAXLEN" --max-num-seqs 16 \
  --disable-custom-all-reduce \
  --gpu-memory-utilization "$UTIL" \
  --offload-backend uva \
  --cpu-offload-gb "$OFFLOAD_GB" \
  --cpu-offload-params experts \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
echo "launched ornith-397b on :$PORT — watch: docker logs -f ornith-397b"
# Escalation ladder if this doesn't come up / is slow:
#  1. OOM during load/warmup      -> OFFLOAD_GB=45 (then 55), or UTIL=0.88
#  2. cpu-offload-params rejected -> try --cpu-offload-params experts (bare), or drop
#     the filter (non-selective UVA up to OFFLOAD_GB — experts dominate anyway)
#  3. GDN cache ValueError        -> lower --max-num-seqs (35B needed <=128; we use 16)
#  4. slow decode (<10 t/s)       -> try --enable-expert-parallel (EP2: whole experts
#     per rank, halves offload need), and/or batch benches to amortize expert reads
