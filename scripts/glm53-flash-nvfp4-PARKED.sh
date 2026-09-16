#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 on 2x RTX PRO 6000 (sm_120), TP2.
# STATUS: PARKED. Documents the attempt; see docs/glm53-flash-sm120-wall.md for why it cannot serve usefully.
#
# TIGHT FIT: 194.7GB of weights (~181 GiB) against ~191 GiB of usable VRAM.
# ~17% of this checkpoint is still BF16 (the whole attention stack is in the
# quant ignore list), which is why it is 195GB and not ~165GB.
# Levers if it OOMs, in order: lower MAXLEN, fp8 KV, then CPU_OFFLOAD_GB.
#
# NCCL_P2P_DISABLE=1 is MANDATORY on kernel >=7.0.0-30 (P2P collectives hang /
# corrupt). See docs/kernel-p2p-regression.md.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-${MODELS_ROOT:-/mnt/models}/GLM-5.3-Flash-NVFP4}
PORT=${PORT:-8002}
MAXLEN=${MAXLEN:-16384}
GPUUTIL=${GPUUTIL:-0.97}
MAXSEQS=${MAXSEQS:-1}
KVDTYPE=${KVDTYPE:-fp8}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}

EXTRA=()
[ "${CPU_OFFLOAD_GB}" != "0" ] && EXTRA+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")

docker rm -f glm53-flash 2>/dev/null || true

exec docker run --rm --name glm53-flash --init \
  --ipc=host --shm-size 32g --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -p ${PORT}:${PORT} \
  -v "${MODEL_DIR}":/model:ro \
  -v glm53-jit-cache:/cache/jit \
  -e HF_HUB_OFFLINE=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  vllm/vllm-openai:glm53-flash \
  /model \
  --served-model-name GLM-5.3-Flash \
  --host 0.0.0.0 --port ${PORT} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization ${GPUUTIL} \
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAXSEQS} \
  --max-num-batched-tokens ${MAXBATCH:-1024} \
  --enforce-eager \
  --kv-cache-dtype ${KVDTYPE} \
  --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune \
  --trust-remote-code \
  "${EXTRA[@]}"
