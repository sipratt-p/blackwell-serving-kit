#!/usr/bin/env bash
# SINGLE-GPU test of dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4 (135GB, W4A4 NVFP4, PLE FP8).
# PLE table offloaded to host RAM -> ~85GB of weights on ONE RTX PRO 6000. Experimental (2026-09-03).
# TRAP: TP1 defaults to the uniproc executor, which never spawns the PLE offload worker -> hangs
# forever in kernel warm-up (cuda-graph replay waiting on IPC). --distributed-executor-backend mp fixes it.
# Runs on port 8003 so it can coexist with :8001 in name, but it CANNOT coexist in VRAM:
# the TP2 build must be stopped first (docker stop qwen38-flash-next).
set -euo pipefail
MODELS_ROOT=${MODELS_ROOT:-/mnt/models}
MODEL_DIR=${MODEL_DIR:-${MODELS_ROOT}/Qwen3.8-Flash-Next-ABLITERATED-NVFP4-dealignai}
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-8003}
GPU=${GPU:-1}
MAXLEN=${MAXLEN:-32768}
GPUUTIL=${GPUUTIL:-0.95}
MAXSEQS=${MAXSEQS:-8}
SPEC_TOKENS=${SPEC_TOKENS:-0}      # MTP head is bf16 (+4.9GB); enable once base fits
MAXBATCH=${MAXBATCH:-8192}
SPEC_ARGS=()
[ "${SPEC_TOKENS}" != "0" ] && SPEC_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")

docker rm -f qwen38fn-ablit-nvfp4 2>/dev/null || true
exec docker run --rm --name qwen38fn-ablit-nvfp4 --init \
  --ipc=host --shm-size 32g --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -p ${PORT}:${PORT} \
  -v "${MODEL_DIR}":/model:ro \
  -v qwen38-jit-cache:/cache/jit \
  -v "${HERE}/../patches/qwen38fn_ple_layer.py":/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -e VLLM_PLE_FORCE_FP8=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=${GPU} \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  vllm/vllm-openai:qwen38-flash-next \
  /model \
  --served-model-name Qwen3.8-Flash-Next-ablit-nvfp4 \
  --host 0.0.0.0 --port ${PORT} \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization ${GPUUTIL} \
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAXSEQS} \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens ${MAXBATCH} \
  --trust-remote-code \
  "${SPEC_ARGS[@]}"
