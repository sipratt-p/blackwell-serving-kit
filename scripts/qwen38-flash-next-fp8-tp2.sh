#!/usr/bin/env bash
# Qwen3.8-Flash-Next-FP8 on 2x RTX PRO 6000 Blackwell (sm_120), TP2.
# 186GB FP8 checkpoint; the 51B n-gram (PLE) embedding table is offloaded to host RAM
# via VLLM_PLE_CPU_OFFLOAD=1, leaving ~122GB of weights on GPU (~61GB/GPU).
#
# NCCL_P2P_DISABLE=1 is mandatory on Linux 7.0.0-30 (GPU P2P regression: collectives
# hang, cross-GPU copies silently corrupt). See docs/kernel-p2p-regression.md.
#
# CAP_SYS_PTRACE + seccomp=unconfined are MANDATORY when VLLM_PLE_CPU_OFFLOAD=1:
# the PLE offload worker hands the n-gram tensor to the TP workers via CUDA IPC
# (torch UntypedStorage._new_shared_cuda), which calls pidfd_getfd(2). Docker's
# default profile blocks it -> "RuntimeError: pidfd_getfd: Operation not permitted".
set -euo pipefail

# DEFAULT = the abliterated build (2026-08-27). Refusal gate 0/12, functional quality
# identical to the censored FP8 (400/400 functional, v2 suite). Censored original:
#   MODEL_DIR=/mnt/models/Qwen3.8-Flash-Next-FP8 bash qwen38-flash-next-fp8-tp2.sh
MODELS_ROOT=${MODELS_ROOT:-/mnt/models}
MODEL_DIR=${MODEL_DIR:-${MODELS_ROOT}/Qwen3.8-Flash-Next-Uncensored-FP8}
PORT=${PORT:-8001}
MAXLEN=${MAXLEN:-262144}
GPUUTIL=${GPUUTIL:-0.92}   # 0.94 fits but leaves <700MB on GPU 0 with the desktop up
MAXSEQS=${MAXSEQS:-32}
# MTP speculative decode: 2.1x decode (114 -> 234 t/s), quality-neutral (functional 400/400 x3 runs).
# TRAP: only nst 1-4 and 9-12 boot. 5-8 die with "QSA ring capacity 12 must divide the attention
# block size 16" (qsa_cache.py: capacity = compress_ratio*ceil((cr+nst)/cr), cr=4, block=16).
# Disable with SPEC_TOKENS=0. MTP costs prefill; MAXBATCH 16384 buys it back (9.4k vs 7.6k tok/s).
SPEC_TOKENS=${SPEC_TOKENS:-4}
MAXBATCH=${MAXBATCH:-16384}
SPEC_ARGS=()
[ "${SPEC_TOKENS}" != "0" ] && SPEC_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")

docker rm -f qwen38-flash-next 2>/dev/null || true

exec docker run --rm --name qwen38-flash-next --init \
  --ipc=host --shm-size 32g --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -p ${PORT}:${PORT} \
  -v "${MODEL_DIR}":/model:ro \
  -v qwen38-jit-cache:/cache/jit \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
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
  vllm/vllm-openai:qwen38-flash-next \
  /model \
  --served-model-name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port ${PORT} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization ${GPUUTIL} \
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAXSEQS} \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-batched-tokens ${MAXBATCH} \
  --trust-remote-code \
  "${SPEC_ARGS[@]}"
