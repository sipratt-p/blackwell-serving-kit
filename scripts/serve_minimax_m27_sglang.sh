#!/usr/bin/env bash
# MiniMax M2.7 NVFP4 (lukealonso/MiniMax-M2.7-NVFP4, 131 GB) on 2x RTX PRO 6000 via SGLang + b12x Blackwell MoE kernels.
# Image: a local build on top of lmsysorg/sglang:latest with SM120 patches, b12x==0.8.7 and transformers==4.57.1
# (that exact transformers version is load-bearing: other versions produced errant output in multi-step tool workflows).
# Measured (2026-04-15): 120-127 tok/s single warm, 97-104 tok/s sustained in an agent session (89% KV hit rate),
# 408 tok/s aggregate at 16 concurrent, TTFT 130-215 ms. Startup ~5.5 min (torch.compile); Triton SM120 autotune
# warnings ("No valid triton configs") are normal.
# Traps: --tool-call-parser minimax-m2 is REQUIRED (0% tool calls without it); kv dtype must be fp8_e4m3, not fp8
# (FP8 KV costs decode: ~97 t/s sustained vs the 178-196 reported with BF16 KV, kept for memory); re-download the
# shards if output is gibberish (a bad first download did exactly that); SGLang issue #16057: the minimax-m2 parser
# fails on union-typed tool parameters (str | null), so write tool schemas with concrete types.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-${MODELS_ROOT:-/mnt/models}/MiniMax-M2.7-NVFP4}
IMAGE=${IMAGE:-sglang-b12x-latest:latest}
PORT=${PORT:-30000}

docker rm -f minimax-b12x 2>/dev/null || true
docker run -d --name minimax-b12x --gpus all --shm-size 64g --ipc=host \
  -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  -p "${PORT}:30000" \
  -v "${MODEL_DIR}":/models/MiniMax-M2.7-NVFP4:ro \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path /models/MiniMax-M2.7-NVFP4 \
    --tp 2 --trust-remote-code \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8_e4m3 \
    --moe-runner-backend b12x \
    --fp4-gemm-backend b12x \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --enable-pcie-oneshot-allreduce \
    --enable-torch-compile \
    --host 0.0.0.0 --port 30000
echo "launched minimax-b12x on :${PORT}; health: curl -s localhost:${PORT}/v1/models"
