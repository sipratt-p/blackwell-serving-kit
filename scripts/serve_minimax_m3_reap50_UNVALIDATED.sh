#!/bin/bash
# MiniMax-M3 NVFP4 REAP50 (sparkarena/Minimax-M3-v0-NVFP4-REAP50, 129 GB) on 2x RTX PRO 6000, SGLang.
# STATUS: STAGED, NOT YET LAUNCHED on this hardware. Kept because the pinning decisions are the useful part.
# Recipe adapted from ressl/MiniMax-M3-uncensored-NVFP4 (validated on 4x RTX PRO 6000 Blackwell, TP4):
# same pinned SGLang image, same two clamped-SwiGLU patches, TP2 here because REAP50 fits 2 cards.
# Needs BOTH GPUs free. The two sglang_patch files come from the ressl recipe and ship with that model repo.
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT:-/mnt/models}/MiniMax-M3-NVFP4-REAP50}"
IMAGE="lmsysorg/sglang@sha256:8cc6e6f90bf803e9817800b679173d0b526f2b42b2c61b7ecafecdadb610eb55"
PORT="${PORT:-30000}"
CTX="${CTX:-65536}"
NAME="${NAME:-minimax-m3-reap50}"

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --runtime=nvidia --gpus all --ipc=host --shm-size 32g \
  -e NCCL_P2P_DISABLE=1 \
  -v "$MODEL_DIR:/model:ro" \
  -v "$MODEL_DIR/sglang_patch/modelopt_quant.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py:ro" \
  -v "$MODEL_DIR/sglang_patch/flashinfer_trtllm.py:/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py:ro" \
  -p "$PORT:30000" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name minimax-m3-reap50 \
    --host 0.0.0.0 --port 30000 \
    --tp-size 2 \
    --quantization modelopt_fp4 \
    --trust-remote-code \
    --dtype auto \
    --context-length "$CTX" \
    --mem-fraction-static 0.88 \
    --max-running-requests 8 \
    --chunked-prefill-size 16384 \
    --page-size 128 \
    --tool-call-parser minimax-m3 \
    --reasoning-parser minimax-m3 \
    --moe-runner-backend flashinfer_cutlass \
    --fp4-gemm-backend flashinfer_cutlass \
    --attention-backend flashinfer \
    --disable-flashinfer-autotune \
    --disable-prefill-cuda-graph \
    --disable-shared-experts-fusion \
    --disable-custom-all-reduce

echo "started $NAME; follow with: docker logs -f $NAME | grep -v -i triton"
echo "health: curl -s localhost:$PORT/v1/models"
