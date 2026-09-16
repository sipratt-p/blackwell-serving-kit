#!/usr/bin/env bash
# Ornith-1.5-35B-A3B bf16 on ONE RTX PRO 6000 with stock vLLM (vllm/vllm-openai:latest, 0.27.1 at the time).
# Measured: 179.5 tok/s single-stream; near-linear concurrency to 3,444 tok/s aggregate at C=128 (not saturated).
# Quality: 489/500 on the 5-task coding suite (direct API, temperature 0), the best direct-API result we measured.
# Traps (each cost a failed launch):
#   - default --max-num-seqs (1024) exceeds the GatedDeltaNet cache blocks -> startup ValueError. Keep it <= 128 here.
#   - tool parser is qwen3_xml (qwen3_engine does not exist; valid names: qwen3_coder, qwen3_xml, hermes, ...).
#   - GGUF quants of this model write measurably worse code (398/500 vs 489 bf16, same scoping bug in both sizes).
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-${MODELS_ROOT:-/mnt/models}/Ornith-1.5-35B-A3B}
GPU=${GPU:-1}            # single GPU; leaves the other one free
PORT=${PORT:-8010}
MAXLEN=${MAXLEN:-131072}
MAXSEQS=${MAXSEQS:-128}
GPUUTIL=${GPUUTIL:-0.92}

docker rm -f ornith-35b 2>/dev/null || true
docker run -d --name ornith-35b --gpus "\"device=${GPU}\"" --shm-size 32g --network host \
  -v "${MODEL_DIR}":/model:ro vllm/vllm-openai:latest \
  --model /model --served-model-name ornith-35b-a3b --max-model-len "${MAXLEN}" \
  --max-num-seqs "${MAXSEQS}" --gpu-memory-utilization "${GPUUTIL}" --port "${PORT}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
echo "launched ornith-35b on :${PORT}; health: curl -s localhost:${PORT}/v1/models"
