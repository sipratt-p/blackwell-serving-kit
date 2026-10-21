#!/bin/bash
# GLM-5.3-Flash-tr3-4bpw on 2x RTX PRO 6000 — adaptation of brandonmusic's serve-glm53-sm120-tp2.sh (v75 image).
# Original by Local Inference Lab, Inc. (published by Brandon M. Music): https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw
# Project home: https://local-inference-lab.ai/ . Licence: Local Inference Lab Attribution License 1.0, see LICENSE-upstream.
# Differences from upstream defaults (why):
#   CACHE=fp8_ds_mla  → FLASHINFER_MLA_SPARSE_SM120 instead of B12X_MLA_SPARSE (Xid 31 crashes on DSV4 with that backend)
#   MAX_MODEL_LEN=131072 / MAX_NUM_SEQS=4 → multi-agent use instead of the author's 500K single-seq profile
#   NCCL_P2P_DISABLE=1 → kernel 7.0.0-30 GPU P2P is broken (NCCL hangs / silent corruption)
#   B12X_DCP_A2A=0 (default here, author=1) → the "direct DCP A2A" path does GPU↔GPU direct copies; on our kernel it timed out
#     and killed CUDA-graph capture with Xid 43 (2026-08-28). 0 = NCCL all-to-all (host-staged with P2P off).
#   DIRECT_DCP_A2A=0 → the "direct DCP A2A timeout" spam persisted with B12X_DCP_A2A=0 alone (attempt 2, same Xid 43);
#     VLLM_USE_DIRECT_DCP_A2A is the switch for that path (author script only sets it when DIRECT_DCP_A2A is non-empty).
#   GPU_MEMORY_UTILIZATION=0.97 → GPU0 hosts the desktop + our 350 W cap; 0.986 leaves no slack
#   --enable-auto-tool-choice --tool-call-parser (2026-08-28): agent clients send tool_choice=auto → 400 without these; parser from the image's glm47_moe_tool_parser.py
# Usage: ./launch.sh            (foreground, logs to stdout)   |  nohup ./launch.sh > server.log 2>&1 &
set -euo pipefail
IMAGE="verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v75@sha256:4605c420cc589be9fd15fc759c7f7c2a6035dab48f885c9466eb2233527bca64"
MODEL="${MODEL:-${MODELS_ROOT:-/mnt/models}/GLM-5.3-Flash-tr3-4bpw}"
CACHE="${CACHE:-fp8_ds_mla}"
DCP="${DCP:-2}"
PORT="${PORT:-8022}"   # vLLM itself; the thinking-budget proxy owns the public port :8012 (some agent clients cache provider URLs, so keep the public port fixed)
GPU_DEVICES="${GPU_DEVICES:-0,1}"
NAME="${NAME:-glm53-tr3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
MTP_TOKENS="${MTP_TOKENS:-3}"
P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
case "${CACHE}" in
  nvfp4_ds_mla) ATTENTION_BACKEND=B12X_MLA_SPARSE ;;
  fp8_ds_mla)   ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_SM120 ;;
  *) echo "CACHE must be nvfp4_ds_mla or fp8_ds_mla" >&2; exit 2 ;;
esac
EXTRA_ARGS=()
[[ "${MTP_TOKENS}" != "0" ]] && EXTRA_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"draft_sample_method\":\"probabilistic\"}")
[[ "${ENFORCE_EAGER:-0}" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)
mkdir -p ~/.cache/glm53-exl3-k4
docker rm -f "${NAME}" >/dev/null 2>&1 || true
exec docker run --name "${NAME}" \
  --init --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g \
  -p "${PORT}:${PORT}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_B12X_GLM_NOPE_NVFP4=1 \
  -e "VLLM_USE_B12X_DCP_A2A=${B12X_DCP_A2A:-0}" \
  -e "VLLM_USE_DIRECT_DCP_A2A=${DIRECT_DCP_A2A:-0}" \
  -e VLLM_B12X_MLA_CKV_GATHER=0 \
  -e OMP_NUM_THREADS=2 \
  -e NCCL_IB_DISABLE=1 \
  -e "NCCL_P2P_DISABLE=${P2P_DISABLE}" \
  -e NCCL_P2P_LEVEL=4 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -v "${MODEL}:/model:ro" \
  -v "${HOME}/.cache/glm53-exl3-k4:/root/.cache" \
  --entrypoint vllm \
  "${IMAGE}" serve /model \
  --served-model-name GLM-5.3-Flash-EXL3-4bpw \
  --host 0.0.0.0 --port "${PORT}" \
  --language-model-only \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --decode-context-parallel-size "${DCP}" \
  --dcp-comm-backend a2a \
  --dtype bfloat16 \
  --load-format safetensors \
  --moe-backend b12x \
  --attention-backend "${ATTENTION_BACKEND}" \
  --kv-cache-dtype "${CACHE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --generation-config /model \
  --reasoning-parser glm45 \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-glm47}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
