# GLM-5.3-Flash TR3-4bpw on two RTX PRO 6000s

Weights: `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (176 GB; routed experts EXL3/TR3 4-bit, attention in the
official FP8). Runtime: the author's closed vLLM fork image
`verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v75` (B12X lineage, not stock vLLM). Author: Local Inference Lab, Inc. (published by Brandon M. Music); author docs and scripts:
https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw (project home: https://local-inference-lab.ai/). `launch.sh` is our adaptation of their
`serve-glm53-sm120-tp2.sh`; licence text in `LICENSE-upstream`.

## Sequence

1. Both GPUs free (`nvidia-smi` under 2 GB each).
2. `nohup ./launch.sh > server.log 2>&1 &` (about 12 minutes to "Application startup complete": weights 45 s,
   TileLang and CUDA-graph capture the rest).
3. `curl -s localhost:8022/v1/models`, then `bash restart_proxies.sh` (thinking-budget proxy on :8012 in
   front of vLLM on :8022; see `../docs/thinking-budget.md`).
4. Speed: `python3 bench_speed.py`.

## What worked (attempt 4, 2026-08-28)

`launch.sh` defaults: `CACHE=fp8_ds_mla` (FLASHINFER_MLA_SPARSE_SM120), `DCP=2`, `MAX_MODEL_LEN=131072`,
`MAX_NUM_SEQS=4`, util 0.97, MTP3, `NCCL_P2P_DISABLE=1`, `VLLM_USE_B12X_DCP_A2A=0`,
`VLLM_USE_DIRECT_DCP_A2A=0`, `--enable-auto-tool-choice --tool-call-parser glm47`.

Measured: KV pool 602,112 tokens (4.59 x 128K); 108 to 117 t/s single stream; 4 concurrent 105 to 185 t/s
aggregate. The author reports 145 t/s on the same GPUs with NVFP4 KV, a memory overclock and 600 W; our
GPU 0 is capped at 350 W for thermals.

Agentic hard suite with an 8K thinking budget: 389 / 359 / 369, mean 372/400, equal to Qwen3.8-Flash-Next
FP8+MTP4 on the same box (380). Without a budget: 4/100 (65,536 tokens of reasoning, no tool call).

## Attempt log

1. Author defaults (direct DCP A2A on): "direct DCP A2A timeout" spam, then Xid 43 and a CUDA unspecified
   launch failure during CUDA-graph capture. Direct GPU-to-GPU copies on a P2P-broken kernel.
2. `B12X_DCP_A2A=0` alone: same. The "direct" switch is separate.
3. Both A2A switches off: up. Every tool-using request then failed with HTTP 400 "auto tool choice requires
   --enable-auto-tool-choice and --tool-call-parser". The author's script omits them.
4. Parser added. `glm47_moe` is the file name, not the flag; registered names are `glm45 glm47 hermes
   qwen3_coder qwen3_xml`. One more relaunch.
5. No thinking budget in the runtime: first agent task produced 231K characters of reasoning and zero tool
   calls. `think_budget_proxy.py` was written for this.
6. Context-overflow 400s in long tool loops: the GLM template re-renders every prior turn's reasoning. The
   proxy now strips history reasoning and retries with a fitted `max_tokens`.

## Fallbacks

- NCCL hang at init: confirm `NCCL_P2P_DISABLE=1` landed (`docker inspect glm53-tr3 | grep NCCL`); try
  `NCCL_P2P_LEVEL=0`.
- `fp8_ds_mla` rejected ("pe_dim must be 64"): `CACHE=nvfp4_ds_mla` (author default, B12X_MLA_SPARSE; watch
  `dmesg` for Xid 31).
- OOM at KV allocation: `MAX_NUM_SEQS=2` or `GPU_MEMORY_UTILIZATION=0.98` with the desktop off GPU 0.
- MTP garbage: `MTP_TOKENS=0 ENFORCE_EAGER=1 ./launch.sh` (the author's correctness profile).

The image, pulled by digest, shows tag `<none>` in `docker images`; that is fine.
