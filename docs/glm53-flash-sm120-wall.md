# GLM-5.3-Flash NVFP4 on SM120: two walls (measured 2026-08-26)

`zai-org/GLM-5.3-Flash` ships already in FP8 at 328 GB, so unlike Qwen3.8-Flash-Next there is no easy
halving. The only 4-bit candidate at the time, `LibertAIDAI/GLM-5.3-Flash-NVFP4`, is 194.7 GB. It needs the
dedicated `vllm/vllm-openai:glm53-flash` image (a pre-merge build of vLLM PR #53906); neither `:latest`
nor the Qwen image registers `Glm5Next*`.

## Wall 1: VRAM

Weights load fine (83 s, TP2, `modelopt_fp4` auto-detected). Then:

| config | result |
|---|---|
| util 0.97, 65K ctx, fp8 KV | OOM in `profile_run`, short by 330 MB |
| `--max-num-seqs 1 --max-num-batched-tokens 1024 --enforce-eager` | passes profiling; 0.29 GiB left for KV = **9,984 tokens** max context |
| desktop stopped (GPU 0 down to 34 MiB), util 0.985 | 0.98 GiB KV = 88,134 tokens, then wall 2 |

Measured KV cost is about 31 KB per token, not the 11 KB that `kv_lora_rank=512` arithmetic predicts. Use
the measured figure for planning. Serving 262K would need roughly 8 GB of KV headroom.

Why 195 GB and not 165: the quant's ignore list has 32 patterns covering the entire attention stack
(`q_proj`, `k_proj`, `v_proj`, `q_a_proj`, `kv_a_proj_with_mqa`, `q_b_proj`, the `f_*`/`g_*` projections,
fused QKV) plus `embed_tokens` and `lm_head`. Sampled dtypes across 20 of 120 shards: U8 (packed NVFP4)
73.8%, BF16 16.9%, F8_E4M3 9.2%. About 33 GB never left BF16. A quant that also takes the attention
projections would land near 165 GB and fit with real room.

## Wall 2: kernel support (the real blocker)

With the VRAM freed, warm-up dies in `compile_or_warm_up_model`:

```
RuntimeError: concat_and_cache_mla, csrc/libtorch_stable/cache_kernels.cu:866, pe_dim must be 64 for fp8_ds_mla
```

GLM-5.3-Flash's 11 sparse-MLA layers are NoPE (`qk_rope_head_dim: 0`, `qk_nope_head_dim 256`,
`kv_lora_rank 512`). vLLM's only backend for `use_mla + use_sparse` on SM120 is
`FLASHINFER_MLA_SPARSE_SM120`, which forces the DeepSeek `fp8_ds_mla` KV layout with its 64-wide RoPE slot.
`--kv-cache-dtype auto` does not change it, and `VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA_SPARSE` or
`FLASHMLA_SPARSE` are silently ignored (the log always says `potential backends:
['FLASHINFER_MLA_SPARSE_SM120']`). The log also warns that no MLA prefill backend supports the model, so
prefill would have used the top-k MQA path only.

Unblock condition: an SM120 sparse-MLA path that accepts `pe_dim=0` (flashinfer PR #4791 "native NoPE
sparse MLA on SM120" appeared as a draft the next day; vLLM PR #54110 fixes a second low-shared-memory
crash) AND a quant near 165 GB that also quantises the attention stack.

## What ran instead

`brandonmusic/GLM-5.3-Flash-tr3-4bpw` (176 GB: routed experts in EXL3/TR3 4-bit, attention in the official
FP8) ships its own vLLM fork as a Docker image (B12X lineage, not stock vLLM). It runs on two RTX PRO 6000s.
See `glm53-tr3/` for the launch adaptation and the thinking-budget proxy it needs.

## Desktop lever

Stopping the display manager takes GPU 0 from 0.5 to 1.3 GB down to 34 MiB. util 0.99 fails on startup
("Free memory 94.01/94.97 GiB < desired 94.02"); 0.985 is the ceiling. Only stop the display manager from a
session that does not live under it (a terminal hosted by the desktop session killed itself doing this).
GPU 0 always carries the desktop and always OOMs first on a tight TP2 fit.
