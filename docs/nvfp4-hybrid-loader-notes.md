# NVFP4 checkpoints of hybrid (GatedDeltaNet + attention) models: loader notes (April 2026)

Case: `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4` (a Qwen3.5-hybrid architecture,
`Qwen3_5ForConditionalGeneration`) on one RTX PRO 6000.

## What is actually in the checkpoint

| layer kind | storage | naming |
|---|---|---|
| `linear_attn.in_proj_qkv`, `_z`, `_b`, `_a`, `out_proj`, `conv1d`, `dt_bias`, `A_log`, `norm` | bf16, plain | `<...>.weight` (not `weight_packed`) |
| `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `lm_head` | NVFP4 packed | `weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale` |

The `recipe.yaml` ignore list excludes `re:.*linear_attn\..*` from quantisation. A correct loader must

1. honour the ignore list and instantiate `linear_attn.*` as plain bf16 Linear rather than NVFP4-wrapped;
2. apply `packed_modules_mapping` (`in_proj_qkvz <- [in_proj_qkv, in_proj_z]`, `in_proj_ba <- [in_proj_b,
   in_proj_a]`) so the checkpoint's split form fuses into the runtime's fused parameters;
3. strip the `model.language_model.` prefix.

vLLM 0.19.2 nightly did all three out of the box. SGLang dev at the time did 2 and 3 but mis-detected
`linear_attn.*` as NVFP4, ending with `in_proj_ba.weight_packed` in its parameter dict and silently skipping
the GDN weights. Patching the lookup made it assert in `load_merged_column_weight` because bf16 GDN tensors
lack the packed leading-dim halving. The right fix is to honour `ignore` before wrapping the quant method,
not to patch the loader.

## Working recipe (vLLM nightly, one GPU)

- `quantization="compressed-tensors"` (auto-detected from `quantization_config`), `dtype="bfloat16"`
  because the GDN layers are bf16.
- **`max_num_seqs=64`.** The default 1024 exceeds the Mamba cache blocks (one per decode sequence) at
  util 0.70: `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (736)`. The same trap hits
  Ornith-1.5-35B at 128.
- CUDA graphs on (31 s capture); eager mode dropped 48 to 3.7 tok/s.

Measured: 48.2 tok/s single stream after warm-up. Alternatives ruled out: `VLLM_NVFP4_GEMM_BACKEND=cutlass`
30.1 tok/s, `flashinfer-cudnn` 29.8 tok/s, `flashinfer-trtllm` errors (`mm_fp4 does not support backend
'trtllm' with capability 120`; sm_100/sm_103 only), flashinfer 0.6.9's b12x NVFP4 GEMM failed to compile
(`ptxas ... Unexpected instruction types specified for '_mma'`: it needs CUDA toolkit 13.3, the box had
13.2.1). The b12x kernel is the path from 48 to the ~140 tok/s others report on Blackwell; the blocker was
the toolkit, not the hardware. TensorRT-LLM had no GDN / Qwen3.5-hybrid model class at the time.
