# patches

`qwen38fn_ple_layer.py` is `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` from the
`vllm/vllm-openai:qwen38-flash-next` image with one addition around line 194: when `VLLM_PLE_FORCE_FP8=1`
is set, the PLE n-gram embedding uses the FP8 embedding method even though the NVFP4 (modelopt) checkpoints
from RadixArk and dealignai list `*.ple.*` under `ignore`. Those checkpoints store the table as FP8 with one
global `weight_scale`; without the patch stock vLLM builds the layer in bf16 and the loader dies on
`ngram_embedding.weight_scale`.

Bind-mount it over the image's file, as `scripts/qwen38-flash-next-nvfp4-1gpu.sh` does. It is a vLLM file and
stays under vLLM's Apache-2.0 license (SPDX header retained); the rest of this repository is MIT.
