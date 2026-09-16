# Qwen3.8-Flash-Next on two RTX PRO 6000s: how it fits and what bit us

## Why it fits at all

`qwen4_exp` is not a Qwen3-Next reskin: 125B total / 6B active MoE (512 experts, 10 routed + 1 shared) plus a
**51B n-gram (PLE) embedding table** and a 4B MTP head, about 180B parameters. Hybrid attention: three
Gated DeltaNet layers then one Qwen Sparse Attention layer (512-token budget), times 12. Vision and video
capable, 262K native context.

BF16 is 360 GB. FP8 is 186 GB, which also does not fit 196 GB of VRAM naively. The lever is
`VLLM_PLE_CPU_OFFLOAD=1`, which parks the 51B n-gram table in host RAM (about 51 GB) and drops
GPU-resident weights to about 122 GB (about 89 GB per GPU including KV at util 0.92). LMSYS measured the
offload at -0.07% throughput on H200 TP4; it is free, not a compromise.

Mainline vLLM does not support `qwen4_exp`; the dedicated `vllm/vllm-openai:qwen38-flash-next` image is
required. TP2 is the documented minimum; upstream validated only 4x GB300 / 8x H200 / 4x MI355X, so the
numbers here are ours.

## Measured, two GPUs, FP8

| config | decode t/s | prefill t/s | C=16 aggregate | KV tokens | draft accept |
|---|---|---|---|---|---|
| no MTP | 113.9 | 10,043 | 996 | 1.58M | |
| MTP nst=3 | 239.2 | 7,568 | 1,376 | 1.21M | 91.9% |
| MTP nst=4 | 242.5 | 7,596 | 1,739 | 1.18M | 83.7% |
| MTP nst=12 | 247.2 | 7,476 | 1,250 | 1.00M | 63.0% |
| nst=4 + batch 16384 | 235.2 | 9,396 | 1,901 | 987K | |
| **nst=4 + batch 16384 + util 0.92 (shipped)** | **233.8** | **9,243** | **1,712** | **1.11M** | |

Quality gate on the 5-task coding suite at temperature 1.0: MTP nst=4 run 1 475/500, run 2 480/500,
no-MTP control 482/500, with the functional subtests **identical in all three runs (400/400)**. The whole
spread is the subjective code-quality heuristic, which swung 75 to 80 between two identical MTP configs.
MTP is free here.

Agentic hard suite, three runs: 392 / 359 / 389, mean 380/400, zero degenerate generations in 12 tasks.

## Measured, one GPU, NVFP4 (dealignai build, W4A4, PLE stored FP8)

| metric | 1-GPU NVFP4 | 2-GPU FP8 + MTP4 | 2-GPU FP8 no MTP |
|---|---|---|---|
| decode, single stream | 90.6 t/s | 233.8 t/s | 113.9 t/s |
| prefill, fresh 123K prompt | 12,062 t/s | 9,243 t/s | 10,043 t/s |
| C=16 aggregate | 496 t/s | 1,712 t/s | 996 t/s |
| KV budget | 485K tokens | 1.11M | 1.58M |

Weights 74.2 GiB plus 12.1 GiB KV at util 0.95 on the one GPU, about 51 GB host RAM for the table. Hard
suite mean 300/400 (274 / 359 / 268): parity on two tasks, a consistent partial on the regex engine, and a
catastrophic JSON-parser task in 2 of 3 runs. The single-GPU build is for nights when the other GPU must
stay free; accept about 80 points on hard agentic coding.

## Traps, in the order they cost time

- **`pidfd_getfd: Operation not permitted`** at startup. With PLE offload the offload worker hands the
  n-gram tensor to the TP workers via CUDA IPC (`torch.UntypedStorage._new_shared_cuda`), which calls
  `pidfd_getfd(2)`. Docker's default seccomp profile blocks it. Fix: `--cap-add=SYS_PTRACE
  --security-opt seccomp=unconfined`.
- **`num_speculative_tokens` is not free choice.** `qsa_cache.py` computes
  `capacity = compress_ratio * ceil((compress_ratio + nst) / compress_ratio)` and asserts
  `block_size % capacity == 0`. With `indexer_compress_ratio 4` and block size 16: nst 1 to 4 boot
  (capacity 8), 5 to 8 crash at startup ("QSA ring capacity 12 must divide the attention block size 16"),
  9 to 12 boot (capacity 16). The conventional 5 from Qwen's own recipe does not start.
- **MTP costs prefill** (10,043 to about 7,600 t/s); `--max-num-batched-tokens 16384` buys most of it back at
  about 3% decode. 32768 gives no more prefill and halves the KV pool.
- **Temperature 0 is a repetition-loop generator on this model.** Its `generation_config.json` says
  temperature 1.0, top_p 0.95, top_k 20, and the server applies that by default. Two of five suite tasks
  never terminated when a harness hard-coded 0.0 (130K+ tokens). Do not hard-code greedy decoding for
  reasoning models.
- **Single-GPU TP1 hangs forever in warm-up** unless `--distributed-executor-backend mp` is passed: the
  uniproc executor never spawns the PLE offload worker, and CUDA-graph replay waits on IPC that never
  comes. No error, just silence after "Model loading took ...".
- **NVFP4 checkpoints store the PLE table as FP8 plus a global scale but list `*.ple.*` under `ignore`**, so
  stock vLLM builds the layer as bf16 and the loader dies on `ngram_embedding.weight_scale`.
  `patches/qwen38fn_ple_layer.py` with `VLLM_PLE_FORCE_FP8=1` forces the same FP8 embedding method the
  official FP8 checkpoint uses. Bind-mount it over the image's `ple_layer.py`.
- **Reasoning comes back as `reasoning`, not `reasoning_content`**; the count is in
  `usage.completion_tokens_details.reasoning_tokens`.
- util 0.94 fits but leaves about 670 MB free on GPU 0 with a desktop running; 0.92 gives about 5.6 GB.
- Harmless startup noise: `rope_parameters ... mrope_section` warnings and Qwen3-VL `min_frames/max_frames`
  ERROR lines.
- `docker rm -f` on an `--ipc=host` container leaks `/dev/shm/psm_*` segments and the next launch hangs
  pre-load. Prefer `docker stop`; after a force kill:
  `for f in /dev/shm/psm_* /dev/shm/sem.mp-*; do fuser "$f" 2>/dev/null || sudo rm -f "$f"; done`.
- vLLM loads die at the 600 s startup timeout if a large download saturates the NVMe at the same time.
