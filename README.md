# blackwell-serving-kit

Launch recipes, measured numbers, and the traps behind them for serving frontier open-weight models on a
workstation with **two RTX PRO 6000 Blackwell GPUs (SM120, 98 GB each, no NVLink)** and 256 GB of RAM.
Everything here ran on that box between April and September 2026. Nothing is theoretical: every recipe
has a tokens-per-second figure and most have a quality score from the same coding suite.

SM120 is consumer Blackwell. Almost every model release, quant, and engine recipe targets datacenter
Blackwell (SM100: B200, GB200) or Hopper. Getting these models to serve here means dedicated images,
patched loaders, offload tricks, and a kernel regression that wedges every tensor-parallel launch. This
repository is the map.

## What is in here

| model | quant, engine | GPUs | decode single | aggregate | quality | recipe |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next (125B/6B active MoE + 51B n-gram table) | FP8, vLLM `qwen38-flash-next` image, TP2, table offloaded to host RAM, MTP nst=4 | 2 | **233.8 t/s** (113.9 without MTP) | 1,712 t/s @16 | agentic hard suite mean 380/400, MTP quality-neutral | `scripts/qwen38-flash-next-fp8-tp2.sh`, `docs/qwen38-flash-next-notes.md` |
| Qwen3.8-Flash-Next, single GPU | NVFP4 W4A4, same image + `patches/`, table offloaded | 1 | 90.6 t/s | 496 t/s @16 | mean 300/400 | `scripts/qwen38-flash-next-nvfp4-1gpu.sh` |
| DeepSeek-V4-Flash (284B/13B active) | NVFP4 + B12X vLLM build with working MTP; FP8 + DSpark build | 2 | 108.8 to 150.6 t/s (MTP, +38%); 182 to 186 t/s (FP8 DSpark) | 201 t/s @4 | 459/500 direct | separate repo: [dsv4-flash-nvfp4-sm120](https://github.com/sipratt-p/dsv4-flash-nvfp4-sm120) |
| Ornith-1.5-35B-A3B | bf16, stock vLLM | 1 | 179.5 t/s | **3,444 t/s @128**, not saturated | **489/500** direct, best measured | `scripts/serve_ornith35b.sh` |
| Ornith-1.5-397B (A15B) | NVFP4, vLLM TP2 + selective UVA expert offload (120 GB pinned RAM) | 2 | 10.9 t/s | 12.5 t/s @4 (PCIe-bound) | overnight-batch class | `scripts/serve_ornith397b_nvfp4_offload.sh` |
| GLM-5.3-Flash (320B/18B active) | TR3-4bpw experts + FP8 attention, author's closed vLLM fork | 2 | 108 to 117 t/s | 105 to 185 t/s @4 | mean 372/400 with an 8K thinking budget, 4/100 without | `glm53-tr3/` |
| GLM-5.3-Flash NVFP4 (195 GB) | vLLM `glm53-flash` image | 2 | parked | | 9,984-token max context, then a kernel wall | `docs/glm53-flash-sm120-wall.md` |
| MiniMax M2.7 | NVFP4, SGLang + b12x kernels | 2 | 120 to 127 t/s | 408 t/s @16 | agent-tested | `scripts/serve_minimax_m27_sglang.sh` |
| MiniMax M3 REAP50 (129 GB) | NVFP4, pinned SGLang + two SwiGLU patches | 2 | staged, not yet launched | | | `scripts/serve_minimax_m3_reap50_UNVALIDATED.sh` |
| Qwen3.6-27B hybrid, NVFP4 | compressed-tensors, vLLM nightly | 1 | 48.2 t/s | | | `docs/nvfp4-hybrid-loader-notes.md` |

"Quality" numbers come from two suites in [coding-agent-bench](https://github.com/sipratt-p/coding-agent-bench):
the 5-task direct-API suite (/500) and the 4-task agentic hard suite (/400, three runs). Direct and agentic
scores are not comparable with each other.

## The four rules for this box

1. **`NCCL_P2P_DISABLE=1` on every multi-GPU launch, paired with `--disable-custom-all-reduce`.** Linux
   7.0.0-30 broke GPU peer-to-peer: collectives hang silently and direct cross-GPU copies return garbage
   while the driver still reports P2P as fine. Measured cost of the workaround: none. Details and a
   two-minute diagnosis in `docs/kernel-p2p-regression.md` and `tools/p2p_check.py`.
2. **`docker stop`, never `docker rm -f`, on `--ipc=host` engines.** A force kill leaks `/dev/shm/psm_*`
   segments and the next launch hangs before loading a single shard. Clean-up:
   `for f in /dev/shm/psm_* /dev/shm/sem.mp-*; do fuser "$f" 2>/dev/null || sudo rm -f "$f"; done`.
3. **One big model at a time.** Anything TP2 wants about 89 GB on each GPU. The single-GPU recipes
   (Ornith-35B, Qwen NVFP4) can share the box with a render job on the other GPU; the TP2 recipes cannot
   share it with anything. `scripts/free-gpus.sh` shows who holds what and stops only the known servers.
4. **GPU 0 is the weak one.** It drives the desktop (0.5 to 1.3 GB gone) and is power-capped at 350 W
   because it hits 90 C at 400 W on the stock fan curve; GPU 1 runs 450 W at 79 C. GPU 0 always OOMs first
   on a tight fit, and 0.985 is the highest memory utilisation that boots even with the desktop stopped.

## Recurring traps

- **Dedicated images.** Mainline vLLM has neither `qwen4_exp` (Qwen3.8-Flash-Next) nor `glm5_next`
  (GLM-5.3-Flash); `pip install vllm` will not load them. DeepSeek-V4 on SM120 needs a community B12X
  build. Pin images by digest.
- **Hybrid GatedDeltaNet models cap `--max-num-seqs`.** The default 1024 exceeds the Mamba cache blocks and
  the engine refuses to start; 128 for Ornith-35B, 64 for Qwen3.6-27B NVFP4, 16 for Ornith-397B.
- **Tool parsers are named, not guessed.** `qwen3_xml` for Ornith (`qwen3_engine` does not exist),
  `qwen3_coder` for Qwen3.8-Flash-Next, `glm47` for GLM-5.3 (the file is `glm47_moe_tool_parser.py`; the
  flag is `glm47`), `minimax-m2` for MiniMax M2.7 (0% tool calls without it). A wrong parser is an HTTP 400 on
  every agent request.
- **Never hard-code temperature 0 for reasoning models.** Qwen3.8-Flash-Next at greedy is a repetition-loop
  generator (two of five suite tasks never terminated); its own `generation_config.json` says 1.0.
- **Agents need a thinking budget.** GLM-5.3 without one reasons for 65K tokens and never acts. 8K beat 32K
  beat unlimited. `docs/thinking-budget.md`, `glm53-tr3/think_budget_proxy.py`.
- **Speculative decoding depth is constrained by the cache geometry.** Qwen3.8-Flash-Next boots with 1 to 4
  or 9 to 12 speculative tokens and crashes with 5 to 8. MTP raised decode 2.1x at zero functional cost.
- **Quant choice moves code quality more than engine choice.** Ornith-35B: 489/500 in bf16 on vLLM and on
  oMLX, 398/500 as Q8 or Q4 GGUF with the same scoping bug in both. Qwen3.8-Flash-Next NVFP4 single-GPU:
  parity on two hard tasks, collapse on one.
- **Community quants that skip the attention stack do not fit.** The 195 GB GLM-5.3 NVFP4 is 17% BF16 by
  bytes; a quant that also took the projections would be about 165 GB and would fit with room.
- **Downloads during a load blow the 600 s vLLM startup timeout** when they saturate the NVMe. Pause them.
- **A headless Chrome with a debug port pins a GPU context and wedges NCCL.** Kill its tree before a launch.

## Layout

```
scripts/   one launcher per model (env-overridable MODEL_DIR, PORT, MAXLEN, GPUUTIL, ...), plus free-gpus.sh
patches/   the one-file vLLM patch the single-GPU Qwen NVFP4 build needs (Apache-2.0, see patches/README.md)
glm53-tr3/ GLM-5.3-Flash TR3 launch adaptation, thinking-budget proxy, proxy restart, speed bench, attempt log
tools/     p2p_check.py: the two-minute kernel P2P diagnosis
docs/      the long-form notes: kernel regression, Qwen3.8 fit and traps, GLM-5.3 walls, thinking budgets,
           NVFP4 hybrid-model loader findings
```

All launchers assume weights under `$MODELS_ROOT` (default `/mnt/models`) in a directory named after the
Hugging Face repo, and they print the health-check command. They run the container in the foreground where
the image was built that way (`exec docker run --rm`); detach with `tmux` or `&`.

## Related repositories

- [dsv4-flash-nvfp4-sm120](https://github.com/sipratt-p/dsv4-flash-nvfp4-sm120): the DeepSeek-V4-Flash kit,
  including the MTP root-cause investigation and the 1M-token context config.
- [omlx-dualane-recipe](https://github.com/sipratt-p/omlx-dualane-recipe): the Mac Studio side of the same
  fleet (dual-ANE prefill + MTP on Apple silicon).
- [coding-agent-bench](https://github.com/sipratt-p/coding-agent-bench): the suites behind the quality
  numbers.
- [refusal-ablation-eval](https://github.com/sipratt-p/refusal-ablation-eval): why several of these serve
  abliterated builds and how that was checked.

## License

MIT, except `patches/qwen38fn_ple_layer.py`, which is a modified vLLM file under Apache-2.0. The GLM-5.3 TR3
runtime image and weights belong to their author under his own source-available license; `glm53-tr3/launch.sh`
is a launch adaptation, not a redistribution.
