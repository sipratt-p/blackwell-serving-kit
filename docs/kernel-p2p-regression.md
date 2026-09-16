# The kernel P2P regression (Linux 7.0.0-30, dual RTX PRO 6000)

Found 2026-08-23 after an unattended kernel upgrade from 7.0.0-28 to 7.0.0-30 was activated by a reboot.

## Symptom

Any NCCL collective between the two GPUs hangs. Both worker processes spin at 100% CPU and 100% GPU
forever, with zero log output, no Xid in `dmesg`, no exception. In a vLLM launch it appears right after
the sampler initialises and looks exactly like a long compilation. The engine core spams
`shm_broadcast: No available shared memory broadcast block` every 60 seconds.

`torch.cuda.can_device_access_peer(0, 1)` returns True and `nvidia-smi topo -p2p r` reports OK. The
driver lies; only a real transfer reveals it.

## Proof that it is the kernel, not the engine

Every working engine run on 2026-08-22 was on 7.0.0-28. On 7.0.0-30 the wedge reproduced
deterministically twice, then a bare 12-line `torch.distributed.all_reduce` between two ranks hung too.

The following day it got worse: plain cross-GPU tensor copies are silently corrupted, not just hung.
`torch.randint(..., device="cuda:0").to("cuda:1")` mismatched the source in 20 of 20 trials from 128 KB
to 128 MB, while `.cpu().to("cuda:1")` matched 20 of 20. Any model sharding, pipeline split, or
cross-GPU helper that copies directly reads garbage with no error. It showed up as a text-encoder
output changing token count from run to run.

## Workaround

- `NCCL_P2P_DISABLE=1` on every tensor-parallel launch. NCCL falls back to shared-memory staging.
- Pair it with `--disable-custom-all-reduce` in vLLM. Disabling P2P alone crashes the custom all-reduce
  (`custom_all_reduce.cuh:164 'invalid argument'`); leaving both on gives a 30-minute gloo
  `is_in_the_same_node` timeout wedge.
- Engines with their own PCIe all-reduce (the B12X DeepSeek builds) also need that path off:
  `VLLM_ENABLE_PCIE_ALLREDUCE=0`.
- The GLM-5.3 TR3 runtime has two more direct GPU-to-GPU switches that both have to be off:
  `VLLM_USE_B12X_DCP_A2A=0` and `VLLM_USE_DIRECT_DCP_A2A=0`. With either on, "direct DCP A2A timeout"
  spam ends in Xid 43 during CUDA-graph capture.
- Stage cross-GPU tensor moves in your own code through the host.

Measured cost of the workaround on the DeepSeek-V4-Flash serve: none (185.2 tok/s with P2P disabled vs
185.7 with it on, 2026-08-19).

## Diagnose in two minutes

```bash
python3 tools/p2p_check.py
```

Run it after every kernel or driver update before touching engine flags. Full speed on the direct
paths returns only by booting the previous kernel, which stays in GRUB until you remove it.
