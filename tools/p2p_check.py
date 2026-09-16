#!/usr/bin/env python3
"""Two-minute check for the GPU peer-to-peer regression that wedges every tensor-parallel launch.

Symptom: after a kernel update (Linux 7.0.0-30 here) any NCCL collective between two RTX PRO 6000s busy-spins
forever at 100% CPU and 100% GPU with zero log output, no Xid, no error. `torch.cuda.can_device_access_peer`
and `nvidia-smi topo -p2p r` still report P2P as fine. Worse: a plain cross-GPU tensor copy returns garbage
with no error (20/20 mismatches at 128 KB to 128 MB while the host-staged copy matched 20/20).

This script runs both probes and exits non-zero if either fails, so you can run it after every kernel or driver
update before blaming an inference engine. Workaround when it fails: NCCL_P2P_DISABLE=1 (and, for engines with
their own PCIe all-reduce, also disable that), and stage cross-GPU moves through the host (.cpu().to(dst)).
Measured cost of the workaround on a 2-GPU tensor-parallel serve: none (185.2 vs 185.7 tok/s).

Usage: python3 tools/p2p_check.py [--timeout 90] [--trials 20]
"""
import argparse
import os
import sys
import time

import torch


def _worker(rank, world, port, ok):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    t = torch.ones(1 << 20, device=f"cuda:{rank}") * (rank + 1)
    dist.all_reduce(t)
    torch.cuda.synchronize()
    expected = sum(range(1, world + 1))
    ok[rank] = int(torch.allclose(t, torch.full_like(t, expected)))
    dist.destroy_process_group()


def nccl_probe(timeout):
    import torch.multiprocessing as mp
    world = torch.cuda.device_count()
    if world < 2:
        print("nccl: skipped (fewer than 2 GPUs)")
        return True
    ctx = mp.get_context("spawn")
    ok = ctx.Array("i", [0] * world)
    port = 29511
    procs = [ctx.Process(target=_worker, args=(r, world, port, ok)) for r in range(world)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(max(1, timeout - (time.time() - t0)))
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.kill()
    if hung:
        print(f"nccl: HANG - {len(hung)} rank(s) still spinning after {timeout}s. This is the P2P regression.")
        return False
    if all(ok):
        print(f"nccl: all_reduce OK across {world} GPUs in {time.time() - t0:.1f}s")
        return True
    print(f"nccl: all_reduce returned WRONG values: {list(ok)}")
    return False


def copy_probe(trials):
    if torch.cuda.device_count() < 2:
        print("copy: skipped (fewer than 2 GPUs)")
        return True
    bad = 0
    for i in range(trials):
        n = 1 << (15 + (i % 6) * 2)  # 128 KB .. 128 MB of int64s
        src = torch.randint(0, 1 << 30, (n,), device="cuda:0")
        direct = src.to("cuda:1")
        staged = src.cpu().to("cuda:1")
        torch.cuda.synchronize()
        if not torch.equal(direct, staged):
            bad += 1
    if bad:
        print(f"copy: {bad}/{trials} direct cuda:0 -> cuda:1 copies MISMATCHED the host-staged copy. Silent corruption.")
        return False
    print(f"copy: {trials}/{trials} direct cross-GPU copies match")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count(),
          "| can_device_access_peer(0,1):", torch.cuda.can_device_access_peer(0, 1) if torch.cuda.device_count() > 1 else "n/a",
          "(the driver may say True and still be broken)")
    results = [copy_probe(a.trials), nccl_probe(a.timeout)]
    if all(results):
        print("P2P looks healthy.")
        sys.exit(0)
    print("P2P is broken: launch tensor-parallel engines with NCCL_P2P_DISABLE=1 and stage copies through the host.")
    sys.exit(1)
