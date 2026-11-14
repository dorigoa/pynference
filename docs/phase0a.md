# Phase 0a — environment and hardware measurements (2026-11-14)

Test PC (`ssh ubuntu`): Ryzen 9 5950X (16 cores, SMT off, AVX2), 107 GB DDR4 at
2133 MT/s (XMP 3200 does not boot), MSI MEG X570 ACE, 2× Intel Arc Pro B70
(BMG-G31, 31.9 GiB each, 256 EU) on PCIe 4.0 x8 each, Ubuntu 26.10, kernel 7.3.

## Software stack

| Component | Version |
|---|---|
| Python (venv from miniforge) | 3.13.13 |
| torch | 2.15.0+xpu (XPU index) |
| triton-xpu | 3.9.0 |
| oneCCL (xccl backend) | 2022.1.2 |
| SYCL runtime (pip) | 2026.1.2 |
| Level Zero loader / compute-runtime (system) | 1.32 / 26.27 |

Setup quirk: the system lacks `libze-dev`, so Triton (headers + `-lze_loader`)
and oneCCL (`dlopen("libze_loader.so")`) fail. `scripts/setup_env.sh` unpacks
`libze-dev` into `.venv/ze-dev` and adds a `libze_loader.so` symlink plus
`CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` to `.venv/bin/activate`. Alternative
system-wide fix: `sudo apt install libze-dev`.

Reported device properties (`has_subgroup_matrix_multiply_accumulate: False`,
`has_bfloat16_conversions: False`) look wrong for Xe2; to verify when prefill
matmuls in Triton matter.

Measurements were taken while a `llama-server` (Ripple-65B) held ~28 GiB per
GPU, idle; buffers were sized to fit the remaining ~2.5 / 4.3 GiB.

## Results

### RAM → GPU (`bench/h2d.py`, pinned host memory, GB/s)

| Copy size | H2D one GPU | H2D both (aggregate) | D2H one GPU | GPU0→GPU1 |
|---|---|---|---|---|
| 256 KB | 12.8 | 25.4 | 13.7 | – |
| 1 MB | 14.0 | 27.8 | 14.3 | 13.3 |
| 16 MB | 14.4 | 28.7 | 14.5 | 14.2 |
| 256 MB | 14.4 | 28.7 | 14.4 | 14.3 |

Pageable (non-pinned) H2D: 9–13.5 GB/s. Single 4 KB copy + sync: ~31 µs.
CPU-side STREAM-style read with 16 threads: 21.5 GB/s; DMA from both GPUs
together reaches 28.7 GB/s, i.e. ~84% of the DDR4-2133 dual-channel peak.

### GPU memory (`bench/devbw.py`, xpu:0, GB/s of weight bytes)

| Test | torch | Triton bf16 | Triton int8 (+row scale) |
|---|---|---|---|
| read (sum) 512 MiB | 568 | | |
| copy 512 MiB (r+w) | 524 | | |
| GEMV 5376→21504 (220 MB) | 586 | 544 | 519 |
| GEMV 21504→5376 (220 MB) | 549 | 559 | 530 |
| GEMV 5120→10240 (100 MB) | 576 | 531 | 523 |

A naive Triton GEMV already reaches ~90% of the card's bandwidth, also with
int8 weights: quantized kernels in Triton are viable.

### All-reduce between the GPUs (`bench/allreduce.py`, bf16)

| Size | xccl, stream-ordered | xccl + host sync | 1 process, via host | 1 process, direct copy |
|---|---|---|---|---|
| 5–32 KB (hidden state) | 35–37 µs | 55–58 µs | 57–64 µs | 70–75 µs |
| 256 KB | 36 µs | 64 µs | 118 µs | 81 µs |
| 2 MB | 233 µs | 266 µs | 676 µs | 348 µs |

### Launch overhead (`bench/launch.py`, xpu:0)

| Test | µs |
|---|---|
| eager tiny op | 5.3 / op |
| graph replay (XPUGraph) | 0.74 / op |
| op + synchronize round trip | 14.3 |

## Design consequences

1. **One process per GPU + xccl** (torchrun-style) for tensor parallelism: the
   stream-ordered all-reduce (~36 µs) beats any single-process scheme (≥57 µs).
   Cost per token ≈ 2 all-reduces × layers: ~4.3 ms (Gemma-4-31B, 60 layers),
   ~2.6 ms (GPT-OSS-120B, 36), ~4.6 ms (Qwen3.8-27B, 64), ~3.5 ms (Flash-Next, 48).
2. **Graph capture of the decode step** (XPUGraph): eager launches cost ~5 µs
   each, ~7× more than replay. Open question: can xccl collectives be captured?
3. **Expert fetch from RAM**: pinned memory is mandatory; per-expert copies
   (1–16 MB) already run at 14 GB/s per card, and the two cards together get
   28.7 GB/s. Splitting each expert across the two GPUs uses both links. Every
   GB of expert misses per token costs ~35 ms.
4. **Dense-model ceiling** (weights read at ~540 GB/s per card, split in two,
   plus all-reduce): ~30 tok/s for Qwen3.8-27B Q8_0 and Gemma-4-31B Q8_0.
5. Development runs need the GPUs free: stop `llama-server` before loading models.
