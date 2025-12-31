# Nsight Systems Report - Performance Metrics Explanation

## Example Output Structure

```
[5/7] Executing 'cuda_gpu_kern_sum' stats report
═══════════════════════════════════════════════════════════════════════════════
 Time (%)  Total Time (ns)  Instances  Name
 --------  ---------------  ---------  ----------------------------------------
   94.5    4,402,169,680     1,900     nvkernel_conj_grad_...
    3.8      176,623,013        76     nvkernel_other_kernel_...
    0.5       24,601,215     1,900     nvkernel_small_kernel_...
    ...
───────────────────────────────────────────────────────────────────────────────
TOTAL KERNEL TIME: ~4,603,393,908 ns (≈ 4,603 ms)
```

**📊 Kernel-only Metric** = Sum of all GPU kernel execution times
- Measures: Pure GPU compute time
- Includes: All kernel launches and their execution
- Excludes: Memory transfers, API calls, OS overhead

---

```
[6/7] Executing 'cuda_gpu_mem_time_sum' stats report
═══════════════════════════════════════════════════════════════════════════════
 Time (%)  Total Time (ns)  Count  Operation
 --------  ---------------  -----  ----------------------------
   90.5       56,732,320       8   [CUDA memcpy Host-to-Device]
    7.2        4,488,909   4,108   [CUDA memcpy Device-to-Host]
    2.3        1,439,364   4,103   [CUDA memset]
───────────────────────────────────────────────────────────────────────────────
TOTAL MEMORY TIME: ~62,660,593 ns (≈ 63 ms)
```

**📊 GPU Metric** = Kernel-only + Memory transfer time
- Measures: GPU-side work (compute + data movement)
- Includes: All kernel execution + GPU memory operations
- Excludes: CUDA API overhead, OS runtime

---

```
[4/7] Executing 'cuda_api_sum' stats report
═══════════════════════════════════════════════════════════════════════════════
 Time (%)  Total Time (ns)  Num Calls  Name
 --------  ---------------  ---------  --------------------
   89.6    5,202,220,003    23,717     cuStreamSynchronize
    4.3      247,035,134     9,959     cuLaunchKernel
    3.5      204,745,327     4,108     cuMemcpyDtoHAsync_v2
    1.2       68,839,718         8     cuMemcpyHtoDAsync_v2
    1.2       66,890,196     4,103     cuMemsetD32Async
    ...
───────────────────────────────────────────────────────────────────────────────
TOTAL CUDA API TIME: ~5,789,730,378 ns (≈ 5,790 ms)
```

**📊 CUDA API Overhead** = Time spent in CUDA driver/runtime calls
- Measures: Synchronization, kernel launches, memory management API calls
- Includes: cuStreamSynchronize, cuLaunchKernel, cuMemcpy*, etc.
- Note: This is overhead, not actual GPU work

---

```
[3/7] Executing 'osrt_sum' stats report
═══════════════════════════════════════════════════════════════════════════════
 Time (%)  Total Time (ns)  Num Calls  Name
 --------  ---------------  ---------  --------------
   93.8    6,012,385,250        64     poll
    3.4      219,234,978    19,243     ioctl
    1.6      103,769,581    19,918     mprotect
    1.1       72,569,372         4     fread
    ...
───────────────────────────────────────────────────────────────────────────────
TOTAL OS RUNTIME TIME: ~6,409,959,181 ns (≈ 6,410 ms)
```

**📊 OS Runtime** = Time spent in operating system calls
- Measures: System-level operations
- Includes: poll, ioctl, mprotect, file I/O, etc.
- Note: Often the largest component due to synchronization overhead

---

## Summary: Three Metrics Explained

### 1️⃣ **Kernel-only** (4,603 ms)
```
┌─────────────────────────────────────┐
│  GPU Kernel Execution Only          │
│  • nvkernel_conj_grad: 4,402 ms     │
│  • nvkernel_other: 176 ms           │
│  • nvkernel_small: 25 ms            │
└─────────────────────────────────────┘
```
**Use case:** Measure pure GPU compute performance

---

### 2️⃣ **GPU Time** (4,666 ms)
```
┌─────────────────────────────────────┐
│  Kernel-only (4,603 ms)             │
│  + Memory Transfers (63 ms)         │
│  ────────────────────────────────   │
│  = 4,666 ms                         │
└─────────────────────────────────────┘
```
**Use case:** Measure total GPU-side work (compute + data movement)

---

### 3️⃣ **Full Execution Time** (16,864 ms)
```
┌─────────────────────────────────────┐
│  Kernel-only:        4,603 ms      │
│  + Memory:              63 ms       │
│  + CUDA API:         5,790 ms      │
│  + OS Runtime:       6,410 ms      │
│  ────────────────────────────────   │
│  = Full Time:       16,864 ms      │
└─────────────────────────────────────┘
```
**Use case:** Measure total application runtime (end-to-end)

---

## Visual Comparison

```
Time (ms)
│
│  ┌─────────────────────────────────────┐
│  │                                     │
│  │  Full Execution Time (16,864 ms)   │
│  │  ├─ OS Runtime (6,410 ms)          │
│  │  ├─ CUDA API (5,790 ms)            │
│  │  ├─ Memory (63 ms)                 │
│  │  └─ Kernels (4,603 ms)             │
│  │                                     │
│  ├─────────────────────────────────────┤
│  │                                     │
│  │  GPU Time (4,666 ms)                │
│  │  ├─ Memory (63 ms)                 │
│  │  └─ Kernels (4,603 ms)              │
│  │                                     │
│  ├─────────────────────────────────────┤
│  │                                     │
│  │  Kernel-only (4,603 ms)            │
│  │                                     │
│  └─────────────────────────────────────┘
│
└───────────────────────────────────────────
```

---

## Key Insights

1. **Kernel-only** is the smallest and represents pure GPU compute
2. **GPU Time** adds memory transfers (usually small increase)
3. **Full Time** includes all overhead:
   - CUDA API synchronization (often significant)
   - OS runtime calls (can be very large due to polling)
4. The difference between Full and GPU shows the **overhead** of API calls and OS operations

---

## When to Use Each Metric

| Metric | Best For |
|--------|----------|
| **Kernel-only** | Optimizing GPU compute kernels, comparing algorithm efficiency |
| **GPU Time** | Understanding GPU-side bottlenecks (compute vs memory) |
| **Full Time** | Measuring actual application performance, user-perceived latency |

