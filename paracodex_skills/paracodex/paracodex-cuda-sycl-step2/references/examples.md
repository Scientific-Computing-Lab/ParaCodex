# Q1: USM vs Buffers?
> **Advice:** Prefer USM (Device/Shared allocations) for performance and pointer-based control.
```cpp
// Buffer (Implicit)
buffer<float, 1> buf(data, range<1>(N));

// USM (Explicit)
float* data = malloc_device<float>(N, queue);
queue.memcpy(data, host_ptr, bytes);
```

# Q2: Sub-groups (Warp Primitives)?
> **Concept:** SYCL `sub_group` maps to CUDA Warp (32) or Intel Subslice (16/32).
```cpp
auto sg = item.get_sub_group();
float val = sg.shuffle(local_val, src_lane);
```

# Q3: Work-Group Size Query (Runtime)?
```cpp
// Query max work-group size for a specific compiled kernel
auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
auto kern   = bundle.get_kernel<class MyKernel>();
size_t wg_max = kern.get_info<sycl::info::kernel_device_specific::work_group_size>(
                    q.get_device());
// Round down to nearest power of 2 or preferred multiple (e.g. 256)
size_t local = std::min(wg_max, (size_t)256);
size_t global = ((N + local - 1) / local) * local;  // Pad to multiple of local
```

# Q4: Kernel Launch with Chained Events (No Blocking .wait())?
```cpp
// Chain work without blocking the host between submissions
auto ev1 = q.memcpy(d_A, h_A, bytes);
auto ev2 = q.submit([&](sycl::handler& h) {
    h.depends_on(ev1);
    h.parallel_for(sycl::nd_range<1>{global, local},
                   [=](sycl::nd_item<1> it) { /* kernel */ });
});
auto ev3 = q.memcpy(h_out, d_A, bytes, ev2);  // memcpy overload with depends_on
ev3.wait();  // Single wait at the end
```

# Q4b: Structural rewrite before micro-tuning
```cpp
// If profile shows many tiny submissions or host waits,
// merge submissions, remove unnecessary waits, and simplify the hot path first.
```

# Q5: Group Reduction (replaces manual shared-memory reduction)?
```cpp
q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>{global, local},
        [=](sycl::nd_item<1> it) {
            float val = d_in[it.get_global_id(0)];
            // Reduce across entire work-group — no shared memory needed
            float sum = sycl::reduce_over_group(it.get_group(), val,
                                                sycl::plus<float>());
            if (it.get_local_id(0) == 0)
                d_out[it.get_group(0)] = sum;
        });
});
```

# Q6: Required Work-Group Size Attribute?
```cpp
// Fixes work-group size at compile time — allows compiler to unroll/optimize
q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>{global, 256},
        sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::work_group_size<256>{}
        },
        [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
            // kernel
        });
});
```

# Q7: Required Sub-Group Size Attribute?
```cpp
// Pin sub-group width to avoid runtime fallback to size 1
// Use 32 for NVIDIA/AMD-like behavior, 16 for Intel GPU
q.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::nd_range<1>{global, 256},
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            auto sg = it.get_sub_group();
            float val = d_in[it.get_global_id(0)];
            // Sub-group reduce — guaranteed sg.get_local_range()[0] == 32
            float sum = sycl::reduce_over_group(sg, val, sycl::plus<float>());
            if (sg.get_local_id() == 0)
                d_out[it.get_group(0) * sg.get_group_range()[0] + sg.get_group_id()[0]] = sum;
        });
});
```
