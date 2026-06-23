# Q1: How to use dpct?
> **Command:**
> `dpct --in-root=. --out-root=dpct_output src/kernel.cu`
>
> **For Makefiles:**
> `intercept-build make`
> `dpct -p compile_commands.json --in-root=. --out-root=dpct_output`

Tool-driven conversion is only the starting point. The final SYCL submission structure should minimize fragmentation and unnecessary waits.

# Q2: Common Warning Fixes
> **DPCT1003: Error code removed**
> *Reason:* SYCL uses exceptions, not multiple return codes.
> *Fix:* Remove error checking variable `status` and wrap in try-catch if needed.

> **DPCT1009: Work group size**
> *Fix:* Ensure `nd_range` global size is a multiple of local size.

# Q3a: USM with Chained Events (for multi-pipeline overlap)
```cpp
#include <sycl/sycl.hpp>

sycl::queue q{sycl::gpu_selector_v,
    [](sycl::exception_list el) {
        for (auto& e : el) std::rethrow_exception(e);
    }};

int N = 1024;
float *d_A = sycl::malloc_device<float>(N, q);
float *d_B = sycl::malloc_device<float>(N, q);
float *d_C = sycl::malloc_device<float>(N, q);

// Async H→D with chained events
auto ev_a = q.memcpy(d_A, h_A, N * sizeof(float));
auto ev_b = q.memcpy(d_B, h_B, N * sizeof(float));

// Kernel depends on both transfers
auto ev_k = q.submit([&](sycl::handler& h) {
    h.depends_on({ev_a, ev_b});
    h.parallel_for(sycl::nd_range<1>{N, 256}, [=](sycl::nd_item<1> it) {
        int i = it.get_global_id(0);
        d_C[i] = d_A[i] + d_B[i];
    });
});

// D→H depends on kernel
q.memcpy(h_C, d_C, N * sizeof(float), ev_k).wait();

sycl::free(d_A, q); sycl::free(d_B, q); sycl::free(d_C, q);
```

# Q3b: in_order Queue (Simpler Alternative for Linear Pipelines)
```cpp
// For purely sequential pipelines — cleaner than depends_on chaining
// All operations execute in submission order; no events needed
sycl::queue q{sycl::gpu_selector_v,
    sycl::property_list{sycl::property::queue::in_order{}}};

float *d_A = sycl::malloc_device<float>(N, q);

q.memcpy(d_A, h_A, N * sizeof(float));   // Implicit ordering
q.parallel_for(sycl::nd_range<1>{N, 256}, [=](sycl::nd_item<1> it) {
    // kernel (runs after memcpy above due to in_order)
});
q.memcpy(h_out, d_A, N * sizeof(float));
q.wait();   // Wait at end only

sycl::free(d_A, q);
```

# Q4: Exception Handling (replaces CUDA error codes)
```cpp
try {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(..., [=](sycl::nd_item<1> it) { ... });
    }).wait_and_throw();
} catch (const sycl::exception& e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
}
```

# Q4b: Submission structure matters
> Prefer one coherent SYCL submission pipeline for the hot path.
> If the migrated code produces many tiny queue submissions or `.wait()` calls,
> rewrite the structure before step2.

# Q5: Interoperability?
> **Calling CUDA from SYCL:**
```cpp
auto native_stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
cudaKernel<<<...>>>(...); // Raw CUDA call
```

# Q6: Compilation?
```bash
icpx -fsycl source.cpp -o app
# With CUDA backend:
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda source.cpp
```
