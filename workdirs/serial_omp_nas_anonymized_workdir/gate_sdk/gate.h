#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


static inline uint64_t gate_fnv1a64_bytes(const void* data, size_t nbytes) {
    const unsigned char* p = (const unsigned char*)data;
    uint64_t h = 1469598103934665603ull;      // offset basis
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;                // FNV prime
    }
    return h;
}

static inline void GATE_CHECKSUM_U8(const char* name, const unsigned char* buf, size_t n) {
    uint64_t h = gate_fnv1a64_bytes(buf, n);
    printf("GATE:SUM name=%s dtype=u8 algo=fnv1a64 value=%016llx n=%zu\n",
           name, (unsigned long long)h, n);
    fflush(stdout);
}
static inline void GATE_CHECKSUM_U32(const char* name, const uint32_t* buf, size_t n) {
    uint64_t h = gate_fnv1a64_bytes(buf, n*sizeof(uint32_t));
    printf("GATE:SUM name=%s dtype=u32 algo=fnv1a64 value=%016llx n=%zu\n",
           name, (unsigned long long)h, n);
    fflush(stdout);
}
static inline void GATE_CHECKSUM_BYTES(const char* name, const void* buf, size_t nbytes) {
    uint64_t h = gate_fnv1a64_bytes(buf, nbytes);
    printf("GATE:SUM name=%s dtype=bytes algo=fnv1a64 value=%016llx nbytes=%zu\n",
           name, (unsigned long long)h, nbytes);
    fflush(stdout);
}

static inline void GATE_STATS_F32(const char* name, const float* a, size_t n) {
    double s=0.0, s2=0.0; float mn=INFINITY, mx=-INFINITY;
    for (size_t i=0;i<n;i++){ float v=a[i];
        if (v<mn) mn=v; if (v>mx) mx=v; s+=v; s2+=(double)v*v; }
    double mean = n? s/n : 0.0, l1=fabs(s), l2=sqrt(s2);
    printf("GATE:STAT name=%s dtype=f32 n=%zu min=%.9g max=%.9g mean=%.9g L1=%.9g L2=%.9g\n",
           name, n, (double)mn,(double)mx, mean, l1, l2);
    fflush(stdout);
}
static inline void GATE_STATS_F64(const char* name, const double* a, size_t n) {
    long double s=0.0L, s2=0.0L; double mn=INFINITY, mx=-INFINITY;
    for (size_t i=0;i<n;i++){ double v=a[i];
        if (v<mn) mn=v; if (v>mx) mx=v; s+=v; s2+=(long double)v*v; }
    long double mean = n? s/n : 0.0L, l1=fabsl(s), l2=sqrtl(s2);
    printf("GATE:STAT name=%s dtype=f64 n=%zu min=%.17g max=%.17g mean=%.17Lg L1=%.17Lg L2=%.17Lg\n",
           name, n, mn, mx, mean, l1, l2);
    fflush(stdout);
}

#ifdef __cplusplus
}
#endif