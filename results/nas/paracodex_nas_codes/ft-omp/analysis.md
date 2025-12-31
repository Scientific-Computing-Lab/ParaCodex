Loop Nesting Structure
- main iteration loop ft.c:188 Type A (driver) -> calls evolve, fft(-1), checksum each iter
  └── evolve triply nested k-j-i ft.c:241 Type A
  └── fft(-1) -> cffts3_neg (outer j-i-k + stage butterflies), then cffts2_neg (outer k-i-j + stages), then cffts1_neg (outer k-j-i + stages); each contains stage loop l with inner i1/k1 butterflies Type C1
  └── checksum loop ft.c:1113 Type F (reduction over 1024 samples)
- setup-only loops in timed region (before iter loop): compute_indexmap ft.c:373 Type A; compute_initial_conditions outer k-j Type A with inner RNG loop ft.c:290 Type E; fft_init outer j with inner cosine loop ft.c:427 Type A; forward FFT cffts1_pos/cffts2_pos/cffts3_pos stage loops Type C1
- pre-timer init_ui triply nested k-j-i ft.c:221 Type A (zero fill)

Loop Details
## Loop: main iteration driver at ft.c:188
- Iterations: niter (default 6)
- Type: A - dense fixed trip count, calls heavy kernels each iteration
- Parent loop: none
- Contains: evolve; fft(-1) (cffts3_neg → cffts2_neg → cffts1_neg); checksum
- Dependencies: none inside driver; ordering matters for algorithm
- Nested bounds: constant
- Private vars: iter
- Arrays: none directly
- Issues: none

## Loop: evolve k-j-i at ft.c:241
- Iterations: d3*d2*d1 = 64*64*64 = 262,144 per iter
- Type: A - dense element-wise complex scale/copy
- Parent loop: main iteration ft.c:188
- Contains: none
- Dependencies: none; independent per point
- Nested bounds: constant
- Private vars: i,j,k
- Arrays: u0_real/u0_imag (RW), twiddle (R), u1_real/u1_imag (W)
- Issues: none

## Loop: cffts3_neg stage nest at ft.c:951 (outer j-i-k with stage l)
- Iterations: outer j*d1*d3 = 64*64*64; inner stage loop l=1..log2(d3)=6 with i1/k1 butterflies
- Type: C1 - FFT butterfly with staged scratch swaps (GTY1/GTY2)
- Parent loop: main iteration via fft(-1)
- Contains: per-stage loops over i1 (li) and k1 (lk)
- Dependencies: stage dependency across l; each stage uses outputs from previous stage in GTY buffers
- Nested bounds: constant powers-of-two
- Private vars: j,i,k,l,i1,k1,i11,i12,i21,i22,uu*,x*,temp*
- Arrays: u1_real/u1_imag (R/W), gty1_real/gty1_imag (RW scratch), gty2_real/gty2_imag (RW scratch), u_real/u_imag twiddle factors (R)
- Issues: scratch swap each stage; no reduction

## Loop: cffts2_neg stage nest at ft.c:683
- Iterations: outer k*d1*d2 = 64*64*64; stage loop l=1..log2(d2)=6 with i1/k1 butterflies
- Type: C1 - FFT butterfly using GTY1/GTY2 scratch
- Parent loop: main iteration via fft(-1)
- Contains: per-stage i1/k1 loops
- Dependencies: stage dependency across l; twiddle factors used per butterfly
- Nested bounds: constant powers-of-two
- Private vars: k,i,j,l,i1,k1,i11,i12,i21,i22,uu*,x*,temp*
- Arrays: u1_real/u1_imag (R/W), gty1_real/gty1_imag (RW), gty2_real/gty2_imag (RW), u_real/u_imag (R)
- Issues: scratch swap per stage; no reduction

## Loop: cffts1_neg stage nest at ft.c:473
- Iterations: outer k*d2*d1 = 64*64*64; stage loop l=1..log2(d1)=6 with i1/k1 butterflies
- Type: C1 - FFT butterfly with staged scratch swaps
- Parent loop: main iteration via fft(-1)
- Contains: per-stage i1/k1 loops
- Dependencies: stage dependency across l; uses twiddle factors per butterfly
- Nested bounds: constant powers-of-two
- Private vars: k,j,i,l,i1,k1,i11,i12,i21,i22,uu*,x*,temp*
- Arrays: u1_real/u1_imag (R/W), gty1_real/gty1_imag (RW), gty2_real/gty2_imag (RW), u_real/u_imag (R)
- Issues: scratch swap per stage; no reduction

## Loop: checksum sum at ft.c:1113
- Iterations: 1024 per iter
- Type: F - reduction to scalar (temp1/temp2) over fixed sample points
- Parent loop: main iteration ft.c:188
- Contains: none
- Dependencies: reduction on temp1/temp2
- Nested bounds: constant
- Private vars: j,q,r,s
- Arrays: u1_real/u1_imag (R)
- Issues: <10K iterations; reduction required if parallelized

## Loop: compute_indexmap k-j-i at ft.c:373
- Iterations: d3*d2*d1 = 262,144 (once per timed run)
- Type: A - dense computation of exponential weight
- Parent loop: none (setup inside timed region before iterations)
- Contains: none
- Dependencies: none
- Nested bounds: constant
- Private vars: i,j,k,kk,kk2,jj,kj2,ii
- Arrays: twiddle (W)
- Issues: setup-only

## Loop: compute_initial_conditions outer k-j with inner RNG at ft.c:273/290
- Iterations: outer k*d2 = 64*64 = 4096 blocks; inner i loop 0..2*NX-1 = 128
- Type: outer Type A (independent grid points); inner Type E due to RNG recurrence on x0 and randlc seeding
- Parent loop: none (setup inside timed region before iterations)
- Contains: inner RNG recurrence loop
- Dependencies: inner loop has loop-carried RNG state (x0); outer k uses starts[k] seeded sequentially by randlc in loop ft.c:268 (Type E across k)
- Nested bounds: constant
- Private vars: k,j,i,t1,t2,t3,t4,a1,a2,x1,x2,z,x0
- Arrays: host_u1_real/host_u1_imag (W), starts (local), u1_real/u1_imag (W via memcpy)
- Issues: RNG recurrence; host→device memcpy after generation

## Loop: fft_init loops at ft.c:427/431
- Iterations: outer j=1..m=ilog2(d1)=6; inner i 0..ln-1 doubling each stage (sum ~63)
- Type: A - dense twiddle table fill
- Parent loop: setup portion before iterations
- Contains: inner cosine loop
- Dependencies: stage dependency on ln/ku updates per j
- Nested bounds: vary with stage (power-of-two growth)
- Private vars: j,i,ku,ln,t,ti
- Arrays: u_real/u_imag (W)
- Issues: small setup-only

## Loop: cffts3_pos stage nest at ft.c:892
- Iterations: same shape as cffts3_neg (64*64*64 with stage l=6)
- Type: C1 - FFT butterfly (forward)
- Parent loop: setup section before iterations via fft(1)
- Contains: stage i1/k1 butterflies
- Dependencies: stage dependency per FFT
- Nested bounds: constant
- Private vars: j,i,k,l,i1,k1,i11,i12,i21,i22,uu*,x*,temp*
- Arrays: u1_real/u1_imag (R/W), u0_real/u0_imag (W), gty1/gty2 scratch (RW), u_real/u_imag (R)
- Issues: setup-only

## Loop: cffts2_pos stage nest at ft.c:577
- Iterations: shape like cffts2_neg
- Type: C1 - FFT butterfly (forward)
- Parent loop: setup section before iterations via fft(1)
- Contains: stage i1/k1 butterflies
- Dependencies: stage dependency
- Nested bounds: constant
- Private vars: k,i,j,l,i1,k1,i11,i12,i21,i22,uu*,x*,temp*
- Arrays: u1_real/u1_imag (R/W), gty1/gty2 scratch (RW), u_real/u_imag (R)
- Issues: setup-only

## Loop: cffts1_pos stage nest at ft.c:473
- Iterations: shape like cffts1_neg
- Type: C1 - FFT butterfly (forward)
- Parent loop: setup section before iterations via fft(1)
- Contains: stage i1/k1 butterflies
- Dependencies: stage dependency
- Nested bounds: constant
- Private vars: k,j,i,l,i1,k1,i11,i12,i21,i22,uu*,x*,temp*
- Arrays: u1_real/u1_imag (R/W), gty1/gty2 scratch (RW), u_real/u_imag (R)
- Issues: setup-only

## Loop: init_ui k-j-i at ft.c:221
- Iterations: d3*d2*d1 = 262,144
- Type: A - dense zero-initialization
- Parent loop: none (pre-timer setup)
- Contains: none
- Dependencies: none
- Nested bounds: constant
- Private vars: i,j,k
- Arrays: u0_real/u0_imag/u1_real/u1_imag/twiddle (W)
- Issues: not timed; initialization only

Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|
| main iter loop (ft.c:188) | A | CRITICAL | none | niter (6) | none | driver only |
| evolve (ft.c:241) | A | IMPORTANT | main iter | 262k per iter | none | none |
| cffts3_neg (ft.c:951) | C1 | CRITICAL | main iter via fft(-1) | 64*64*64 * stages | stage dependency | scratch swap |
| cffts2_neg (ft.c:683) | C1 | CRITICAL | main iter via fft(-1) | 64*64*64 * stages | stage dependency | scratch swap |
| cffts1_neg (ft.c:473) | C1 | CRITICAL | main iter via fft(-1) | 64*64*64 * stages | stage dependency | scratch swap |
| checksum (ft.c:1113) | F | AVOID | main iter | 1024 | reduction temp1/temp2 | <10K iters |
| compute_indexmap (ft.c:373) | A | SECONDARY | setup | 262k | none | setup-only |
| compute_initial_conditions outer (ft.c:273) | A/E | SECONDARY | setup | 64*64*128 | RNG recurrence inner | RNG sequential, memcpy |
| fft_init (ft.c:427) | A | SECONDARY | setup | ~63 | stage dependency via ln/ku | small |
| cffts3_pos (ft.c:892) | C1 | SECONDARY | setup via fft(1) | 64*64*64 * stages | stage dependency | scratch swap |
| cffts2_pos (ft.c:577) | C1 | SECONDARY | setup via fft(1) | 64*64*64 * stages | stage dependency | scratch swap |
| cffts1_pos (ft.c:473) | C1 | SECONDARY | setup via fft(1) | 64*64*64 * stages | stage dependency | scratch swap |
| init_ui (ft.c:221) | A | AVOID | setup | 262k | none | outside timer |
| ipow46 while (ft.c:325) | E | AVOID | compute_initial_conditions | log2(exponent) | recurrence on q/r | small |

Data Details
- Dominant compute loop: main iteration ft.c:188; heavy work in cffts*_neg butterflies (O(N log N)) and evolve (O(N)).
- Arrays swapped between functions?: YES - u1_real/u1_imag filled in evolve, transformed in FFTs; u0_real/u0_imag used as output in forward FFT setup.
- Scratch arrays?: YES - gty1_real/gty1_imag and gty2_real/gty2_imag used as FFT stage buffers.
- Mid-computation sync?: NO explicit sync; stage ordering within FFT implies serial dependency per transform.
- RNG in timed loop?: YES in compute_initial_conditions (within timed T_total before iterations) but not inside per-iteration path.
