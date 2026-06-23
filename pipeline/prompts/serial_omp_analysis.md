# Loop Classification for GPU Offload - Analysis Phase

## Task
Analyze loops in `{source_dir}/` and produce `{kernel_dir}/analysis.md`. Copy source files unmodified to `{kernel_dir}/`.

**Files:** {file_listing}  
**Reference:** Check Makefile in `{kernel_dir}/` (do not modify)

## Process

> **NVIDIA GPU:** If running on an NVIDIA GPU, the compiler defined in the provided makefile must be used as the compiler. The provided Makefile already sets the right compiler — **do NOT change the compiler in the Makefile**.

### 0. COPY THE SOURCE FILES - {file_listing} TO THE KERNEL DIRECTORY {kernel_dir}

### 1. Find All Loops
```bash
# Find main compute loop
grep -n "for.*iter\|for.*it\|while\|main(" *.c *.cpp 2>/dev/null | head -50

# List all loop-containing functions
grep -n "for\s*(" *.c *.cpp 2>/dev/null | head -100
```

Prioritize functions called in main compute loop:
- Every iteration → CRITICAL/IMPORTANT
- Once at setup → SECONDARY/AVOID

### 2. Classify Priority
For each loop: `iterations × ops/iter = total work`

- **CRITICAL:** >50% runtime OR called every iteration with O(N) work
- **IMPORTANT:** 5-50% runtime OR called every iteration with small work
- **SECONDARY:** Called once at setup
- **AVOID:** Setup/IO/RNG OR <10K iterations

### 3. Determine Loop Type (Decision Tree)

```
Q0: Nested inside another loop? → Note parent
Q1: Writes A[idx[i]] with varying idx? → Type D (Histogram)
Q2: Reads A[i-1] or accumulates across iterations? → Type E (Recurrence - CPU only)
Q3: Stage loop where L+1 depends on L?
    - Scratch swap (tmp1↔tmp2)? → C1 (FFT/Butterfly)
    - Level traversal with stencil calls? → C2 (Multigrid)
Q4: Inner bound varies with outer index? → Type B (Sparse)
Q5: Accumulates to scalar? → Type F (Reduction)
Q6: Accesses neighbors? → Type G (Stencil)
Default → Type A (Dense)
```

**Special Case - Outer A + Inner E:**
When outer loop iterates over INDEPENDENT samples and inner has RNG:
- Mark outer as Type A (CRITICAL) - parallelizable with per-thread RNG
- Mark inner RNG as Type E - sequential WITHIN each thread
- Note: "RNG replicable: YES - each sample can compute its own seed"

### 4. Type Reference

| Type | Pattern | Parallelizable |
|------|---------|----------------|
| A | Dense, constant bounds | YES |
| B | Sparse (CSR), inner bound varies | Outer only |
| C1 | FFT/Butterfly, scratch swap | Outer only |
| C2 | Multigrid, hierarchical calls | Outer only |
| D | Histogram, indirect write | YES + atomic |
| E | Recurrence, loop-carried dep | NO |
| F | Reduction to scalar | YES + reduction |
| G | Stencil, neighbor access | YES |

### 5. Data Analysis
For each array:
- Definition: flat vs pointer-to-pointer
- Allocation: static vs dynamic
- Struct members accessed?
- Global variables used?

### 6. Flag Issues
- Variable bounds
- Reduction needed
- Atomic required
- Stage dependency
- RNG in loop
- <10K iterations

## Output: analysis.md

### Loop Nesting Structure
```
- outer_loop (line:X) Type A
  └── inner_loop_1 (line:Y) Type E
- standalone_loop (line:Z) Type A
```

### Loop Details
For each CRITICAL/IMPORTANT/SECONDARY loop:
```
## Loop: [function] at [file:line]
- **Iterations:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [inner loops or none]
- **Dependencies:** [none / reduction:vars / stage / recurrence]
- **Nested bounds:** [constant / variable]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW)]
- **Issues:** [flags]
```

### Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|

### Data Details
- **Dominant compute loop:** [main timed loop]
- **Arrays swapped between functions?:** YES/NO
- **Scratch arrays?:** YES/NO
- **Mid-computation sync?:** YES/NO
- **RNG in timed loop?:** YES/NO (only if inside timer)

## Constraints
- Find all loops in functions called from main compute loop
- Document only - no pragmas or code modifications
- When uncertain between B and C, choose C
- Copy all source files unmodified to `{kernel_dir}/`
```
