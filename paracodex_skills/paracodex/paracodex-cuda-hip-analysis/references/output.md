# Analysis Output Template

## File Conversion Mapping
```
original.cu → original.hip.cpp
main.cu → main.hip.cpp
```

## Migration Complexity
- **Overall Difficulty:** [Low/Medium/High]
- **Warp Size Assumptions:** [Detected/Not Detected] - Check for `32`, `0x1f`.
- **Inline PTX:** [Detected/Not Detected] - Requires manual rewrite.
- **Libraries used:** [cuBLAS, cuFFT, etc.]

## API Inventory
| API Group | Count | Notes |
|-----------|-------|-------|
| Memory (Malloc/Memcpy) | [N] | Auto-convertible |
| Kernel Launch | [N] | Auto-convertible |
| Atomics | [N] | Check for specific atomic support |
| Warp Shuffle | [N] | Check mask width |

## Kernel Details
For each critical kernel:
```
## Kernel: [name]
- **Launch config:** [grid/block]
- **Warp Assumptions:** [YES/NO]
- **Shared Memory:** [YES/NO]
- **Inline ASM:** [YES/NO]
```

## Action Items
1. [ ] Run `hipify-perl`
2. [ ] Manually fix Warp Size 32 assumptions
3. [ ] Replace Makefile compiler
