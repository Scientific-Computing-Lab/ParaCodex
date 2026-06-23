# Analysis Output Template

## Loop Nesting Structure
```
- outer_loop (line:X) Type A
  └── inner_loop_1 (line:Y) Type E
- standalone_loop (line:Z) Type A
```

## Loop Details
For each CRITICAL/IMPORTANT/SECONDARY loop:

### Loop: [function] at [file:line]
- **Iterations:** [count]
- **Type:** [A-H] - [reason]
- **Parent loop:** [none / line:X]
- **Contains:** [inner loops or none]
- **Dependencies:** [none / reduction:vars / stage / recurrence]
- **Nested bounds:** [constant / variable]
- **Private vars:** [list]
- **Arrays:** [name(R/W/RW)]
- **Issues:** [flags]

## Summary Table
| Function | Type | Priority | Parent | Iterations | Dependencies | Issues |
|----------|------|----------|--------|------------|--------------|--------|

## Structural Recommendations
- **Natural offload unit:** [single loop nest / fused routine / helper-by-helper]
- **Why:** [tie this to workload size, helper count, and synchronization shape]
- **Fragmentation risk:** [none / low / medium / high]
- **Would preserving helper boundaries likely create tiny kernels?** [YES/NO]

## Scalability Check
- **Default input likely too small to guide optimization?** [YES/NO]
- **Small-input sensitive?** [YES/NO]
- **Expected scaling risks:** [kernel count / transfer count / transfer volume / host sync / memory footprint / none]
- **Larger practical profile size recommendation:** [value or rule]
- **Why this size materially exercises the GPU:** [occupancy / kernel duration / transfer volume / enough parallel work / constrained by memory-time budget]
- **Hardware basis from `system_info_summary.txt`:** [device memory / expected load / practical short-run budget / why this is near the largest short-run safe size]

## Data Details
- **Dominant compute loop:** [main timed loop]
- **Arrays swapped between functions?:** YES/NO
- **Scratch arrays?:** YES/NO
- **Mid-computation sync?:** YES/NO
- **RNG in timed loop?:** YES/NO (only if inside timer)
- **Preferred GPU pragma family:** [`target teams distribute parallel for` / `target teams loop` / other]
