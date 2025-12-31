#!/usr/bin/env python3
"""
Create a performance comparison graph from results JSON file.
Handles the hecbench format with reference_ms, baseline_ms, and paracodex_ms.
"""
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def create_graph(json_path: Path, output_path: Path):
    """Create a comparison graph from JSON results."""
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if not data:
        print("Error: No data found in JSON file", file=sys.stderr)
        return False
    
    # Extract data
    names = [item["name"] for item in data]
    ref_vals = [item.get("reference_ms") if item.get("reference_ms") is not None else float("nan") for item in data]
    baseline_vals = [item.get("baseline_ms") if item.get("baseline_ms") is not None else float("nan") for item in data]
    paracodex_vals = [item.get("paracodex_ms") if item.get("paracodex_ms") is not None else float("nan") for item in data]
    
    # Check if baseline data exists
    has_baseline = any(v is not None and not (isinstance(v, float) and (v != v or v == float("nan"))) for v in baseline_vals)
    
    # Create figure
    x = list(range(len(names)))
    width = 0.25 if has_baseline else 0.35
    
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.5), 6))
    
    # Plot bars
    if has_baseline:
        ax.bar([i - width for i in x], ref_vals, width, label="Reference", color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar([i for i in x], paracodex_vals, width, label="ParaCodex", color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar([i + width for i in x], baseline_vals, width, label="Baseline", color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
    else:
        ax.bar([i - width/2 for i in x], ref_vals, width, label="Reference", color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar([i + width/2 for i in x], paracodex_vals, width, label="ParaCodex", color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms) [log scale]", fontsize=11)
    ax.set_yscale("log")
    title = "Performance Comparison: Reference vs ParaCodex" + (" vs Baseline" if has_baseline else "")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return True


def main():
    json_path = Path("/root/codex_baseline/cuda_omp_workdir/results_perf_nsys_baseline_hecbench/results.json")
    output_path = json_path.parent / "comparison.png"
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 1
    
    if create_graph(json_path, output_path):
        print(f"Generated plot: {output_path}")
        return 0
    else:
        print("Error: Failed to generate plot", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

