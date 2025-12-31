#!/usr/bin/env python3
"""
Script to regenerate comparison plot from existing results JSON file.
Supports baseline data if present.
"""
import json
import sys
from pathlib import Path
from typing import List, Optional

# Import the BenchmarkResult and plot_results from the main script
sys.path.insert(0, str(Path(__file__).parent))
from performance_comparison_nas import BenchmarkResult, plot_results

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_results_from_json(json_path: Path) -> List[dict]:
    """Load results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def convert_json_to_benchmark_results(data: List[dict]) -> tuple[List[BenchmarkResult], bool]:
    """Convert JSON data to BenchmarkResult objects and detect if baseline is present."""
    results = []
    has_baseline = False
    
    for item in data:
        # Check if baseline data exists
        if "baseline_kernel_ms" in item or "baseline_ms" in item:
            has_baseline = True
        
        result = BenchmarkResult(
            name=item["name"],
            reference_ms=item.get("reference_ms"),
            candidate_ms=item.get("candidate_ms"),
            reference_log=None,
            candidate_log=None,
            reference_kernel_ms=item.get("reference_kernel_ms"),
            reference_gpu_ms=item.get("reference_gpu_ms"),
            reference_full_ms=item.get("reference_full_ms"),
            candidate_kernel_ms=item.get("candidate_kernel_ms"),
            candidate_gpu_ms=item.get("candidate_gpu_ms"),
            candidate_full_ms=item.get("candidate_full_ms"),
            baseline_ms=item.get("baseline_ms"),
            baseline_kernel_ms=item.get("baseline_kernel_ms"),
            baseline_gpu_ms=item.get("baseline_gpu_ms"),
            baseline_full_ms=item.get("baseline_full_ms"),
        )
        results.append(result)
    
    return results, has_baseline


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate comparison plot from JSON results")
    parser.add_argument("--json", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output", type=str, help="Output PNG path (default: same dir as JSON, named comparison.png)")
    args = parser.parse_args()
    
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.parent / "comparison.png"
    
    try:
        data = load_results_from_json(json_path)
        if not data:
            print("Error: No data found in JSON file", file=sys.stderr)
            return 1
        
        # Convert to BenchmarkResult objects
        results, has_baseline = convert_json_to_benchmark_results(data)
        
        # Determine if measure_all is used (check if kernel_ms fields exist)
        measure_all = "reference_kernel_ms" in data[0] if data else False
        
        # Generate plot
        plot_path = plot_results(
            results, 
            output_path.parent, 
            gpu_time=False, 
            full_time=False, 
            measure_all=measure_all, 
            has_baseline=has_baseline
        )
        
        if plot_path:
            print(f"Generated plot: {plot_path}")
            return 0
        else:
            print("Error: Failed to generate plot (matplotlib issue?)", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

