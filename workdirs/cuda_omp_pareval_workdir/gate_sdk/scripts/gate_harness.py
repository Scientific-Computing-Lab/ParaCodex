#!/usr/bin/env python3
import os, re, subprocess, sys, math

RTOL = float(os.getenv("GATE_RTOL", "1e-6"))
ATOL = float(os.getenv("GATE_ATOL", "1e-7"))
RUNS = int(os.getenv("GATE_RUNS", "5"))

# Allow optional leading spaces and more tolerant numeric tokens (incl. nan/inf)
SUM_RE  = re.compile(r"^\s*GATE:SUM name=(\S+) dtype=(\S+) algo=(\S+) value=([0-9a-fA-F]+)")
STAT_RE = re.compile(r"^\s*GATE:STAT name=(\S+) dtype=(f32|f64) n=(\d+) min=([^\s]+) max=([^\s]+) mean=([^\s]+) L1=([^\s]+) L2=([^\s]+)")

def run_and_capture(cmd, extra_env=None):
    env = os.environ.copy()
    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
    if extra_env: env.update(extra_env)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, check=False, env=env)
    if p.returncode != 0:
        out = p.stdout if isinstance(p.stdout, str) else ""
        raise RuntimeError(f"subprocess failed (rc={p.returncode}) for: {' '.join(cmd)}\n" + out)
    out = p.stdout.splitlines()
    sums, stats = {}, {}
    for ln in out:
        m = SUM_RE.match(ln)
        if m:
            name,dtype,algo,val = m.groups()
            key = f"{name}:{dtype}"
            if key in sums:
                raise RuntimeError(f"duplicate checksum metric in single run: {key}")
            sums[key] = val.lower()
        m = STAT_RE.match(ln)
        if m:
            name,dtype,n,mi,mx,mean,l1,l2 = m.groups()
            key = f"{name}:{dtype}"
            if key in stats:
                raise RuntimeError(f"duplicate stats metric in single run: {key}")
            n_val = int(n)
            # Parse floats and enforce finiteness
            try:
                mi_v = float(mi); mx_v = float(mx); mean_v = float(mean); l1_v = float(l1); l2_v = float(l2)
            except ValueError:
                raise RuntimeError(f"non-numeric stat encountered for {key}")
            if not (math.isfinite(mi_v) and math.isfinite(mx_v) and math.isfinite(mean_v) and math.isfinite(l1_v) and math.isfinite(l2_v)):
                raise RuntimeError(f"non-finite (nan/inf) stat encountered for {key}")
            if n_val <= 0:
                raise RuntimeError(f"invalid sample count n<=0 for {key}")
            stats[key] = dict(n=n_val, min=mi_v, max=mx_v, mean=mean_v, L1=l1_v, L2=l2_v)
    return sums, stats, "\n".join(out)

def approx_eq(a,b):
    return abs(a-b) <= ATOL + RTOL*max(1.0, abs(a), abs(b))

def compare_stats(ref, cand, label):
    ok = True
    for k in ["min","max","mean","L1","L2"]:
        if not approx_eq(ref[k], cand[k]):
            print(f"[FAIL][{label}] {k}: ref={ref[k]} cand={cand[k]} (rtol={RTOL}, atol={ATOL})")
            ok=False
    return ok

def main():
    if len(sys.argv) < 3:
        print("Usage: gate_harness.py <ref_bin> <cand_bin> <args...>")
        sys.exit(2)
    ref_bin, cand_bin, *args = sys.argv[1:]

    try:
        ref_sums, ref_stats, ref_log = run_and_capture([ref_bin]+args)
    except Exception as e:
        print("[FAIL][ref_nonzero_exit] Reference program failed to run")
        print(str(e))
        sys.exit(2)

    first_sums=None; first_stats=None; last_log=""
    for i in range(RUNS):
        try:
            sums, stats, log = run_and_capture([cand_bin]+args)
        except Exception as e:
            print("[FAIL][cand_nonzero_exit] Candidate program failed to run")
            print(str(e))
            sys.exit(2)
        last_log = log
        if i==0:
            first_sums, first_stats = sums, stats
        else:
            if sums != first_sums:
                print("[FAIL][determinism] checksum changed between runs")
                print(last_log); sys.exit(2)
            # Enforce identical key sets for stats across runs
            if set(stats.keys()) != set(first_stats.keys()):
                print("[FAIL][determinism] stats key set changed between runs")
                print(last_log); sys.exit(2)
            # Enforce numerical determinism for all stats keys
            for k in first_stats.keys():
                if not compare_stats(first_stats[k], stats[k], f"det:{k}"):
                    print(last_log); sys.exit(2)

    # Check if either program produced any GATE output
    ref_has_gate_output = len(ref_sums) > 0 or len(ref_stats) > 0
    cand_has_gate_output = len(first_sums) > 0 or len(first_stats) > 0
    
    if not ref_has_gate_output and not cand_has_gate_output:
        print("[FAIL][no_gate_output] Neither reference nor candidate produced any GATE macros output")
        print("This indicates that neither program contains GATE_CHECKSUM_* or GATE_STATS_* macros")
        print("---- REF LOG ----"); print(ref_log)
        print("---- CAND LOG ----"); print(last_log)
        sys.exit(2)
    elif not ref_has_gate_output:
        print("[FAIL][no_ref_gate_output] Reference program produced no GATE macros output")
        print("The reference program should contain GATE_CHECKSUM_* or GATE_STATS_* macros")
        print("---- REF LOG ----"); print(ref_log)
        sys.exit(2)
    elif not cand_has_gate_output:
        print("[FAIL][no_cand_gate_output] Candidate program produced no GATE macros output")
        print("The candidate program should contain GATE_CHECKSUM_* or GATE_STATS_* macros")
        print("---- CAND LOG ----"); print(last_log)
        sys.exit(2)

    # Candidate must include at least all reference keys (extras allowed)
    missing_sums = set(ref_sums.keys()) - set(first_sums.keys())
    missing_stats = set(ref_stats.keys()) - set(first_stats.keys())
    if missing_sums:
        print(f"[FAIL][missing_candidate_checksums] Candidate missing checksum keys: {sorted(missing_sums)}")
        print("---- CAND LOG ----"); print(last_log)
        sys.exit(2)
    if missing_stats:
        print(f"[FAIL][missing_candidate_stats] Candidate missing stats keys: {sorted(missing_stats)}")
        print("---- CAND LOG ----"); print(last_log)
        sys.exit(2)

    ok=True
    for k,v in ref_sums.items():
        if k in first_sums and v != first_sums[k]:
            print(f"[FAIL][checksum] {k}: ref={v} cand={first_sums[k]}"); ok=False
    for k,v in ref_stats.items():
        if k in first_stats:
            if v['n'] != first_stats[k]['n']:
                print(f"[FAIL][n] {k}: ref={v['n']} cand={first_stats[k]['n']}")
                ok=False
            if not compare_stats(v, first_stats[k], k):
                ok=False

    if not ok:
        print("---- REF LOG ----"); print(ref_log)
        print("---- CAND LOG ----"); print(last_log)
        sys.exit(2)

    print("[Correctness Gate] PASS")

if __name__ == "__main__":
    main()