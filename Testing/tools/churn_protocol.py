#!/usr/bin/env python3
"""Churn reproduction protocol runner (plan Appendix B, task 0.5).

Runs the churn benchmark binary N times per configuration, stops a configuration at
its first crash or after N clean invocations (predeclared stop rule), moves any crash
dump out of the working directory so it cannot be overwritten, and writes a manifest.

  python churn_protocol.py --bin <build>/bin/benchmark_memory_cudacachingallocatorchurn.exe \
      --repo <source dir> --out Docs/baselines/churn_<sha>.json [--runs 30]

Configurations (fixed here so they are predeclared, not chosen after seeing results):
  bounded     : original iteration-capped benchmark binary, 10 repetitions
  uncapped    : churn binary, the historical reproduction command (time-based, no cap)
  warm_stress : churn binary, warm path only, 2,000,000 fixed iterations x 5 repetitions
  control     : churn binary, direct cudaMalloc/cudaFree only (no Memory code)
There is no RNG in the benchmark, so there is no seed; the unknown is thread/driver timing.
"""
import argparse, json, os, platform, shutil, subprocess, sys, time

CONFIGS = {
    # (binary key, flags). "churn" = benchmark_memory_cudacachingallocatorchurn,
    # "bench" = benchmark_memory_cudacachingallocator (iteration-bounded originals).
    "bounded": ("bench", ["--benchmark_repetitions=10"]),
    "uncapped": ("churn", ["--benchmark_min_time=0.05s", "--benchmark_repetitions=10"]),
    "warm_stress": ("churn", ["--benchmark_filter=BM_Churn_WarmAllocFree", "--benchmark_min_time=2000000x",
                              "--benchmark_repetitions=5"]),
    "control": ("churn", ["--benchmark_filter=BM_Churn_DirectMalloc", "--benchmark_min_time=0.05s",
                          "--benchmark_repetitions=10"]),
}
DUMPS = ("churn_crash.dmp", "churn_crash_report.txt")


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="churn benchmark binary")
    ap.add_argument("--bench-bin", help="iteration-capped CUDA benchmark binary (config: bounded)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--configs", default="bounded,uncapped,warm_stress,control")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    bin_path = os.path.abspath(a.bin)
    wd = os.path.dirname(bin_path)
    sha = sh(["git", "-C", a.repo, "rev-parse", "--short", "HEAD"])
    dirty = bool(sh(["git", "-C", a.repo, "status", "--porcelain", "--untracked-files=no"]))
    stale = [d for d in DUMPS if os.path.exists(os.path.join(wd, d))]
    quarantine = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"churn_dumps_{sha}")
    for d in stale:  # pre-existing artifacts are evidence from an earlier run: keep, do not overwrite
        os.makedirs(quarantine, exist_ok=True)
        shutil.move(os.path.join(wd, d), os.path.join(quarantine, "preexisting_" + d))

    manifest = {
        "suite": "churn_protocol", "backend": "cuda", "evidence": "hardware", "git_sha": sha, "git_tree": "dirty" if dirty else "clean",
        "host": platform.node(), "os": platform.platform(), "binary": bin_path,
        "binary_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(bin_path))),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "runs_per_config": a.runs, "stop_rule": "stop a config at its first crash or after runs clean invocations",
        "evidence_level": "real hardware, real driver",
        "preexisting_artifacts_moved": stale, "configs": {},
    }
    for name in a.configs.split(","):
        which, flags = CONFIGS[name]
        exe = os.path.abspath(a.bench_bin) if which == "bench" else bin_path
        if exe is None or not os.path.exists(exe):
            print(f"skip {name}: binary missing"); continue
        rec = {"binary": exe, "flags": flags, "invocations": [], "crashed": False}
        for i in range(a.runs):
            t0 = time.time()
            try:
                p = subprocess.run([exe] + flags, cwd=os.path.dirname(exe), capture_output=True, text=True,
                                   timeout=a.timeout)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            entry = {"n": i + 1, "exit_code": rc, "seconds": round(time.time() - t0, 1)}
            moved = []
            for d in DUMPS:
                src = os.path.join(wd, d)
                if os.path.exists(src):
                    os.makedirs(quarantine, exist_ok=True)
                    dst = os.path.join(quarantine, f"{name}_{i + 1}_{d}")
                    shutil.move(src, dst)
                    moved.append(dst)
            entry["dump_files"] = moved
            rec["invocations"].append(entry)
            print(f"{name} #{i + 1}: rc={rc} {entry['seconds']}s {'DUMP' if moved else ''}", flush=True)
            if rc != 0:
                rec["crashed"] = True
                break
        n = len(rec["invocations"])
        rec["summary"] = (f"{n} invocations, crashed at #{n}" if rec["crashed"]
                          else f"{n} clean invocations (95% upper bound on per-invocation crash rate ~ {3 / n:.0%}; "
                               "absence of a crash is not a proof)")
        manifest["configs"][name] = rec
    with open(a.out, "w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
