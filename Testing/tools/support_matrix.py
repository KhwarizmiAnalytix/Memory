#!/usr/bin/env python3
"""Support matrix generated from executed manifests only (plan §9.3, task 0.6).

  record : run (or stat) a built test/benchmark binary and write a manifest JSON.
  render : read every manifest under a directory and print/update the matrix.

A cell is filled only by a manifest file; with none it shows "no manifest". Nothing is
inferred from source, CMake options or CI configuration. Manifests carry the git SHA and
tree state, so a dirty-tree result is visibly weaker than a clean one.

  python support_matrix.py record --exe bin/MemoryCopyCudaRuntimeTests.exe --gtest \
      --backend cuda --evidence shim --suite copy_runtime --repo . --out Docs/baselines/shim_copy_cuda.json
  python support_matrix.py render --dir Docs/baselines --update-plan Docs/memory_runtime_implementation_plan.md
"""
import argparse, glob, json, os, subprocess, sys, tempfile, time

BACKENDS = ["cpu", "cuda", "hip", "metal"]
EVIDENCE = ["compile", "shim", "hardware"]
BEGIN, END = "<!-- support-matrix:begin -->", "<!-- support-matrix:end -->"


def git(repo, *args):
    try:
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def record(a):
    sha = git(a.repo, "rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(git(a.repo, "status", "--porcelain", "--untracked-files=no"))
    exe = os.path.abspath(a.exe)
    m = {"suite": a.suite, "backend": a.backend, "evidence": a.evidence, "git_sha": sha,
         "git_tree": "dirty" if dirty else "clean", "binary": exe,
         "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}
    res = {}
    if a.evidence == "compile":
        if not os.path.exists(exe):
            print(f"no binary at {exe}: nothing to record", file=sys.stderr)
            return 2
        m["binary_mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(exe)))
        res = {"built": True}
    elif a.gtest:
        with tempfile.TemporaryDirectory() as td:
            j = os.path.join(td, "r.json")
            p = subprocess.run([exe, f"--gtest_output=json:{j}"], cwd=os.path.dirname(exe),
                               capture_output=True, text=True)
            try:
                d = json.load(open(j))
            except (OSError, ValueError):
                d = {}
            tests = d.get("tests", 0)
            skipped = sum(1 for s in d.get("testsuites", []) for t in s.get("testsuite", [])
                          if t.get("result") == "SKIPPED")
            res = {"exit_code": p.returncode, "tests": tests, "failures": d.get("failures", 0),
                   "errors": d.get("errors", 0), "skipped": skipped,
                   "passed": tests - d.get("failures", 0) - d.get("errors", 0) - skipped}
    else:
        p = subprocess.run([exe], cwd=os.path.dirname(exe), capture_output=True, text=True)
        res = {"exit_code": p.returncode}
    json.dump({"manifest": m, "result": res}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}: {res}")
    return 0 if res.get("exit_code", 0) == 0 else 1


def summarize(doc):
    m, r = doc.get("manifest", {}), doc.get("result")
    tag = f"{m.get('suite', '?')} @{m.get('git_sha', '?')}" + (" dirty" if m.get("git_tree") == "dirty" else "")
    if r is None:                                   # harness-style manifest
        n = len(doc.get("results", []))
        ok = sum(1 for x in doc.get("results", []) if x.get("status") == "ok")
        return f"{tag}: {ok}/{n} cases ran"
    if "tests" in r:
        extra = f", {r['skipped']} skipped" if r.get("skipped") else ""
        bad = f", {r['failures'] + r['errors']} FAILED" if r.get("failures") or r.get("errors") else ""
        return f"{tag}: {r['passed']}/{r['tests']} passed{extra}{bad}"
    if "built" in r:
        return f"{tag}: built"
    if "configs" in doc:
        return tag
    return f"{tag}: exit {r.get('exit_code')}"


def render(a):
    cells = {(b, e): [] for b in BACKENDS for e in EVIDENCE}
    for path in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        try:
            doc = json.load(open(path))
        except (OSError, ValueError):
            continue
        m = doc.get("manifest", doc)
        b, e = m.get("backend"), m.get("evidence")
        if (b, e) in cells:
            summ = summarize({"manifest": m, **{k: v for k, v in doc.items() if k != "manifest"}}) \
                if "manifest" in doc else summarize({"manifest": m, "configs": doc.get("configs")})
            cells[(b, e)].append(f"{summ} (`{os.path.basename(path)}`)")
    lines = ["| Backend | Compile | Deterministic shim | Hardware |", "|---|---|---|---|"]
    for b in BACKENDS:
        row = [" <br> ".join(cells[(b, e)]) or "no manifest" for e in EVIDENCE]
        lines.append(f"| {b} | " + " | ".join(row) + " |")
    table = "\n".join(lines)
    if not a.update_plan:
        print(table)
        return 0
    text = open(a.update_plan, encoding="utf-8", newline="").read()
    if BEGIN not in text or END not in text:
        print("plan has no support-matrix markers", file=sys.stderr)
        return 2
    nl = "\r\n" if "\r\n" in text else "\n"
    i, j = text.index(BEGIN) + len(BEGIN), text.index(END)
    text = text[:i] + nl + table.replace("\n", nl) + nl + text[j:]
    open(a.update_plan, "w", encoding="utf-8", newline="").write(text)
    print(table)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--exe", required=True); r.add_argument("--backend", required=True, choices=BACKENDS)
    r.add_argument("--evidence", required=True, choices=EVIDENCE); r.add_argument("--suite", required=True)
    r.add_argument("--repo", default="."); r.add_argument("--out", required=True)
    r.add_argument("--gtest", action="store_true")
    d = sub.add_parser("render")
    d.add_argument("--dir", required=True); d.add_argument("--update-plan")
    a = ap.parse_args()
    return record(a) if a.cmd == "record" else render(a)


if __name__ == "__main__":
    sys.exit(main())
