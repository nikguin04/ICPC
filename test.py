#!/usr/bin/env python3
import argparse, subprocess, sys, pathlib, difflib, shlex

def read(p): 
    return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")

def norm_ws(s): 
    return " ".join(s.split())

def compare(a, b, mode):
    if mode == "exact": 
        return a == b, a, b
    if mode == "ws":
        return norm_ws(a) == norm_ws(b), a, b
    return a == b, a, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--in-ext", default=".in")
    ap.add_argument("--out-ext", default=".out")
    ap.add_argument("--mode", choices=["exact","ws"], default="ws")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--args", default="")
    args = ap.parse_args()

    exe = pathlib.Path(args.exe)
    if not exe.exists():
        print("executable not found", file=sys.stderr); sys.exit(2)

    tdir = pathlib.Path(args.tests)
    ins = sorted(tdir.glob(f"*{args.in_ext}"))
    if not ins:
        print("no input files", file=sys.stderr); sys.exit(2)

    total = 0
    passed = 0
    for inf in ins:
        outf = inf.with_suffix(args.out_ext)
        if not outf.exists():
            print(f"[SKIP] {inf.name} (missing {outf.name})")
            continue
        total += 1
        try:
            cp = subprocess.run(
                [str(exe), *shlex.split(args.args)],
                input=read(inf),
                capture_output=True,
                text=True,
                timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            print(f"[TLE]  {inf.name}"); continue
        if cp.returncode != 0:
            print(f"[RE]   {inf.name} rc={cp.returncode}")
            if cp.stderr: print(cp.stderr.strip())
            continue
        exp = read(outf)
        ok, got_raw, exp_raw = compare(cp.stdout, exp, args.mode)
        if ok:
            print(f"[OK]   {inf.name}")
            passed += 1
        else:
            print(f"[WA]   {inf.name}")
            g = got_raw.splitlines(keepends=False)
            e = exp_raw.splitlines(keepends=False)
            diff = difflib.unified_diff(e, g, fromfile="expected", tofile="got", lineterm="")
            for line in diff: print(line)

    print(f"\npassed {passed}/{total}")
    sys.exit(0 if passed == total and total > 0 else 1)


if __name__ == "__main__":
    main()