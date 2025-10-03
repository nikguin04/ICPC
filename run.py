import argparse
import subprocess
import sys
import pathlib
import shlex
import os


def read(p):
    return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")


def exepath(name):
    return name + (".exe" if os.name == "nt" else "")


def build_cpp():
    out = exepath("main")
    r = subprocess.run(["g++", "main.cpp", "-O3", "-std=c++17",
                       "-o", out], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or "compile failed", file=sys.stderr)
        sys.exit(2)
    return ("./" if os.name != "nt" else "") + out


def parse_exts(s):
    exts = []
    for e in s.split(","):
        e = e.strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        exts.append(e)
    return exts or [".in"]


def gather_inputs(tdir, exts):
    files = []
    for e in exts:
        files.extend(tdir.glob(f"*{e}"))
    return sorted(set(files), key=lambda p: p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("lang", nargs="?", choices=["py", "c"])
    ap.add_argument("--in-ext", default=".in,.iin")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--args", default="")
    ap.add_argument("--sep", default="-----")
    args = ap.parse_args()

    root = pathlib.Path(args.folder).resolve()
    if not root.exists() or not root.is_dir():
        print("folder not found", file=sys.stderr)
        sys.exit(2)
    os.chdir(root)

    lang = args.lang
    if not lang:
        if pathlib.Path("main.py").exists():
            lang = "py"
        elif pathlib.Path("main.cpp").exists():
            lang = "c"
        else:
            print("missing lang and no main.py or main.cpp", file=sys.stderr)
            sys.exit(2)

    if lang == "py":
        exe_cmd = [sys.executable, "main.py"]
    else:
        exe_cmd = [build_cpp()]

    exts = parse_exts(args.in_ext)
    ins = gather_inputs(pathlib.Path("."), exts)
    if not ins:
        print("no input files", file=sys.stderr)
        sys.exit(2)

    first = True
    for inf in ins:
        try:
            cp = subprocess.run(
                [*exe_cmd, *shlex.split(args.args)],
                input=read(inf),
                capture_output=True,
                text=True,
                timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            if not first:
                print(args.sep)
            print(f"[TLE] {inf.name}")
            first = False
            continue

        if not first:
            print(args.sep)
        first = False

        if cp.returncode != 0:
            print(f"[RE] {inf.name} rc={cp.returncode}")
            if cp.stderr:
                print(cp.stderr.rstrip(), file=sys.stderr)
            continue

        out = cp.stdout.rstrip()
        if out:
            print(out)


if __name__ == "__main__":
    main()
