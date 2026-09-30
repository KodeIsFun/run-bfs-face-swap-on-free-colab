#!/usr/bin/env python3
"""Independent end-to-end verification of this repo on a fresh GPU VM.

Assumes the repo is present (fresh clone) and a T4 is attached. Runs the
REAL pipeline exactly as the notebook does — install, weights, boot,
preflight, head swap (GGUF), body swap (GGUF), head swap (int8) — asserts
every artifact, and prints VERIFY OK. Used for the verification receipts in
colab/verification-*.log and the Kaggle kernel.

    python3 scripts/verify_run.py [--skip-int8]
"""
import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("bfs_swap", HERE / "bfs_swap.py")
bfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfs)

T0 = time.time()
RESULTS = []


def step(name, fn):
    print(f"[{time.time()-T0:7.1f}s] === {name} ===", flush=True)
    t0 = time.time()
    out = fn()
    RESULTS.append({"step": name, "s": round(time.time() - t0, 1), "ok": True, "out": out})
    print(f"[{time.time()-T0:7.1f}s] OK {name} ({time.time()-t0:.0f}s) -> {out}", flush=True)


def check_img(path, want_w, want_h):
    assert os.path.exists(path) and os.path.getsize(path) > 100_000, f"bad artifact: {path}"
    w, h = Image.open(path).size
    assert (w, h) == (want_w, want_h), f"{path}: {(w,h)} != {(want_w,want_h)}"
    return f"{path} {w}x{h} {os.path.getsize(path)//1024}KB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-int8", action="store_true")
    a = ap.parse_args()
    work = os.getcwd()

    step("setup_comfy", lambda: bfs.setup_comfy())
    step("weights gguf", lambda: bfs.download_weights("gguf"))
    step("boot", lambda: bfs.boot())
    step("preflight", lambda: bfs.preflight())
    scene, head, person = bfs.ensure_inputs(work)

    step("head swap (gguf)", lambda: bfs.run_head(scene, head, out="verify_head_gguf.png",
                                                  engine="gguf"))
    step("check head gguf", lambda: check_img("verify_head_gguf.png", 1024, 1024))
    step("body swap (gguf)", lambda: bfs.run_body(scene, person, out="verify_body_gguf.png",
                                                  engine="gguf"))
    step("check body gguf", lambda: check_img("verify_body_gguf.png", 768, 768))
    if not a.skip_int8:
        step("weights int8", lambda: bfs.download_weights("int8"))
        step("head swap (int8)", lambda: bfs.run_head(scene, head, out="verify_head_int8.png",
                                                      engine="int8"))
        step("check head int8", lambda: check_img("verify_head_int8.png", 1024, 1024))

    print(json.dumps(RESULTS, indent=1))
    print(f"VERIFY OK ({len(RESULTS)} steps, {time.time()-T0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
