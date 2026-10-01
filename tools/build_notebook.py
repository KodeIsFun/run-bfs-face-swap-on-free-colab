#!/usr/bin/env python3
"""Assemble colab/run-bfs-faceswap-t4.ipynb (and the Kaggle variant) from the
cell sources below. Run after editing any cell:

    python3 tools/build_notebook.py && python3 tools/check_workflows.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "KodeIsFun/run-bfs-face-swap-on-free-colab"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"

MD_INTRO = f"""# BFS face & body swap on a free Colab T4

Runs the **BFS (Best Face Swap) LoRAs** ([Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap))
on **Qwen-Image-2.1** via ComfyUI headless: swap a face/head from a reference
photo onto a target body/scene, or swap the whole person's outfit — then
prints a **public API URL** so you can run more swaps from any machine.

- ⏱ **~15 minutes end to end** (install ~2 min → downloads ~5 min → boot ~1 min
  → head swap ~2–4 min → body swap ~2 min → tunnel 15 s). Measured on real
  free T4s; the numbers and gotchas live in [the repo]({RAW.replace('/main','')}) and the
  [GPUTests lab](https://github.com/KodeIsFun/GPUTests).
- ⚠️ **Requires the T4 runtime**: *Runtime → Change runtime type → T4 GPU →
  Save* (free tier), then *Runtime → Run all*. Cell 1 checks and warns.
- 🖼 Measured recipe baked in: GGUF Q4_K_M weights (2.3× faster than the
  official int8 on T4, same quality), Pruna 8-step accelerator + BFS LoRA
  (stock merge), 8 steps, CFG off, `deis_2m` sampler, references pre-scaled
  **768/512** (the template's 2MP references cost 258.84 s/step on a T4).
- ⚖️ **Consent**: the bundled sample inputs are AI-generated at a fixed seed
  (no real person). The BFS model card forbids results involving public
  figures or people who have not consented — use your own photos only with
  permission.
"""

CELL_GPU = """# Cell 1 — make sure this runtime actually has a GPU
import subprocess, sys
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    print("GPU:", out.strip())
    assert "T4" in out, "This recipe was measured on a T4; other GPUs may work but are unverified."
except Exception as e:
    print("WARNING: no GPU visible —", e)
    print("Runtime -> Change runtime type -> T4 GPU -> Save, then re-run.")
"""

CELL_INSTALL = f"""# Cell 2 — ComfyUI + the leejet GGUF fork + RES4LYF (deis_2m) + the BFS pipeline (idempotent)
# The pipeline lives in one stdlib file; the notebook fetches it so there is a
# single source of truth (the consistency gate keeps it in sync with the
# bundled workflows/*.json and clients/headswap.py).
!wget -q --tries=3 --timeout=60 {RAW}/scripts/bfs_swap.py -O bfs_swap.py
import bfs_swap, time
t0 = time.time()
bfs_swap.setup_comfy()
print(f"OK: install ({{time.time()-t0:.0f}}s)")
"""

CELL_WEIGHTS = """# Cell 3 — download the weights: GGUF DiT (default, 2.3x faster) + int8 text
# encoder + bf16 VAE + the three LoRAs (idempotent, resumes partial downloads)
import bfs_swap, time
t0 = time.time()
ENGINE = "gguf"   # "int8" = the official Comfy-Org weights from the BFS workflow
bfs_swap.download_weights(ENGINE)
print(f"OK: weights ({time.time()-t0:.0f}s)")
"""

CELL_BOOT = """# Cell 4 — boot ComfyUI headless with the measured T4 flags (idempotent).
# --force-fp16: T4 has no bf16 (default cast would be fp32, ~6x slower).
# --disable-comfy-compiler: with fp16 on a T4 the compiler otherwise dies
#   with "aimdo memory compile error" on the first forward. The pair is not
#   optional.
import bfs_swap, time
t0 = time.time()
bfs_swap.boot()
bfs_swap.preflight()
print(f"OK: boot + preflight ({time.time()-t0:.0f}s)")
"""

CELL_INPUTS = """# Cell 5 — get the two input images.
# Default: the repo's consent-free samples (AI-generated, fixed seed 42 — no
# real person). To use YOUR photos: upload them with the folder icon on the
# left, then set BODY_PATH / FACE_PATH to your filenames. Only photos you
# have permission to use — the BFS model card forbids non-consenting subjects.
import bfs_swap
BODY_PATH, FACE_PATH, PERSON_PATH = bfs_swap.ensure_inputs()
print("head swap:", BODY_PATH, "+", FACE_PATH)
print("body swap:", BODY_PATH, "+", PERSON_PATH)
"""

CELL_HEAD = """# Cell 6 — BFS HEAD swap: face/head from the reference onto the scene body.
# <image1> = body/scene (kept), <image2> = reference head (transferred) —
# keep that order. Measured: ~96 s warm on the GGUF engine.
import bfs_swap, time
t0 = time.time()
head_out = bfs_swap.run_head(BODY_PATH, FACE_PATH, out="head_swap.png", engine=ENGINE)
print(f"OK: head swap ({time.time()-t0:.0f}s) -> {head_out}")
from PIL import Image
Image.open(head_out)
"""

CELL_BODY = """# Cell 7 — BFS BODY swap: clothing/body proportions from the reference person
# onto the scene's pose, framing and background.
import bfs_swap, time
t0 = time.time()
body_out = bfs_swap.run_body(BODY_PATH, PERSON_PATH, out="body_swap.png", engine=ENGINE)
print(f"OK: body swap ({time.time()-t0:.0f}s) -> {body_out}")
Image.open(body_out)
"""

CELL_TUNNEL = """# Cell 8 — expose the API through a public quick tunnel (no account needed),
# then run more swaps from ANY machine with clients/headswap.py (stdlib only).
import os, re, subprocess, time, urllib.request
CF = "/tmp/cloudflared"   # /tmp exists on both Colab and Kaggle
if not os.path.exists(CF):
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CF)
    os.chmod(CF, 0o755)
subprocess.run(["pkill", "-f", "cloudflared tunnel"], check=False)  # fresh URL on re-run
time.sleep(1)
tun_log = open("/content/cloudflared.log", "w")
subprocess.Popen([CF, "tunnel", "--url", "http://localhost:8188", "--no-autoupdate"],
                 stdout=tun_log, stderr=subprocess.STDOUT)
url = None
t0 = time.time()
while url is None and time.time() - t0 < 60:
    time.sleep(2)
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com",
                  open("/content/cloudflared.log").read())
    url = m.group(0) if m else None
assert url, "tunnel failed - see /content/cloudflared.log"
print("API URL:", url)
print()
print(f"# from any machine:")
print(f"# python3 clients/headswap.py --url {url} --mode head --body scene.png --face reference_head.png --out swapped.png")
"""

MD_TWEAK = f"""## Cheat sheet (all measured on free T4s, 2026-09-30)

| Want | Do this | Cost |
|---|---|---|
| Faster head swap | keep `ENGINE = "gguf"` (default) | 96.1 s vs 222.2 s on int8, same quality |
| The official weights exactly | set `ENGINE = "int8"` in cell 3 and re-run cells 6–7 | 2.3× slower |
| Whole-body/outfit swap | cell 7 (BFS body LoRA) | ~270 s at 1408², ~2 min at 768² |
| Your own photos | upload in cell 5 (with consent!) | — |
| Call from anywhere | cell 8's URL + `clients/headswap.py` | stdlib only |

Known T4 limits (measured, see the repo's references/05-troubleshooting.md):
the template's 2MP references cost 258.84 s/step — this pipeline pre-scales
references to 768/512; at a 1408² latent identity transfer fails and the
frame goes posterized, so 1024² is the sweet spot; the "runtime hook" LoRA
loader is broken on current ComfyUI master, so the LoRAs load via stock merge.

Model licenses: BFS weights MIT; Qwen-Image-2.1 Qwen Research License
(experiments/research/demos — check before commercial use).
"""


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build_cells():
    return [
        md(MD_INTRO),
        code(CELL_GPU),
        code(CELL_INSTALL),
        code(CELL_WEIGHTS),
        code(CELL_BOOT),
        code(CELL_INPUTS),
        code(CELL_HEAD),
        code(CELL_BODY),
        code(CELL_TUNNEL),
        md(MD_TWEAK),
    ]


def main():
    for path in ("colab/run-bfs-faceswap-t4.ipynb", "kaggle/run-bfs-faceswap-t4.ipynb"):
        nb = {"cells": build_cells(),
              "metadata": {
                  "kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"},
                  **({"accelerator": "GPU"} if "kaggle" in path else {}),
                  **({"colab": {"provenance": []}} if "colab" in path else {}),
              },
              "nbformat": 4, "nbformat_minor": 5}
        p = ROOT / path
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
