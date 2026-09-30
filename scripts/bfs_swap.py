#!/usr/bin/env python3
"""BFS (Best Face Swap) on a free T4 — the whole pipeline in one stdlib file.

Runs the Alissonerdx/BFS-Best-Face-Swap LoRAs (head swap v1.1, body swap
v1.0) on Qwen-Image-2.1 via ComfyUI, on a free Colab or Kaggle T4. Every
setting here was measured on a real free T4 (2026-09-30); the recipe and the
measured tables live in the repo README + references/. Library usage:

    import bfs_swap
    bfs_swap.setup_comfy()            # ComfyUI + nodes (idempotent)
    bfs_swap.download_weights("gguf") # GGUF engine (default, 2.3x faster)
    bfs_swap.boot()                   # headless server, measured T4 flags
    bfs_swap.ensure_inputs()          # consent-free sample inputs
    bfs_swap.run_head("scene.png", "reference_head.png", out="head.png")
    bfs_swap.run_body("scene.png", "reference_person.png", out="body.png")

CLI (same machine as the server, e.g. inside the notebook VM):
    python3 bfs_swap.py --mode head --body scene.png --face reference_head.png --out head.png

Remote (from any machine, against the notebook's tunnel URL) — use
clients/headswap.py instead; it speaks the same three HTTP endpoints.

The three T4-specific facts baked into this file (each paid for in a failed
run — see references/05-troubleshooting.md):
  1. ComfyUI's autogrow `images` input takes FLAT dotted keys
     ("images.image_1") in the /prompt API. A nested dict validates AND
     executes with ZERO images bound — every "swap" silently becomes plain
     text-to-image.
  2. The official template's 2MP references are ~40k DiT sequence tokens and
     cost 258.84 s/step on a T4. References are pre-scaled DOWN (scene 768²,
     head 512²) — the BFS author's own "give the reference room, not
     resolution" rule — while the latent stays at 1024².
  3. The T4 flag pair is mandatory: --force-fp16 (no bf16 on T4) AND
     --disable-comfy-compiler (fp16 otherwise dies in ComfyUI's compiler).
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import quote as urlquote

BASE = "http://127.0.0.1:8188"
SEED = 42
FLAGS = ["--force-fp16", "--disable-comfy-compiler"]
SAMPLER_LADDER = ["deis_2m", "res_2m", "deis", "res_multistep"]

# --- weights (all measured on a free T4) ---
GGUF_NAME = "qwen-image-2.1-Q4_K_M.gguf"            # unsloth, 4.20 GB — DEFAULT (2.3x faster)
GGUF_URL = f"https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/{GGUF_NAME}"
DIT_INT8_NAME = "qwen_image_2.1_int8_convrot.safetensors"  # Comfy-Org official, 7.26 GB
DIT_INT8_URL = ("https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/"
                f"diffusion_models/{DIT_INT8_NAME}")
TE_NAME = "qwen3vl_8b_int8_convrot.safetensors"     # 9.35 GB
TE_URL = f"https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/{TE_NAME}"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"    # 0.68 GB
VAE_URL = f"https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/{VAE_NAME}"
BFS_BASE = "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main"
BFS_HEAD = "bfs_head_v1.1_qwen_2.1.safetensors"     # recommended release, 260 MB
BFS_BODY = "bfs_body_swap_v1.0_qwen_2.1.safetensors"  # 210 MB
PRUNA8 = "p_qwen_image_2.1_8step_v0.1.safetensors"  # PrunaAI accelerator, 336 MB
PRUNA_URL = f"https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1/resolve/main/{PRUNA8}"
NODES_URL = "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/comfyui/viggle_turbo.py"

# the official BFS prompts (docs/qwen-image-2.1*.md), verbatim
HEAD_PROMPT = ("head_swap: start with <image1> as the base image, keeping its lighting, "
               "environment, and background. remove the head from <image1> completely and "
               "replace it with the head from <image2>, strictly preserving the hair, eye "
               "color, nose structure from <image2>. copy the direction of the eye, head "
               "rotation, micro expressions from <image1>, high quality, sharp details, 4k")
BODY_PROMPT = ("body_swap: start with <image1> as the base image, keeping its lighting, "
               "environment, and background. replace the body from <image1> with the body "
               "from <image2>, strictly preserving the clothing, body shape and proportions "
               "from <image2>. copy the pose, direction of the eye, head rotation, micro "
               "expressions from <image1>")

SAMPLE_INPUTS = ["scene.png", "reference_head.png", "reference_person.png"]
SAMPLES_RAW = ("https://raw.githubusercontent.com/KodeIsFun/run-bfs-face-swap-on-free-colab/"
               "main/samples/inputs")


def comfy_root():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "/tmp/ComfyUI"      # keep 16 GB of weights out of /kaggle/working's output cap
    if os.path.isdir("/content"):
        return "/content/ComfyUI"
    return os.path.expanduser("~/ComfyUI")


# --- graph builders (single source of truth; workflows/*.json are exported
# from these by tools/export_workflows.py, and clients/headswap.py carries a
# stdlib copy checked byte-for-byte by tools/check_workflows.py) ---

def build_workflow_head(body_image, face_image, seed=SEED, engine="gguf", prefix="bfs_head"):
    """BFS head swap: <image1> = body/base scene, <image2> = reference head.
    Keep the order — reversing it swaps the roles."""
    g = _common(body_image, face_image, seed, engine, prefix,
                scene_size=768, ref_size=512, prompt=HEAD_PROMPT)
    return g


def build_workflow_body(body_image, person_image, seed=SEED, engine="gguf", prefix="bfs_body"):
    """BFS body swap: <image1> = scene (pose/framing/bg kept), <image2> =
    reference person (face/clothing/proportions transferred). The sampler's
    latent comes from the text-encode node, sized to the scene."""
    g = _common(body_image, person_image, seed, engine, prefix,
                scene_size=768, ref_size=768, prompt=BODY_PROMPT)
    del g["31"]  # no EmptyLatentImage: latent comes from node 30 (output 2)
    g["32"]["inputs"]["latent_image"] = ["30", 2]
    return g


def _common(body_image, face_image, seed, engine, prefix, scene_size, ref_size, prompt):
    g = {}
    if engine == "gguf":
        g["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}}
        use_cache = False   # cache node unverified on the GGUF path — omitted
    else:
        g["1"] = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": DIT_INT8_NAME, "weight_dtype": "default"}}
        use_cache = True    # the official workflow includes QwenImage21Cache
    g["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": TE_NAME, "type": "qwen_image"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}}
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": body_image}}
    g["11"] = {"class_type": "LoadImage", "inputs": {"image": face_image}}
    g["12"] = {"class_type": "ImageScale",
               "inputs": {"upscale_method": "lanczos", "width": scene_size,
                          "height": scene_size, "crop": "disabled", "image": ["10", 0]}}
    g["13"] = {"class_type": "ImageScale",
               "inputs": {"upscale_method": "lanczos", "width": ref_size,
                          "height": ref_size, "crop": "disabled", "image": ["11", 0]}}
    # stock merge (LoraLoaderModelOnly) is the verified path on T4. The
    # "runtime hook" exact-update loader (ViggleTurboLora) is broken on
    # current ComfyUI master — see references/05-troubleshooting.md.
    g["20"] = {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": PRUNA8, "strength_model": 1.0}}
    g["21"] = {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["20", 0], "lora_name": BFS_HEAD, "strength_model": 1.0}}
    model_src = "21"
    if use_cache:
        # device/dtype are REQUIRED in the API format ('auto'/'default' = the
        # official workflow's own widget values)
        g["22"] = {"class_type": "QwenImage21Cache",
                   "inputs": {"model": [model_src, 0], "device": "auto", "dtype": "default"}}
        model_src = "22"
    g["30"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                          "negative_prompt": "", "resolution": 0,
                          # FLAT dotted keys — a nested dict binds ZERO images
                          "images.image_1": ["12", 0], "images.image_2": ["13", 0]}}
    g["31"] = {"class_type": "EmptyLatentImage",
               "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    g["32"] = {"class_type": "KSampler",
               "inputs": {"model": [model_src, 0], "positive": ["30", 0],
                          "negative": ["30", 1], "latent_image": ["31", 0],
                          "seed": seed, "steps": 8, "cfg": 1.0,
                          "sampler_name": "deis_2m", "scheduler": "simple", "denoise": 1.0}}
    g["33"] = {"class_type": "VAEDecode", "inputs": {"samples": ["32", 0], "vae": ["3", 0]}}
    g["34"] = {"class_type": "SaveImage", "inputs": {"images": ["33", 0],
                                                     "filename_prefix": prefix}}
    return g


# --- setup --------------------------------------------------------------------

def log(m):
    print(f"[bfs] {m}", flush=True)


def _sh(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, text=True, timeout=timeout)
    return r.returncode


def _expected_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def wget_resume(url, dest, min_bytes=1e6):
    """HEAD-verify + resume-loop (a truncated model must never render)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    want = _expected_size(url)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes and (
            not want or os.path.getsize(dest) >= want):
        log(f"cached: {os.path.basename(dest)}")
        return
    t0 = time.time()
    for attempt in range(1, 9):
        rc = subprocess.run(["wget", "-q", "-c", "--tries=3", "--timeout=120",
                             "--waitretry=10", "-O", dest, url]).returncode
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        if rc == 0 and size >= min_bytes and (not want or size >= want):
            log(f"downloaded {os.path.basename(dest)} {size/1e9:.2f} GB in {time.time()-t0:.0f}s")
            return
        log(f"retry {attempt}: {size/1e9:.2f}/{want/1e9:.2f} GB — resuming")
        time.sleep(10)
    sys.exit(f"download failed: {url}")


def setup_comfy():
    """ComfyUI + leejet GGUF fork + RES4LYF (deis_2m) + viggle node. Idempotent."""
    C = comfy_root()
    if not os.path.isdir(C):
        _sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {C}")
        _sh(f"pip install -q -r {C}/requirements.txt", timeout=1200)
    gg = f"{C}/custom_nodes/ComfyUI-GGUF"
    if not os.path.isdir(gg):
        _sh(f"git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF {gg}")
        _sh(f"pip install -q -r {gg}/requirements.txt", timeout=1200)
    res = f"{C}/custom_nodes/RES4LYF"
    if not os.path.isdir(res):
        _sh(f"git clone --depth 1 https://github.com/ClownsharkBatwing/RES4LYF {res}")
        req = f"{res}/requirements.txt"
        if os.path.exists(req):
            _sh(f"pip install -q -r {req}", timeout=900)
    node = f"{C}/custom_nodes/viggle_turbo.py"
    if not os.path.exists(node):
        subprocess.run(["wget", "-q", "--tries=3", "--timeout=60", "-O", node, NODES_URL])
    log(f"setup ok: {C}")
    return C


def download_weights(engine="gguf"):
    """DiT (per engine) + int8 text encoder + bf16 VAE + the three LoRAs."""
    C = comfy_root()
    m = f"{C}/models"
    files = [(TE_URL, f"{m}/text_encoders/{TE_NAME}", 9e9),
             (VAE_URL, f"{m}/vae/{VAE_NAME}", 6e8),
             (f"{BFS_BASE}/{BFS_HEAD}", f"{m}/loras/{BFS_HEAD}", 2.5e8),
             (f"{BFS_BASE}/{BFS_BODY}", f"{m}/loras/{BFS_BODY}", 2e8),
             (PRUNA_URL, f"{m}/loras/{PRUNA8}", 3e8)]
    if engine == "gguf":
        files.append((GGUF_URL, f"{m}/unet/{GGUF_NAME}", 4e9))  # UnetLoaderGGUF reads unet/
    else:
        files.append((DIT_INT8_URL, f"{m}/diffusion_models/{DIT_INT8_NAME}", 7e9))
    for url, dest, minb in files:
        if os.path.exists(dest) and os.path.getsize(dest) >= minb:
            continue
        wget_resume(url, dest, minb)
    log(f"weights ok ({engine} engine)")
    return C


def boot():
    """Launch ComfyUI headless with the measured T4 flags. Idempotent; kills
    any zombie server on the port first (a stale server answers preflight
    while the new one dies at bind)."""
    C = comfy_root()
    try:
        urllib.request.urlopen(BASE + "/system_stats", timeout=3)
        log("server already running")
        return
    except Exception:
        pass
    subprocess.run("pkill -f 'main.py --listen 127.0.0.1 --port 8188' || true", shell=True)
    time.sleep(3)
    logf = open(f"{os.path.dirname(C)}/comfyui.log", "w")
    global _boot_proc
    _boot_proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1",
                                   "--port", "8188", *FLAGS],
                                  cwd=C, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            urllib.request.urlopen(BASE + "/system_stats", timeout=5)
            log(f"boot ok ({time.time()-t0:.0f}s)")
            return
        except Exception:
            if _boot_proc_died():
                break
            time.sleep(3)
    sys.exit(f"ComfyUI failed to boot — log: {os.path.dirname(C)}/comfyui.log")


_boot_proc = None


def _boot_proc_died():
    return _boot_proc is not None and _boot_proc.poll() is not None


def preflight():
    """Fail in seconds with the actual enum lists, not after a 30-min image."""
    oi = _http("/object_info")
    need = ["UNETLoader", "UnetLoaderGGUF", "CLIPLoader", "VAELoader",
            "LoraLoaderModelOnly", "TextEncodeQwenImage21", "QwenImage21Cache",
            "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage", "LoadImage",
            "ImageScale"]
    missing = [n for n in need if n not in oi]
    assert not missing, f"missing nodes {missing} — custom_nodes not loaded? see troubleshooting"
    samplers = oi["KSampler"]["input"]["required"]["sampler_name"][0]
    avail = [s for s in SAMPLER_LADDER if s in samplers]
    assert avail, f"none of {SAMPLER_LADDER} available — RES4LYF missing?"
    types = oi["CLIPLoader"]["input"]["required"]["type"][0]
    assert "qwen_image" in types, f"qwen_image not in CLIPLoader types {types}"
    log(f"preflight ok (sampler={avail[0]})")
    return {"sampler": avail[0]}


def ensure_inputs(dest="."):
    """Fetch the consent-free sample inputs (generated at seed 42 — no real
    person; the BFS card forbids results involving non-consenting people)."""
    outs = []
    for fn in SAMPLE_INPUTS:
        p = os.path.join(dest, fn)
        if not os.path.exists(p) or os.path.getsize(p) < 1e5:
            urllib.request.urlretrieve(f"{SAMPLES_RAW}/{fn}", p)
        outs.append(p)
    log(f"inputs ready: {outs}")
    return outs


# --- run ----------------------------------------------------------------------

def _http(path, payload=None, timeout=60, base=None):
    base = base or BASE
    if payload is not None:
        req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(base + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def upload_image(path, base=None):
    """POST /upload/image (multipart) so LoadImage can find the file. Remote-
    friendly: pass base='https://<tunnel>.trycloudflare.com'."""
    boundary = "bfsformboundary7MA4YWxkTrZu0gW"
    fn = os.path.basename(path)
    body = ((f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="image"; filename="{fn}"\r\n'
             f"Content-Type: image/png\r\n\r\n").encode()
            + open(path, "rb").read()
            + (f"\r\n--{boundary}\r\n"
               f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
               f"--{boundary}--\r\n").encode())
    req = urllib.request.Request((base or BASE) + "/upload/image", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["name"]


def run_swap(mode, body_path, face_path, out="bfs_out.png", seed=SEED, engine="gguf",
             base=None, poll_s=1500):
    """Upload inputs (if remote/local-diff), submit, poll, save the PNG."""
    assert mode in ("head", "body")
    body_name = upload_image(body_path, base=base)
    face_name = upload_image(face_path, base=base)
    build = build_workflow_head if mode == "head" else build_workflow_body
    wf = build(body_name, face_name, seed=seed, engine=engine)
    pid = _http("/prompt", {"prompt": wf, "client_id": "bfs_repo"}, timeout=120,
                base=base)["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < poll_s:
        time.sleep(3)
        hist = _http(f"/history/{pid}", timeout=15, base=base)
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        imgs = [im["filename"] for o in entry.get("outputs", {}).values()
                for im in o.get("images", [])]
        if status.get("status_str") == "error":
            raise RuntimeError(f"comfy error: {json.dumps(status.get('messages', []))[:600]}")
        if imgs:
            data = urllib.request.urlopen(
                f"{(base or BASE)}/view?filename={urlquote(imgs[0])}&subfolder=&type=output",
                timeout=120).read()
            with open(out, "wb") as f:
                f.write(data)
            log(f"{mode} swap ok: {out} ({time.time()-t0:.0f}s)")
            return out
    raise TimeoutError(f"{mode} swap timed out after {poll_s}s")


def run_head(body_path, face_path, out="head_swap.png", seed=SEED, engine="gguf", base=None):
    return run_swap("head", body_path, face_path, out, seed, engine, base=base)


def run_body(body_path, person_path, out="body_swap.png", seed=SEED, engine="gguf", base=None):
    return run_swap("body", body_path, person_path, out, seed, engine, base=base)


# --- CLI ----------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="BFS face/body swap on a free T4 (local server)")
    ap.add_argument("--mode", choices=["head", "body"], required=True)
    ap.add_argument("--body", required=True, help="target body/scene image")
    ap.add_argument("--face", required=True, help="reference head (head mode) or person (body mode)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--engine", choices=["gguf", "int8"], default="gguf")
    a = ap.parse_args()
    setup_comfy()
    download_weights(a.engine)
    boot()
    preflight()
    out = a.out or f"{a.mode}_swap.png"
    fn = run_head if a.mode == "head" else run_body
    fn(a.body, a.face, out=out, seed=a.seed, engine=a.engine)


if __name__ == "__main__":
    main()
