#!/usr/bin/env python3
"""BFS face/body swap against a remote ComfyUI — stdlib only, any machine.

The notebook (colab/run-bfs-faceswap-t4.ipynb) prints a public API URL
(Cloudflare quick tunnel). From ANY machine with Python 3:

    python3 clients/headswap.py --url https://<words>.trycloudflare.com \
        --mode head --body scene.png --face reference_head.png --out swapped.png
    python3 clients/headswap.py --url $URL --mode body --body scene.png \
        --face reference_person.png --out body.png

Options: --seed 42 --engine gguf|int8. Defaults = the measured recipe
(GGUF weights, 8 steps, CFG off, deis_2m, references pre-scaled 768/512).
Three endpoints are used: POST /upload/image, POST /prompt, GET /history,
GET /view — the same surface curl can drive (references/03-api-from-anywhere.md).
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from urllib.parse import quote as urlquote

SEED = 42

# the official BFS prompts (verbatim from the model card's guides)
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

GGUF_NAME = "qwen-image-2.1-Q4_K_M.gguf"
DIT_INT8_NAME = "qwen_image_2.1_int8_convrot.safetensors"
TE_NAME = "qwen3vl_8b_int8_convrot.safetensors"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"
BFS_HEAD = "bfs_head_v1.1_qwen_2.1.safetensors"
BFS_BODY = "bfs_body_swap_v1.0_qwen_2.1.safetensors"
PRUNA8 = "p_qwen_image_2.1_8step_v0.1.safetensors"


def build_workflow_head(body_image, face_image, seed=SEED, engine="gguf", prefix="bfs_head"):
    """MUST stay byte-identical (modulo the ignored inputs) to
    scripts/bfs_swap.py — tools/check_workflows.py enforces it."""
    g = _common(body_image, face_image, seed, engine, prefix,
                scene_size=768, ref_size=512, prompt=HEAD_PROMPT)
    return g


def build_workflow_body(body_image, person_image, seed=SEED, engine="gguf", prefix="bfs_body"):
    g = _common(body_image, person_image, seed, engine, prefix,
                scene_size=768, ref_size=768, prompt=BODY_PROMPT)
    del g["31"]
    g["32"]["inputs"]["latent_image"] = ["30", 2]
    return g


def _common(body_image, face_image, seed, engine, prefix, scene_size, ref_size, prompt):
    g = {}
    if engine == "gguf":
        g["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}}
        use_cache = False
    else:
        g["1"] = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": DIT_INT8_NAME, "weight_dtype": "default"}}
        use_cache = True
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
    g["20"] = {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": PRUNA8, "strength_model": 1.0}}
    g["21"] = {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["20", 0], "lora_name": BFS_HEAD, "strength_model": 1.0}}
    model_src = "21"
    if use_cache:
        g["22"] = {"class_type": "QwenImage21Cache",
                   "inputs": {"model": [model_src, 0], "device": "auto", "dtype": "default"}}
        model_src = "22"
    g["30"] = {"class_type": "TextEncodeQwenImage21",
               "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                          "negative_prompt": "", "resolution": 0,
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


def _req(url, payload=None, headers=None, data=None, timeout=120):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
        req.method = "POST"
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def upload_image(url, path):
    boundary = "bfsformboundary7MA4YWxkTrZu0gW"
    fn = os.path.basename(path)
    body = ((f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="image"; filename="{fn}"\r\n'
             f"Content-Type: image/png\r\n\r\n").encode()
            + open(path, "rb").read()
            + (f"\r\n--{boundary}\r\n"
               f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
               f"--{boundary}--\r\n").encode())
    r = _req(f"{url}/upload/image", data=body,
             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return r["name"]


def main():
    ap = argparse.ArgumentParser(description="BFS swap against a remote (tunneled) ComfyUI")
    ap.add_argument("--url", required=True, help="the API URL printed by the notebook")
    ap.add_argument("--mode", choices=["head", "body"], required=True)
    ap.add_argument("--body", required=True, help="target body/scene image (local file)")
    ap.add_argument("--face", required=True, help="reference head (head) or person (body)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--engine", choices=["gguf", "int8"], default="gguf")
    a = ap.parse_args()
    body_name = upload_image(a.url, a.body)
    face_name = upload_image(a.url, a.face)
    build = build_workflow_head if a.mode == "head" else build_workflow_body
    wf = build(body_name, face_name, seed=a.seed, engine=a.engine)
    pid = _req(f"{a.url}/prompt", payload={"prompt": wf, "client_id": "bfs_client"})["prompt_id"]
    print(f"submitted {a.mode} swap: {pid}")
    t0 = time.time()
    while time.time() - t0 < 1500:
        time.sleep(3)
        hist = _req(f"{a.url}/history/{pid}", timeout=15)
        if pid not in hist:
            continue
        entry = hist[pid]
        imgs = [im["filename"] for o in entry.get("outputs", {}).values()
                for im in o.get("images", [])]
        if entry.get("status", {}).get("status_str") == "error":
            sys.exit(f"comfy error: {json.dumps(entry['status'].get('messages', []))[:600]}")
        if imgs:
            data = urllib.request.urlopen(
                f"{a.url}/view?filename={urlquote(imgs[0])}&subfolder=&type=output",
                timeout=120).read()
            out = a.out or f"{a.mode}_swap.png"
            with open(out, "wb") as f:
                f.write(data)
            print(f"OK: {out} ({time.time() - t0:.0f}s)")
            return
    sys.exit("timed out")


if __name__ == "__main__":
    main()
