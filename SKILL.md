---
name: run-bfs-face-swap-on-free-colab
description: Run BFS (Best Face Swap) head/body-swap LoRAs on Qwen-Image-2.1 with
  ComfyUI on a free Colab/Kaggle T4, and expose the ComfyUI API through a public
  tunnel. Use when the user wants face swap / head swap / body swap / outfit swap
  on a free GPU, a swap API without paying for GPU, or to tune BFS on quantized
  Qwen-Image weights. Includes a verified notebook, a Kaggle kernel, a stdlib-only
  client, the measured T4 recipe (GGUF engine, references pre-scaled 768/512,
  8 steps CFG off deis_2m, ~96 s per head swap), and every gotcha paid for in
  real failed runs (flat autogrow keys!, QwenImage21Cache widgets, zombie port,
  2MP refs = 258 s/step, hook loader broken on master).
---

# Run BFS face/body swap on a free Colab T4

Everything here was **measured on real free T4s** (2026-09-30, Colab torch
2.11.0+cu128), not assembled from forum posts. The numbers live in
[references/04-tweaking.md](references/04-tweaking.md) and trace to
[projects/2026-W40-bfs-headswap-t4](https://github.com/KodeIsFun/GPUTests/tree/main/projects/2026-W40-bfs-headswap-t4)
in the GPUTests lab repo.

## The one decision that routes everything

| The user wants | Path | First action |
|---|---|---|
| To see a face swap work | Notebook in browser | "Open in Colab" badge in the README → T4 runtime → Run all |
| Head swap AND body swap | Same notebook | Cells 6 and 7 run both on the consent-free samples |
| A swap API from anywhere | Same notebook | Cell 8 prints a `trycloudflare.com` URL → `clients/headswap.py --url $URL --mode head --body a.png --face b.png` |
| Kaggle instead of Colab | [kaggle/](kaggle/) | fork the public kernel or push `kernel-metadata.json` |
| Agent-driven headless run | clone this repo, run `scripts/bfs_swap.py --mode head --body scene.png --face reference_head.png` | see references/02-colab-cli-lane.md |
| Something broke | [references/05-troubleshooting.md](references/05-troubleshooting.md) | the five gotchas with exact error strings |

**Default recipe (measured):** Qwen-Image-2.1 **Q4_K_M GGUF** (unsloth, 4.20 GB)
+ int8 text encoder + bf16 VAE via ComfyUI (leejet GGUF fork + RES4LYF for
`deis_2m`), LoRAs **Pruna 8-step + BFS head v1.1 at strength 1.0 via stock
merge**, `TextEncodeQwenImage21` with **flat** `images.image_1/2` keys and
references pre-scaled **768 (scene) / 512 (head)**, 1024² latent, **8 steps,
CFG 1.0, deis_2m/simple, seed 42**, flags `--force-fp16 --disable-comfy-compiler`
→ **~96 s per head swap on a free T4**.

**Input order is part of the contract:** `<image1>` = body/scene (kept),
`<image2>` = reference head/person (transferred). Reversing the images swaps
the roles. The bundled sample inputs are AI-generated (seed 42, no real
person) — the model card forbids non-consenting subjects.

**Do not "fix" these:** nested `images` dict (binds zero images — silent t2i),
references at the latent size (258.84 s/step on T4), dropping
`--disable-comfy-compiler` (fp16 crash), the QwenImage21Cache device/dtype
widgets (required in API format), the viggle hook loader on current master
(broken — wrapper API drift).
