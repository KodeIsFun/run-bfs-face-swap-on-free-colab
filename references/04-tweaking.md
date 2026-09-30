# 04 — Measured tables (free T4s, 2026-09-30, seed 42)

## Engines (head swap, 1024², same inputs/seed)

| Engine | DiT weights | wall | s/step | ArcFace sim → reference | beret kept |
|---|---|---|---|---|---|
| **gguf (default)** | unsloth Q4_K_M, 4.20 GB | **96.1 s** | 10.54 | 0.632 | yes |
| int8 | Comfy-Org int8_convrot, 7.26 GB | 222.2 s | 20.71 | 0.640 | yes |

Same output quality by eye; GGUF is 2.3× cheaper. The int8 path includes
`QwenImage21Cache` (the official workflow's node); the GGUF path omits it.

## Resolution (official int8, references at template size)

| Latent | refs | wall | s/step | identity |
|---|---|---|---|---|
| 1024² | 768/512 | 222.2 s | 20.71 | transferred (red curls + freckles) |
| 1408² (template's 2MP budget) | 768/512 | 477.6 s | 51.96 | FAILED — target hair kept, posterized frame |
| (attempt-3 probe) 1408² | 1408/1408 | — | **258.84** | n/a — killed; ~35 min/image |

The 2MP cost is attention over the reference-latent sequence (~40k tokens);
`QwenImage21Cache` (auto) did not measurably relieve it on the int8+patches
path. Scaling references DOWN is the fix the BFS author themselves document
for body swap ("give the reference room, not resolution" — scaling the
reference UP to the scene size measured *worst*).

## The no-LoRA anchor (1024², int8, same conditioning)

| Row | wall | sim → reference head | sim → scene person | beret |
|---|---|---|---|---|
| BFS head swap | 222.2 s | 0.640 | 0.217 | kept |
| anchor (Pruna8 only) | 168.1 s | **0.867** | 0.086 | dropped |

The base model with the same two-image conditioning is already a strong head
swapper — it transfers the hair too, with higher face-likeness, and drops the
scene's hat. BFS's add: scene coherence (the beret) + body swap. Background
SSIM vs the input scene is 0.39–0.42 on every row (unrelated-image floor
0.083) — background fidelity is free either way.

## Body swap (int8, scene 768² + person 768², latent from the TE node)

270.3 s, 26.63 s/step; outfit + proportions transferred, pose/frame/bg kept;
small-in-frame face likeness 0.381 (upstream documents the same limitation).

Quality provenance: ArcFace = insightface buffalo_l (CPU); SSIM = block-SSIM
on grayscale; inputs are AI-generated at seed 42 (no real person). Raw
timings JSON: GPUTests `projects/2026-W40-bfs-headswap-t4/results/timings_bfs.json`.
