# 05 — Troubleshooting: the five gotchas (all paid for in real failed runs)

1. **Your "swap" looks like plain text-to-image and ignores both photos.**
   The `/prompt` API takes autogrow image inputs as FLAT dotted keys —
   `"images.image_1": ["12", 0], "images.image_2": ["13", 0]`. A nested
   `{"images": {...}}` dict VALIDATES and EXECUTES with zero images bound:
   no error, just a generic portrait from the prompt text. Proof it happened
   to us: the outputs were a studio man for `head_swap` and a forest child
   for `body_swap`; the executed-node list had no LoadImage/ImageScale nodes.

2. **`Prompt outputs failed validation ... required_input_missing: device/dtype`.**
   `QwenImage21Cache`'s `device` and `dtype` are required in the API format —
   pass `"device": "auto", "dtype": "default"` (the official workflow's own
   widget values).

3. **`aimdo memory compile error` on the first forward.** You used
   `--force-fp16` alone. On a T4 the pair is mandatory:
   `--force-fp16 --disable-comfy-compiler` (no bf16 on T4 → fp16 cast;
   ComfyUI's model compiler can't compile the fp16 path on this card).

4. **`QwenImage21Transformer2DModel has no attribute 'diffusion_model'`**
   from the Viggle runtime-hook LoRA loader (`viggle_turbo.py`). Wrapper API
   drift on ComfyUI master (worked 2026-09-24, broken 2026-09-30). Use stock
   `LoraLoaderModelOnly` merge — the BFS author ships it, quality is fine
   (int8 row measured with it).

5. **Server "boots" but weird 404s/timeouts after a crashed run.** A zombie
   ComfyUI holds port 8188 and answers `/system_stats` while your new server
   died at bind. Kill it first: `pkill -f 'main.py --listen 127.0.0.1 --port 8188'`.

Also: sampler `deis_2m` comes from **RES4LYF** (stock ComfyUI only has
`deis`); GGUF loading needs the **leejet** fork of ComfyUI-GGUF (city96
throws "Unknown model architecture!"); downloads must be HEAD-verified —
HF egress stalls mid-file on Colab, and a truncated model must never render.
