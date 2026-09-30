# 03 — The swap API, consumable from anywhere

Cell 8 tunnels ComfyUI's `127.0.0.1:8188` through a Cloudflare quick tunnel
(no account) and prints `API URL: https://<random-words>.trycloudflare.com`.

Three + one endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload/image` | POST multipart (`image=@file`, `overwrite=true`) | register your two input images; returns `{"name": ...}` |
| `/prompt` | POST `{"prompt": <graph>, "client_id": "bfs"}` | queue a workflow; returns `prompt_id` |
| `/history/<prompt_id>` | GET | `{}` while running; `outputs` + `status.status_str` on completion |
| `/view?filename=X&subfolder=&type=output` | GET | the PNG bytes |

The easy way is the bundled client (stdlib only, any machine):

    python3 clients/headswap.py --url $URL --mode head --body scene.png --face reference_head.png --out swapped.png
    python3 clients/headswap.py --url $URL --mode body --body scene.png --face reference_person.png

`--engine int8` reproduces the official-weights row; defaults are the measured
GGUF recipe. The workflow JSONs live in `workflows/` — submit them with curl
via `{"prompt": <graph>}` (note the outer `"prompt"` key). The graph MUST use
**flat** `images.image_1` / `images.image_2` keys (see 05-troubleshooting).
