# 01 — What the human does by hand

1. Open the [notebook via the Colab badge](../README.md#what-you-get) (or Kaggle: fork the kernel).
2. Pick the T4 runtime (Colab: *Runtime → Change runtime type → T4 GPU*; Kaggle: *Settings → Accelerator → GPU T4 ×2* — one T4 is used).
3. *Run all*. First run is ~15 minutes (downloads ~14 GB on the GGUF engine); re-runs are warm (~4 min).
4. To swap your own photos: upload two images in cell 5's file panel and point `BODY_PATH`/`FACE_PATH` at them. **Consent required** — no public figures, no people who didn't agree.
5. Optional: cell 8's URL lets you swap from your laptop (`clients/headswap.py`).

Everything else — installs, weights, server, workflow, polling — is in the cells. Nothing here needs an account beyond Colab/Kaggle itself; the tunnel is account-less.
