# 02 — Agent / headless lane (colab CLI)

The repo is written to be executed by a coding agent with Bash + Python only.

    # auth once (interactive, human): colab auth
    colab new -s bfs-verify --gpu T4
    colab upload <anything> /content/...           # upload requires local+remote paths
    git clone https://github.com/KodeIsFun/run-bfs-face-swap-on-free-colab /content/repo
    python3 /content/repo/scripts/bfs_swap.py --mode head --body /content/repo/samples/inputs/scene.png \
        --face /content/repo/samples/inputs/reference_head.png --out head.png
    # poll progress in /content/comfyui.log; artifacts land in the cwd
    colab download /content/head.png ./head.png -s bfs-verify
    colab stop -s bfs-verify                       # always stop

Gotchas paid for (see the lab's colab-cli notes): `colab upload` needs both
paths; `exec` takes Python files, not shell strings; download every artifact
and verify sizes BEFORE `colab stop`; Colab VMs recycle mid-session — detect
the surface via env (`COLAB_GPU`), not `/kaggle` paths (Colab ships stray
`/kaggle` dirs).
