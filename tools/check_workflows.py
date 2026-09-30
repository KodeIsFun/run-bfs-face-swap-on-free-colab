#!/usr/bin/env python3
"""Consistency gate: the BFS graphs exist in THREE places —
scripts/bfs_swap.py (the library), workflows/*.json (the exported API JSONs)
and clients/headswap.py (the stdlib remote client) — and MUST not drift.
Run after any change:

    python3 tools/check_workflows.py

Graphs are compared node-by-node after ignoring the inputs that are *supposed*
to differ between copies (LoadImage filenames, SaveImage filename_prefix, the
prompt text, the seed). Everything else — loaders, LoRA stack, ImageScale
sizes, sampler wiring, the FLAT images.image_N keys, the latent routing — must
be identical. Also checks: the notebook parses and its cells compile, and the
sample inputs referenced by the notebook ship in samples/inputs/.
"""

import ast
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IGNORED = {"image", "filename_prefix", "prompt", "seed"}
CASES = [("head", "gguf"), ("head", "int8"), ("body", "gguf")]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def normalize(g):
    out = {}
    for node_id, node in g.items():
        inputs = {k: v for k, v in node["inputs"].items() if k not in IGNORED}
        out[node_id] = {"class_type": node["class_type"], "inputs": inputs}
    return out


def main():
    bfs = load_module(ROOT / "scripts" / "bfs_swap.py", "bfs_swap")
    client = load_module(ROOT / "clients" / "headswap.py", "headswap_client")
    fails = []

    for mode, engine in CASES:
        build = bfs.build_workflow_head if mode == "head" else bfs.build_workflow_body
        cbuild = client.build_workflow_head if mode == "head" else client.build_workflow_body
        lib = normalize(build("scene.png", "reference_head.png" if mode == "head"
                              else "reference_person.png", engine=engine))
        cli = normalize(cbuild("scene.png", "reference_head.png" if mode == "head"
                               else "reference_person.png", engine=engine))
        wf_path = ROOT / "workflows" / f"bfs-{mode}-{engine}.json"
        if not wf_path.exists():
            fails.append(f"missing workflows/bfs-{mode}-{engine}.json (run tools/export_workflows.py)")
            continue
        wf = normalize(json.loads(wf_path.read_text()))
        if lib != cli:
            fails.append(f"{mode}/{engine}: scripts/bfs_swap.py != clients/headswap.py")
        if lib != wf:
            fails.append(f"{mode}/{engine}: scripts/bfs_swap.py != workflows JSON")

    # sanity on the load-bearing gotchas
    head_gguf = bfs.build_workflow_head("s.png", "f.png", engine="gguf")
    if "images.image_1" not in head_gguf["30"]["inputs"]:
        fails.append("node 30 lost the FLAT 'images.image_1' keys (nested dict binds ZERO images)")
    if "images" in head_gguf["30"]["inputs"]:
        fails.append("node 30 carries a nested 'images' dict — silently unbound at /prompt")
    head_int8 = bfs.build_workflow_head("s.png", "f.png", engine="int8")
    if head_int8["22"]["inputs"].get("device") != "auto" or \
            head_int8["22"]["inputs"].get("dtype") != "default":
        fails.append("QwenImage21Cache missing explicit device/dtype (required in API format)")
    body = bfs.build_workflow_body("s.png", "p.png", engine="gguf")
    if body["32"]["inputs"]["latent_image"] != ["30", 2]:
        fails.append("body swap must take its latent from the TextEncodeQwenImage21 node")

    # notebook + samples
    nb_path = ROOT / "colab" / "run-bfs-faceswap-t4.ipynb"
    if not nb_path.exists():
        fails.append("missing colab/run-bfs-faceswap-t4.ipynb (run tools/build_notebook.py)")
    else:
        nb = json.loads(nb_path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                # strip IPython magics/shell-escapes ('!wget', '%pip') — they
                # are valid in the notebook runtime, not in pure Python
                py = "\n".join(l for l in "".join(cell["source"]).splitlines()
                               if not l.lstrip().startswith(("!", "%")))
                try:
                    compile(py, f"cell{i}", "exec")
                except SyntaxError as e:
                    fails.append(f"notebook cell {i} does not compile: {e}")
        src = "".join("".join(c["source"]) for c in nb["cells"])
        for marker in ["bfs_swap.py", "run_head", "run_body", "trycloudflare"]:
            if marker not in src:
                fails.append(f"notebook missing expected marker: {marker}")
    for fn in ["scene.png", "reference_head.png", "reference_person.png"]:
        p = ROOT / "samples" / "inputs" / fn
        if not p.exists() or p.stat().st_size < 1e5:
            fails.append(f"missing sample input samples/inputs/{fn}")

    if fails:
        print("GATE FAIL")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("GATE PASS: graphs identical across scripts/clients/workflows for all "
          f"{len(CASES)} cases; notebook compiles; gotcha invariants hold")


if __name__ == "__main__":
    main()
