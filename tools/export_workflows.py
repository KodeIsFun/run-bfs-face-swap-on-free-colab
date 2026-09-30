#!/usr/bin/env python3
"""Export the API-format workflow JSONs from scripts/bfs_swap.py (the single
source of truth). Run after changing any graph builder:

    python3 tools/export_workflows.py
    python3 tools/check_workflows.py   # then confirm the gate is green
"""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_bfs():
    spec = importlib.util.spec_from_file_location("bfs_swap", ROOT / "scripts" / "bfs_swap.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    bfs = load_bfs()
    out = ROOT / "workflows"
    out.mkdir(exist_ok=True)
    exports = {
        "bfs-head-gguf.json": bfs.build_workflow_head("scene.png", "reference_head.png",
                                                      engine="gguf"),
        "bfs-head-int8.json": bfs.build_workflow_head("scene.png", "reference_head.png",
                                                      engine="int8"),
        "bfs-body-gguf.json": bfs.build_workflow_body("scene.png", "reference_person.png",
                                                      engine="gguf"),
    }
    for name, wf in exports.items():
        (out / name).write_text(json.dumps(wf, indent=1) + "\n")
        print(f"wrote workflows/{name} ({len(wf)} nodes)")


if __name__ == "__main__":
    main()
