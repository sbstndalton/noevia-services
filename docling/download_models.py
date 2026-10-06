"""Build-time model fetch for the docling image, pinned to exact Hugging Face commits.

`docling-tools models download layout tableformer` resolves the layout model at the floating
revision "main" (docling 2.129.0, `layout_model_specs`/`stage_model_specs`), so a rebuild after an
upstream push silently ships different weights. This script downloads the same three repositories
that command fetches for `layout` + `tableformer`, into the same directories
(`<out>/<repo with / replaced by -->`), but at the commit hashes below.

Run by the Dockerfile only; it is not copied into the runtime image. Needs network, so it is not
exercised by the pytest suite beyond a static check of the pins (test_requirements_pinned.py).

When docling is bumped, re-derive PINS from the new version's source (the repo ids and revisions it
passes to `download_hf_model`), resolve each revision to a commit with
`GET https://huggingface.co/api/models/<repo>/revision/<revision>`, and update the table.
"""

import sys
from pathlib import Path

# repo id -> (upstream revision docling 2.129.0 uses, commit that revision resolved to).
# The commit is what is downloaded; the revision is kept so a reader can see what moved.
PINS = {
    # layout model, PyTorch weights (LayoutObjectDetectionOptions default = Heron)
    "docling-project/docling-layout-heron": ("main", "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"),
    # layout model, ONNX export (the engine_overrides repo docling also fetches)
    "docling-project/docling-layout-heron-onnx": ("main", "40bde044036bb181c130ddf6c51792187268748f"),
    # TableFormer weights (TableStructureModel.download_models)
    "docling-project/docling-models": ("v2.3.0", "fc0f2d45e2218ea24bce5045f58a389aed16dc23"),
}


def main(out_dir: str) -> int:
    from huggingface_hub import snapshot_download

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for repo_id, (_revision, commit) in PINS.items():
        target = out / repo_id.replace("/", "--")
        print(f"fetching {repo_id}@{commit} -> {target}", flush=True)
        snapshot_download(repo_id=repo_id, revision=commit, local_dir=target)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: download_models.py <output-dir>")
    sys.exit(main(sys.argv[1]))
