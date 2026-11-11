"""Runs a backend in the current Python environment on behalf of a seflag process elsewhere.

Usage (internal): ``python -m seflag.worker <request.json> <output dir>``. The request holds the evaluation
configuration and the sentences; the output directory receives the predictions as a spaCy ``DocBin``
(``docs.spacy``) and metadata (``info.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from spacy.tokens import DocBin

LAYERS: tuple[str, ...] = ("entities", "morph", "upos", "xpos")


def main(argv: list[str]) -> int:
    from seflag import provenance, registry
    from seflag.config import load_config, validate_component_params

    request_path, output_dir = Path(argv[0]), Path(argv[1])
    request: dict = json.loads(request_path.read_text(encoding="utf-8"))
    evaluation = load_config({**request["config"], "base_dir": request["base_dir"]})
    evaluation.model.launcher = None  # we are the launched process already
    component = registry.get("backends", evaluation.model.backend)
    backend = component(validate_component_params(component, evaluation.model.params, "model"), evaluation)
    docs = backend.predict(request["sentences"])
    output_dir.mkdir(parents=True, exist_ok=True)
    DocBin(docs=docs, store_user_data=False).to_disk(output_dir / "docs.spacy")
    inventories = {layer: sorted(labels) if (labels := backend.label_inventory(layer)) is not None else None
                   for layer in LAYERS}
    info = {"metadata": backend.metadata(), "inventories": inventories, "python": sys.version.split()[0],
            "packages": provenance.package_versions([str(evaluation.model.params.get("name", ""))])}
    (output_dir / "info.json").write_text(json.dumps(info, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
