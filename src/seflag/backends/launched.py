"""Proxy backend that runs the configured backend in another Python environment (see :mod:`seflag.launcher`)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab

from seflag.backends.base import Backend
from seflag.config import EvaluationConfig
from seflag.errors import SeflagError


class LaunchedBackend(Backend):
    description = "Runs another backend in a separate Python process and environment."

    def __init__(self, evaluation: EvaluationConfig, command: list[str]):
        super().__init__(None, evaluation)
        self.command: list[str] = command
        self._info: dict[str, Any] | None = None

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        with tempfile.TemporaryDirectory(prefix="seflag-") as tmp:
            request = Path(tmp) / "request.json"
            request.write_text(json.dumps({
                "config": self.evaluation.model_dump(mode="json"), "base_dir": str(self.evaluation.base_dir),
                "sentences": sentences,
            }), encoding="utf-8")
            output = Path(tmp) / "output"
            try:
                done = subprocess.run([*self.command, "-m", "seflag.worker", str(request), str(output)])
            except OSError as e:
                raise SeflagError(f"Could not start the backend with {self.command}: {e}") from e
            if done.returncode != 0:
                raise SeflagError(f"The backend failed in its separate process (exit code {done.returncode}); "
                                  f"command: {' '.join(self.command)}")
            docs: list[Doc] = list(DocBin().from_disk(output / "docs.spacy").get_docs(Vocab()))
            self._info = json.loads((output / "info.json").read_text(encoding="utf-8"))
        if len(docs) != len(sentences):
            raise SeflagError(f"The backend returned {len(docs)} documents for {len(sentences)} sentences.")
        return docs

    def label_inventory(self, layer: str) -> set[str] | None:
        labels = (self._info or {}).get("inventories", {}).get(layer)
        return None if labels is None else set(labels)

    def metadata(self) -> dict[str, Any]:
        info: dict[str, Any] = self._info or {}
        launcher = self.evaluation.model.launcher
        return {**info.get("metadata", {}), "launcher": {
            "spec": launcher.model_dump(exclude_none=True) if launcher else None, "command": self.command,
            "python": info.get("python"), "packages": info.get("packages")}}
