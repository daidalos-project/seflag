"""File locations for file-based dataset sources: local paths or files inside a Hugging Face Hub repository."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seflag.config import EvaluationConfig
from seflag.errors import DataError, MissingDependencyError


def collect_files(entries: list[Path], pattern: str) -> list[Path]:
    """Expands directories to the files inside matching ``pattern``; keeps files as they are."""
    files: list[Path] = []
    for p in entries:
        if p.is_dir():
            found: list[Path] = sorted(p.glob(pattern))
            if not found:
                raise DataError(f"No files matching '{pattern}' in {p}")
            files += found
        elif p.is_file():
            files.append(p)
        else:
            raise DataError(f"Path does not exist: {p}")
    return files


def file_hashes(files: list[Path], relative_to: Path) -> dict[str, str]:
    return {os.path.relpath(p, relative_to): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


class FileSource(BaseModel):
    """Common fields of sources that read files.

    Without ``repo``, ``path`` is local (relative to the configuration file). With ``repo``, ``path`` is relative
    to the root of that Hub repository, which can be a dataset or a model repository; only the matching files
    are downloaded (and cached by ``huggingface_hub``). With ``url``, a single file is downloaded (e.g. from
    Zenodo), cached, and verified against ``checksum``.
    """

    model_config = ConfigDict(extra="forbid")
    path: str | list[str] = Field(".", description="A file, a directory, or a list of these.")
    pattern: str = Field("*", description="Glob pattern for files inside directories.")
    repo: str | None = Field(None, description="Hugging Face Hub repository to read the files from.")
    repo_type: Literal["dataset", "model"] = Field("dataset", description="Type of the Hub repository.")
    revision: str | None = Field(None, description="Hub revision; pin a commit hash for reproducible results.")
    url: str | None = Field(None, description="Download a single file from this URL instead (e.g. Zenodo).")
    checksum: str | None = Field(
        None, description="Expected checksum of the downloaded file, e.g. 'md5:...' (as on Zenodo) or 'sha256:...'.")

    @model_validator(mode="after")
    def _one_location(self) -> FileSource:
        if self.url is not None and self.repo is not None:
            raise ValueError("give either 'url' or 'repo', not both")
        if self.checksum is not None and self.url is None:
            raise ValueError("'checksum' is only used together with 'url'")
        return self


class FileResolver:
    """Resolves a :class:`FileSource` to local files, downloading from the Hub if needed."""

    def __init__(self, source: FileSource, evaluation: EvaluationConfig):
        self.source = source
        self.evaluation = evaluation
        self._snapshot: Path | None = None

    @property
    def entries(self) -> list[str]:
        return [self.source.path] if isinstance(self.source.path, str) else list(self.source.path)

    def root(self) -> Path:
        """Local directory that relative paths are resolved against."""
        if self.source.url is not None:
            return self._downloaded().parent
        if self.source.repo is None:
            return self.evaluation.base_dir
        if self._snapshot is None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise MissingDependencyError("Reading from the Hub needs huggingface_hub: pip install 'seflag[hf]'") from e
            patterns: list[str] = []
            for entry in self.entries:
                entry = entry.strip("/")
                if entry in ("", "."):
                    patterns.append(self.source.pattern)
                else:
                    patterns += [entry, f"{entry}/{self.source.pattern}"]
            self._snapshot = Path(snapshot_download(self.source.repo, repo_type=self.source.repo_type,
                                                    revision=self.source.revision, allow_patterns=patterns))
        return self._snapshot

    def _downloaded(self) -> Path:
        if self._snapshot is None:
            from seflag.data.download import download

            self._snapshot = download(self.source.url, self.source.checksum)
        return self._snapshot

    def files(self) -> list[Path]:
        if self.source.url is not None:
            return [self._downloaded()]
        root: Path = self.root()
        if self.source.repo is None:
            paths: list[Path] = [self.evaluation.resolve_path(e) for e in self.entries]
        else:
            paths = [root / e.strip("/") if e.strip("/") not in ("", ".") else root for e in self.entries]
        files: list[Path] = collect_files(paths, self.source.pattern)
        # never pick up files that the Hub client or git keep next to the data
        return [f for f in files if not f.name.startswith(".")]

    def fingerprint(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self.source.url is not None:
            info = {"url": self.source.url, "checksum": self.source.checksum}
        elif self.source.repo is not None:
            info = {"repo": self.source.repo, "repo_type": self.source.repo_type, "revision": self.source.revision,
                    "resolved_revision": self.root().name}  # snapshot directories are named after the commit
        info["files"] = file_hashes(self.files(), self.root())
        return info
