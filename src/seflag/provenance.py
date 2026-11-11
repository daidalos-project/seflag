"""Collects everything needed to reproduce a result: software versions, model and data identity."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

TRACKED_PACKAGES: tuple[str, ...] = (
    "spacy", "pydantic", "datasets", "huggingface-hub", "torch", "transformers", "flair")


def package_versions(extra: list[str] | None = None) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (*TRACKED_PACKAGES, *(extra or [])):
        if not name or "/" in name:  # e.g. a model given as a directory path rather than a package
            continue
        try:
            versions[name] = version(name)
        except (PackageNotFoundError, ValueError):
            pass
    return versions


def collect(dataset_fingerprint: dict[str, Any], model_metadata: dict[str, Any],
            extra_packages: list[str] | None = None) -> dict[str, Any]:
    from seflag import __version__

    return {
        "seflag_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(extra_packages),
        "dataset": dataset_fingerprint,
        "model": model_metadata,
    }
