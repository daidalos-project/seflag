"""Name -> component lookup for datasets, backends and tasks.

Built-in components are listed here as import strings, so heavy optional dependencies
(e.g. ``datasets``, ``flair``) are only imported when a config actually uses them.
Third-party packages can add components without touching seflag by declaring entry points
in their own ``pyproject.toml``::

    [project.entry-points."seflag.backends"]
    my_backend = "my_package.module:MyBackend"
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

from seflag.errors import ConfigError

BUILTINS: dict[str, dict[str, str]] = {
    "datasets": {
        "conllu": "seflag.data.conllu:ConlluDataset",
        "huggingface": "seflag.data.huggingface:HuggingFaceDataset",
        "columns": "seflag.data.columns:ColumnsDataset",
    },
    "backends": {
        "spacy": "seflag.backends.spacy:SpacyBackend",
        "transformers": "seflag.backends.transformers:TransformersBackend",
        "flair": "seflag.backends.flair:FlairBackend",
        "morpheus": "seflag.backends.morpheus:MorpheusBackend",
    },
    "tasks": {
        "lemmatization": "seflag.tasks.lemmatization:LemmatizationTask",
        "pos_tagging": "seflag.tasks.pos_tagging:PosTaggingTask",
        "morphology": "seflag.tasks.morphology:MorphologyTask",
        "dependency_parsing": "seflag.tasks.dependency_parsing:DependencyParsingTask",
        "ner": "seflag.tasks.ner:NerTask",
    },
}


def _import(target: str) -> Any:
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def available(kind: str) -> dict[str, str]:
    """Returns all registered components of a kind (``datasets``, ``backends``, ``tasks``) as name -> target."""
    found: dict[str, str] = dict(BUILTINS[kind])
    for ep in entry_points(group=f"seflag.{kind}"):
        found.setdefault(ep.name, ep.value)
    return found


def get(kind: str, name: str) -> Any:
    """Resolves a component class by its registered name."""
    components: dict[str, str] = available(kind)
    if name not in components:
        raise ConfigError(f"Unknown {kind[:-1]} '{name}'. Available: {', '.join(sorted(components))}")
    return _import(components[name])
