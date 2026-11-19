"""Name -> component lookup for datasets, backends and tasks.

Built-in components are listed here as import strings, so heavy optional dependencies
(e.g. ``datasets``, ``flair``) are only imported when a config actually uses them.
Third-party packages can add components without changing SEFLAG by declaring entry points
in their own ``pyproject.toml``::

    [project.entry-points."seflag.backends"]
    my_backend = "my_package.module:MyBackend"

For experiments (e.g. in a notebook), :func:`register` adds a component class directly.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

from seflag.enums import ComponentKind
from seflag.errors import ConfigError

BUILTINS: dict[str, dict[str, str]] = {
    "datasets": {
        "conllu": "seflag.data.conllu:ConlluDataset",
        "huggingface": "seflag.data.huggingface:HuggingFaceDataset",
        "columns": "seflag.data.columns:ColumnsDataset",
        "word_pairs": "seflag.data.lexical:WordPairsDataset",
        "word_choices": "seflag.data.lexical:WordChoicesDataset",
    },
    "backends": {
        "spacy": "seflag.backends.spacy:SpacyBackend",
        "transformers": "seflag.backends.transformers:TransformersBackend",
        "flair": "seflag.backends.flair:FlairBackend",
        "morpheus": "seflag.backends.morpheus:MorpheusBackend",
        "gensim": "seflag.backends.embeddings:GensimBackend",
    },
    "tasks": {
        "lemmatization": "seflag.tasks.lemmatization:LemmatizationTask",
        "pos_tagging": "seflag.tasks.pos_tagging:PosTaggingTask",
        "morphology": "seflag.tasks.morphology:MorphologyTask",
        "dependency_parsing": "seflag.tasks.dependency_parsing:DependencyParsingTask",
        "ner": "seflag.tasks.ner:NerTask",
        "word_similarity": "seflag.tasks.embeddings:WordSimilarityTask",
        "word_retrieval": "seflag.tasks.embeddings:WordRetrievalTask",
        "synonym_selection": "seflag.tasks.embeddings:SynonymSelectionTask",
    },
}


# components added at runtime with register(): kind -> name -> class
_REGISTERED: dict[str, dict[str, type]] = {kind: {} for kind in BUILTINS}


def register(kind: ComponentKind | str, name: str, component: type) -> None:
    """Makes a component class available under ``name`` for the rest of the session.

    ``kind`` is a :class:`~seflag.enums.ComponentKind`, e.g. ``ComponentKind.BACKENDS``; afterwards configurations
    can use the component, e.g. ``model: {backend: <name>}``. Registered components take precedence over built-ins
    and entry points.
    """
    try:
        kind = ComponentKind(kind)
    except ValueError:
        raise ValueError(f"Unknown component kind '{kind}'; expected one of {', '.join(ComponentKind)}") from None
    _REGISTERED[kind][name] = component


def _import(target: str) -> Any:
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def available(kind: str) -> dict[str, str]:
    """Returns all registered components of a kind (``datasets``, ``backends``, ``tasks``) as name -> target."""
    found: dict[str, str] = dict(BUILTINS[kind])
    for ep in entry_points(group=f"seflag.{kind}"):
        found.setdefault(ep.name, ep.value)
    found.update({name: f"{c.__module__}:{c.__qualname__}" for name, c in _REGISTERED[kind].items()})
    return found


def name_of(kind: str, cls: type) -> str | None:
    """The registered name of a component class, or of the component a nested ``Config`` class belongs to."""
    for name, registered in _REGISTERED[kind].items():
        if cls is registered or cls is getattr(registered, "Config", None):
            return name
    qualname: str = cls.__qualname__.removesuffix(".Config")
    target: str = f"{cls.__module__}:{qualname}"
    return next((name for name, t in available(kind).items() if t == target), None)


def get(kind: str, name: str) -> Any:
    """Resolves a component class by its registered name."""
    if name in _REGISTERED[kind]:
        return _REGISTERED[kind][name]
    components: dict[str, str] = available(kind)
    if name not in components:
        raise ConfigError(f"Unknown {kind[:-1]} '{name}'. Available: {', '.join(sorted(components))}")
    return _import(components[name])
