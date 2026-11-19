"""Building SEFLAG's components (dataset, model, task) from a configuration, also outside of an evaluation.

The runner and the command line use the ``build_*`` functions. ``load_dataset`` and ``load_backend`` are the
convenient entry points for using SEFLAG's data access and model wrappers in your own code::

    import seflag

    # the dataset of an evaluation configuration ...
    dataset = seflag.load_dataset("examples/latincy_lemma_ud.yaml")
    # ... or just a dataset section
    dataset = seflag.load_dataset({"source": "huggingface", "repo": "daidalos-project/latin_treebanks_ud_test",
                                   "split": "test"}, language="la")
    for record in dataset.records():          # SentenceRecord objects (CoNLL-U semantics)
        ...
    docs = dataset.docs()                     # spaCy Docs with every annotation layer the data has

    model = seflag.load_backend("examples/latincy_lemma_ud.yaml")
    predicted = model.predict([["Gallia", "est", "omnis", "divisa", "in", "partes", "tres"]])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from seflag import registry
from seflag.config import (
    DatasetSpec,
    EvaluationConfig,
    EvaluationOptions,
    OutputOptions,
    load_config,
    validate_component_params,
)
from seflag.enums import ComponentKind, Language
from seflag.errors import ConfigError


def build_task(config: EvaluationConfig) -> Any:
    """The configuration's task (a :class:`seflag.tasks.base.Task`)."""
    component: type = registry.get(ComponentKind.TASKS, config.task.type)
    return component(validate_component_params(component, config.task.params, "task"), config.options)


def build_dataset(config: EvaluationConfig) -> Any:
    """The configuration's dataset: a :class:`seflag.data.base.DatasetAdapter` (sentences) or a
    :class:`seflag.data.lexical.LexicalDataset` (word pairs, multiple-choice items)."""
    component: type = registry.get(ComponentKind.DATASETS, config.dataset.source)
    return component(validate_component_params(component, config.dataset.params, "dataset"), config)


def build_backend(config: EvaluationConfig) -> Any:
    """The configuration's model: a :class:`seflag.backends.base.Backend`, an embedding backend, or a backend that
    runs in another environment if ``model.launcher`` is set."""
    from seflag.backends.embeddings import EmbeddingBackend

    component: type = registry.get(ComponentKind.BACKENDS, config.model.backend)
    params = validate_component_params(component, config.model.params, "model")  # validate before launching
    if config.model.launcher is None:
        return component(params, config)
    if issubclass(component, EmbeddingBackend):
        raise ConfigError("'launcher' is not supported for embedding backends yet; run them in the current "
                          "environment.")
    from seflag.backends.launched import LaunchedBackend
    from seflag.launcher import python_command

    return LaunchedBackend(config, python_command(config.model.launcher))


_LABEL_KEYS: dict[str, tuple[str, ...]] = {"dataset": ("repo", "path", "url", "name", "source"),
                                           "model": ("name", "path", "url", "backend")}


def section_label(kind: str, section: dict[str, Any] | None) -> str | None:
    """A short name for a dataset or model section (e.g. its Hub repository or model name), used in tables."""
    for key in _LABEL_KEYS[kind]:
        value = (section or {}).get(key)
        if value:
            return ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    return None


def _config(source: str | Path | dict[str, Any] | EvaluationConfig, **overrides: Any) -> EvaluationConfig:
    if isinstance(source, EvaluationConfig):
        return source
    from seflag.matrix import is_matrix

    if is_matrix(source):
        raise ConfigError("This is a matrix configuration; load its cells with seflag.load_matrix().")
    return load_config(source, **overrides)


def load_dataset(source: str | Path | dict[str, Any] | BaseModel | EvaluationConfig,
                 language: Language | str | None = None, **overrides: Any) -> Any:
    """Loads a dataset without running an evaluation.

    ``source`` is a configuration (file, dictionary or :class:`EvaluationConfig`) or only its ``dataset`` section,
    as a dictionary (``{"source": "conllu", "path": "data/"}``) or a dataset's typed Config
    (``ConlluDataset.Config(path="data/")``); a dataset section needs ``language``, and
    relative paths in it are resolved against the current working directory. ``overrides`` update top-level
    sections of a configuration, as in :func:`seflag.load_config`.
    """
    only_dataset_section: bool = (
        (isinstance(source, BaseModel) and not isinstance(source, EvaluationConfig))  # DatasetSpec or a typed Config
        or (isinstance(source, dict) and "source" in source and "dataset" not in source))
    if only_dataset_section:
        if language is None:
            raise ConfigError("load_dataset() with a dataset section needs the language, e.g. "
                              "language=Language.LATIN.")
        try:
            dataset: DatasetSpec = DatasetSpec.model_validate(source)
            lang: Language = Language(language)
        except (ValidationError, ValueError) as e:
            raise ConfigError(f"Invalid dataset section or language:\n{e}") from e
        # dataset adapters only use the language, the options and the base directory of the configuration
        return build_dataset(EvaluationConfig.model_construct(
            name=None, description=None, language=lang, dataset=dataset, options=EvaluationOptions(),
            output=OutputOptions(), base_dir=Path.cwd()))
    if language is not None:
        overrides["language"] = language
    return build_dataset(_config(source, **overrides))


def load_backend(source: str | Path | dict[str, Any] | EvaluationConfig, **overrides: Any) -> Any:
    """Loads the model of a configuration (file, dictionary or :class:`EvaluationConfig`) without running an
    evaluation; ``predict()`` then annotates pre-tokenized sentences."""
    return build_backend(_config(source, **overrides))
