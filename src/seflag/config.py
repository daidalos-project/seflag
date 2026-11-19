"""Configuration schema for evaluations.

A configuration has four main parts: the ``task`` to evaluate, the ``dataset`` with gold annotations,
the ``model`` that produces predictions, and ``options`` that control how predictions are compared
to the gold standard. Example (YAML)::

    name: latincy-lemma-ud
    task: lemmatization
    language: la
    dataset:
      source: conllu
      path: data/ud_latin_test
    model:
      backend: spacy
      name: la_core_web_lg
    options:
      label_normalization: [nfc]

The ``dataset``, ``model`` and ``task`` sections accept component-specific fields, which are validated against
the selected component's typed ``Config`` when the configuration is loaded (``seflag list -v`` shows all fields
and their allowed values). Fixed vocabularies are enums (:mod:`seflag.enums`); in Python, use them and the
components' ``Config`` classes instead of strings and dictionaries::

    from seflag import Language, Normalizer, Upos, load_config
    from seflag.backends.spacy import SpacyBackend
    from seflag.config import EvaluationOptions
    from seflag.data.conllu import ConlluDataset
    from seflag.tasks.lemmatization import LemmatizationTask

    config = load_config({
        "language": Language.LATIN,
        "task": LemmatizationTask.Config(exclude_upos=[Upos.PUNCT]),
        "dataset": ConlluDataset.Config(path="data/ud_latin_test"),
        "model": SpacyBackend.Config(name="la_core_web_lg"),
        "options": EvaluationOptions(label_normalization=[Normalizer.NFC]),
    })

Both forms produce the same configuration (and the same result files).
"""

from __future__ import annotations

import difflib
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from seflag.enums import (
    BackendName,
    ComponentKind,
    DatasetSource,
    Language,
    MultiwordTokens,
    Normalizer,
    SentenceSegmentation,
    TaskType,
    Tokenization,
)
from seflag.errors import ConfigError
from seflag.launcher import LauncherSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def component_name(kind: str, builtins: type[Enum], value: Any) -> Any:
    """A built-in component name as enum member, or the name of a registered plugin component.

    Unknown names are rejected with the closest registered name as suggestion, so that typos fail when the
    configuration is loaded rather than when the evaluation runs.
    """
    from seflag import registry

    try:
        return builtins(value)
    except ValueError:
        pass
    names: dict[str, str] = registry.available(kind)
    if isinstance(value, str) and value in names:
        return value
    close: list[str] = difflib.get_close_matches(str(value), list(names), n=1)
    hint: str = f"; did you mean '{close[0]}'?" if close else ""
    raise ValueError(f"unknown {kind[:-1]} {value!r}{hint} Available: {', '.join(sorted(names))}")


class ComponentSpec(BaseModel):
    """Selects a component by name; all other fields are passed on to that component.

    Instead of a dictionary, a component's typed ``Config`` object can be given, e.g.
    ``SpacyBackend.Config(name="la_core_web_lg")``; it is stored as the equivalent dictionary.
    """

    model_config = ConfigDict(extra="allow")
    kind: ClassVar[ComponentKind]
    key: ClassVar[str]  # field that names the component

    @classmethod
    def section(cls, data: Any) -> Any:
        """A typed component Config (or a spec) as the equivalent section dictionary; other values unchanged."""
        if isinstance(data, ComponentSpec):
            return data.model_dump(mode="json", exclude_none=True)
        if isinstance(data, BaseModel):
            from seflag import registry

            name: str | None = registry.name_of(cls.kind, type(data))
            if name is None:
                raise ValueError(f"{type(data).__qualname__} is not the Config of a registered {cls.kind[:-1]}; "
                                 f"give the section as a dictionary with '{cls.key}' instead")
            return {cls.key: name, **data.model_dump(mode="json", exclude_unset=True)}
        return data

    @model_validator(mode="before")
    @classmethod
    def _from_typed_config(cls, data: Any) -> Any:
        return data if isinstance(data, cls) else cls.section(data)

    @property
    def component(self) -> str:
        """The name of the selected component (built-in names as plain string)."""
        return str(getattr(self, self.key))

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class TaskSpec(ComponentSpec):
    kind = ComponentKind.TASKS
    key = "type"
    type: TaskType | str = Field(description="Task: a TaskType value or the name of a registered task.")

    @field_validator("type", mode="before")
    @classmethod
    def _known(cls, value: Any) -> Any:
        return component_name(ComponentKind.TASKS, TaskType, value)


class DatasetSpec(ComponentSpec):
    kind = ComponentKind.DATASETS
    key = "source"
    source: DatasetSource | str = Field(description="Dataset source: a DatasetSource value or a registered name.")

    @field_validator("source", mode="before")
    @classmethod
    def _known(cls, value: Any) -> Any:
        return component_name(ComponentKind.DATASETS, DatasetSource, value)


class ModelSpec(ComponentSpec):
    kind = ComponentKind.BACKENDS
    key = "backend"
    backend: BackendName | str = Field(description="Model backend: a BackendName value or a registered name.")
    launcher: LauncherSpec | None = Field(
        None, description="Run the backend in another Python environment (default: the current one).")

    @field_validator("backend", mode="before")
    @classmethod
    def _known(cls, value: Any) -> Any:
        return component_name(ComponentKind.BACKENDS, BackendName, value)


class EvaluationOptions(_Strict):
    tokenization: Tokenization = Field(
        Tokenization.GOLD,
        description="'gold': models receive the gold tokens, so predictions align 1:1 with the gold standard.")
    sentence_segmentation: SentenceSegmentation = Field(
        SentenceSegmentation.GOLD,
        description="'gold': each gold sentence is processed as exactly one sentence, so parsers cannot split it "
                    "(which would create several roots). 'model': the model may split sentences.")
    multiword_tokens: MultiwordTokens = Field(
        MultiwordTokens.SYNTACTIC_WORDS,
        description="How CoNLL-U multiword tokens (e.g. 'nobiscum' = 'nobis' + 'cum') are handled. "
                    "'syntactic_words': evaluate the individual syntactic words, ignore the surface token line.")
    input_normalization: list[Normalizer] = Field(
        default_factory=list, description="Normalizers applied to the tokens before they are given to the model.")
    label_normalization: list[Normalizer] = Field(
        default_factory=list,
        description="Normalizers applied to both gold and predicted string labels (e.g. lemmas) before comparison.")
    max_sentences: int | None = Field(None, ge=1, description="Evaluate only the first N sentences (for quick tests).")
    max_errors_reported: int = Field(50, ge=0, description="Number of most frequent error types listed in the results.")


class OutputOptions(_Strict):
    dir: Path | None = Field(
        Path("seflag_results"),
        description="Where result files are written, relative to the current working directory (not to the "
                    "configuration file); null disables writing.")


class EvaluationConfig(_Strict):
    name: str | None = None
    description: str | None = None
    language: Language = Field(description="Evaluated language: Language.LATIN ('la') or Language.ANCIENT_GREEK "
                                           "('grc').")
    task: TaskSpec
    dataset: DatasetSpec
    model: ModelSpec
    options: EvaluationOptions = Field(default_factory=EvaluationOptions)
    output: OutputOptions = Field(default_factory=OutputOptions)
    # directory against which relative dataset/model paths in the configuration are resolved; set by load_config()
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthands(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("task"), str):
            data = {**data, "task": {"type": data["task"]}}
        return data

    @model_validator(mode="after")
    def _validate_components(self) -> EvaluationConfig:
        """Checks each section's component options against the component's typed Config, already when loading."""
        from seflag import registry

        for label, spec in (("task", self.task), ("dataset", self.dataset), ("model", self.model)):
            validate_component_params(registry.get(spec.kind, spec.component), spec.params, label)
        return self

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        model_name: str = str(self.model.params.get("name", self.model.backend))
        return f"{self.task.type}-{self.language}-{model_name}"

    def resolve_path(self, path: str | Path) -> Path:
        p: Path = Path(path).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def load_config(source: str | Path | dict[str, Any], **overrides: Any) -> EvaluationConfig:
    """Loads and validates an evaluation configuration.

    ``source`` is either a path to a YAML/JSON file or an already parsed dictionary. Relative paths
    inside a configuration file are resolved relative to the directory of that file, except ``output.dir``,
    which is relative to the current working directory.
    ``overrides`` replace top-level sections, e.g. ``load_config(path, options={"max_sentences": 10})``.
    """
    base_dir: Path = Path.cwd()
    if isinstance(source, dict):
        data: dict[str, Any] = dict(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise ConfigError(f"Configuration file not found: {path}")
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        base_dir = path.resolve().parent
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    data.setdefault("base_dir", base_dir)
    try:
        return EvaluationConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration:\n{e}") from e


def validate_component_params(component: type, params: dict[str, Any], label: str) -> BaseModel:
    """Validates the component-specific part of a configuration section against ``component.Config``."""
    config_cls: type[BaseModel] = getattr(component, "Config", _Strict)
    try:
        return config_cls.model_validate(params)
    except ValidationError as e:
        hints: list[str] = []
        for error in e.errors():
            if error["type"] == "extra_forbidden" and error["loc"]:
                close = difflib.get_close_matches(str(error["loc"][0]), list(config_cls.model_fields), n=1)
                if close:
                    hints.append(f"'{error['loc'][0]}': did you mean '{close[0]}'?")
        fields: str = ", ".join(config_cls.model_fields) or "none"
        raise ConfigError(f"Invalid settings in section '{label}' ({component.__name__}):\n{e}"
                          + "".join(f"\n  {h}" for h in hints) + f"\n  Available settings: {fields}") from e
