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

The ``dataset``, ``model`` and ``task`` sections accept component-specific fields. These are validated
by the selected component (see ``seflag list`` for available components and their fields).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from seflag.errors import ConfigError
from seflag.launcher import LauncherSpec
from seflag.normalization import NORMALIZERS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(BaseModel):
    """Selects a component by name; all other fields are passed on to that component."""

    model_config = ConfigDict(extra="allow")

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class TaskSpec(ComponentSpec):
    type: str


class DatasetSpec(ComponentSpec):
    source: str


class ModelSpec(ComponentSpec):
    backend: str
    launcher: LauncherSpec | None = Field(
        None, description="Run the backend in another Python environment (default: the current one).")


class EvaluationOptions(_Strict):
    tokenization: Literal["gold"] = Field(
        "gold", description="'gold': models receive the gold tokens, so predictions align 1:1 with the gold standard.")
    sentence_segmentation: Literal["gold", "model"] = Field(
        "gold",
        description="'gold': each gold sentence is processed as exactly one sentence, so parsers cannot split it "
                    "(which would create several roots). 'model': the model may split sentences.")
    multiword_tokens: Literal["syntactic_words"] = Field(
        "syntactic_words",
        description="How CoNLL-U multiword tokens (e.g. 'nobiscum' = 'nobis' + 'cum') are handled. "
                    "'syntactic_words': evaluate the individual syntactic words, ignore the surface token line.")
    input_normalization: list[str] = Field(
        default_factory=list, description="Normalizers applied to the tokens before they are given to the model.")
    label_normalization: list[str] = Field(
        default_factory=list,
        description="Normalizers applied to both gold and predicted string labels (e.g. lemmas) before comparison.")
    max_sentences: int | None = Field(None, ge=1, description="Evaluate only the first N sentences (for quick tests).")
    max_errors_reported: int = Field(50, ge=0, description="Number of most frequent error types listed in the results.")

    @field_validator("input_normalization", "label_normalization")
    @classmethod
    def _known_normalizers(cls, value: list[str]) -> list[str]:
        unknown: list[str] = [v for v in value if v not in NORMALIZERS]
        if unknown:
            raise ValueError(f"unknown normalizer(s) {unknown}; available: {', '.join(NORMALIZERS)}")
        return value


class OutputOptions(_Strict):
    dir: Path | None = Field(Path("seflag_results"), description="Where result files are written; null disables writing.")


class EvaluationConfig(_Strict):
    name: str | None = None
    description: str | None = None
    language: str = Field(description="ISO 639 code of the evaluated language, e.g. 'la' or 'grc'.")
    task: TaskSpec
    dataset: DatasetSpec
    model: ModelSpec
    options: EvaluationOptions = Field(default_factory=EvaluationOptions)
    output: OutputOptions = Field(default_factory=OutputOptions)
    # directory against which relative paths in the configuration are resolved; set by load_config()
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthands(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("task"), str):
            data = {**data, "task": {"type": data["task"]}}
        return data

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
    inside a configuration file are resolved relative to the directory of that file.
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
        raise ConfigError(f"Invalid settings in section '{label}':\n{e}") from e
