"""Evaluation matrices: several datasets × models in one configuration, summarized in a table.

A matrix configuration is an ordinary configuration with a ``matrix`` section. Its top-level sections are shared
by all cells; each entry under ``datasets`` and ``models`` is a ``label`` plus any of the sections ``language``,
``task``, ``dataset``, ``model`` and ``options``. A cell's configuration is the shared one, updated by the dataset
entry and then by the model entry (section by section: a ``model`` entry's ``options`` replace individual
options, e.g. its own ``label_normalization`` list). Example::

    name: lemmatization-ud
    task: lemmatization
    matrix:
      datasets:
        - label: UD Latin
          language: la
          dataset: {source: huggingface, repo: daidalos-project/latin_treebanks_ud_test, split: test}
        - label: UD Greek
          language: grc
          dataset: {source: huggingface, repo: daidalos-project/ancient_greek_treebanks_ud_test, split: test}
      models:
        - label: LatinCy lg
          language: la                  # not run on UD Greek
          model: {backend: spacy, name: la_core_web_lg}
        - label: Morpheus                 # no language: runs on both
          model: {backend: morpheus}
          options: {label_normalization: [strip_homonym_digits]}
      table:
        metrics: [accuracy]             # default: the task's headline metric
        by_subset: false                # true: one column per subset (e.g. treebank)

A cell whose dataset and model entries name different languages is not applicable and not run. Either axis may
be left out, if the shared configuration has a ``dataset`` or ``model`` section.

Each cell is an ordinary evaluation with its own result file. The matrix writes them, a summary JSON
(``matrix.json``) and the table as Markdown (``table.md``) to ``<output.dir>/<name>_<timestamp>/``. A cell that
fails (e.g. a model that is not installed) is reported in the table, and the other cells still run.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from seflag import __version__, registry
from seflag.components import section_label
from seflag.config import EvaluationConfig, OutputOptions, load_config
from seflag.errors import ConfigError, SeflagError
from seflag.runner import EvaluationResult, run

logger = logging.getLogger("seflag")

ENTRY_SECTIONS: tuple[str, ...] = ("language", "task", "dataset", "model", "options")
NOT_APPLICABLE: str = "–"


class AxisEntry(BaseModel):
    """One row (model) or column (dataset) of the matrix: a label and the sections it changes."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    label: str | None = Field(None, description="Name in the table; default: the model/dataset name.")
    language: str | None = None
    task: str | dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    model: dict[str, Any] | None = None
    options: dict[str, Any] | None = None

    def sections(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ENTRY_SECTIONS if getattr(self, key) is not None}


class TableOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics: list[str] | None = Field(
        None, description="Metrics shown per cell; nested ones with dots, e.g. 'unlabeled.f1'. Default: each task's "
                          "headline metric (accuracy, las, f1, spearman, ...).")
    by_subset: bool = Field(False, description="One column per dataset subset (e.g. treebank) next to the total.")
    percent: bool = Field(False, description="Show scores between 0 and 1 as percentages.")
    digits: int = Field(4, ge=0, le=10, description="Decimal places.")


class MatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    datasets: list[AxisEntry] = Field(default_factory=list)
    models: list[AxisEntry] = Field(default_factory=list)
    table: TableOptions = Field(default_factory=TableOptions)


@dataclass
class Cell:
    dataset: str
    model: str
    config: EvaluationConfig | None  # None: not applicable
    reason: str | None = None


@dataclass
class MatrixConfig:
    name: str
    spec: MatrixSpec
    cells: list[Cell]
    output_dir: Path | None
    source: dict[str, Any]  # the configuration as given, for the summary file

    @property
    def dataset_labels(self) -> list[str]:
        return list(dict.fromkeys(c.dataset for c in self.cells))

    @property
    def model_labels(self) -> list[str]:
        return list(dict.fromkeys(c.model for c in self.cells))


def is_matrix(source: str | Path | dict[str, Any]) -> bool:
    """Whether a configuration (file) describes an evaluation matrix, i.e. has a ``matrix`` section."""
    if isinstance(source, dict):
        return "matrix" in source
    try:
        data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False  # load_config reports the problem
    return isinstance(data, dict) and "matrix" in data


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Section-wise update: mapping sections are merged one level deep, everything else is replaced."""
    merged: dict[str, Any] = dict(base)
    for key, value in update.items():
        if key == "task" and isinstance(value, str):
            value = {"type": value}
        current = merged.get(key)
        if key == "task" and isinstance(current, str):
            current = {"type": current}
        merged[key] = {**current, **value} if isinstance(current, dict) and isinstance(value, dict) else value
    return merged


def _default_label(kind: str, section: dict[str, Any] | None, index: int) -> str:
    return section_label(kind, section) or f"{kind} {index + 1}"


def _labels(kind: str, entries: list[AxisEntry], base: dict[str, Any]) -> list[str]:
    labels: list[str] = [entry.label or _default_label(kind, _merge(base, entry.sections()).get(kind), i)
                         for i, entry in enumerate(entries)]
    duplicates: set[str] = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        raise ConfigError(f"Matrix {kind} labels must be unique; set 'label' for: {', '.join(sorted(duplicates))}")
    return labels


def load_matrix(source: str | Path | dict[str, Any], **overrides: Any) -> MatrixConfig:
    """Loads a matrix configuration and validates the configuration of every cell.

    ``overrides`` replace or update shared top-level sections, as in :func:`seflag.load_config`.
    """
    base_dir: Path = Path.cwd()
    name: str = "matrix"
    if isinstance(source, dict):
        data: dict[str, Any] = dict(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise ConfigError(f"Configuration file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        base_dir, name = path.resolve().parent, path.stem
    given: dict[str, Any] = json.loads(json.dumps(data, default=str))
    if "matrix" not in data:
        raise ConfigError("Not a matrix configuration: no 'matrix' section.")
    try:
        spec: MatrixSpec = MatrixSpec.model_validate(data.pop("matrix") or {})
    except ValidationError as e:
        raise ConfigError(f"Invalid 'matrix' section:\n{e}") from e
    base: dict[str, Any] = _merge(data, overrides)
    name = str(base.get("name") or name)
    if not spec.datasets and "dataset" not in base:
        raise ConfigError("The matrix needs 'matrix.datasets' or a shared 'dataset' section.")
    if not spec.models and "model" not in base:
        raise ConfigError("The matrix needs 'matrix.models' or a shared 'model' section.")
    datasets: list[AxisEntry] = spec.datasets or [AxisEntry()]
    models: list[AxisEntry] = spec.models or [AxisEntry()]
    dataset_labels: list[str] = _labels("dataset", datasets, base)
    model_labels: list[str] = _labels("model", models, base)

    cells: list[Cell] = []
    for d_entry, d_label in zip(datasets, dataset_labels, strict=True):
        for m_entry, m_label in zip(models, model_labels, strict=True):
            if d_entry.language and m_entry.language and d_entry.language != m_entry.language:
                cells.append(Cell(d_label, m_label, None,
                                  f"model language {m_entry.language}, dataset language {d_entry.language}"))
                continue
            cell_data: dict[str, Any] = _merge(_merge(base, d_entry.sections()), m_entry.sections())
            cell_data["name"] = f"{name}-{d_label}-{m_label}"
            cell_data.setdefault("base_dir", base_dir)
            try:
                config: EvaluationConfig = load_config(cell_data)
            except ConfigError as e:
                raise ConfigError(f"Matrix cell '{d_label}' × '{m_label}': {e}") from e
            cells.append(Cell(d_label, m_label, config))
    try:
        output_dir: Path | None = OutputOptions.model_validate(base.get("output") or {}).dir
    except ValidationError as e:
        raise ConfigError(f"Invalid 'output' section:\n{e}") from e
    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()  # relative to the working directory, like single runs
    return MatrixConfig(name=name, spec=spec, cells=cells, output_dir=output_dir, source=given)


@dataclass
class CellResult:
    dataset: str
    model: str
    status: str  # ok, failed, not_applicable
    reason: str | None = None
    result: EvaluationResult | None = None


def _lookup(values: dict[str, Any], metric: str) -> Any:
    value: Any = values
    for part in metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _headline(result: EvaluationResult) -> str | None:
    metric: str | None = getattr(registry.get("tasks", result.task), "primary_metric", None)
    if metric is not None and _lookup(result.metrics, metric) is not None:
        return metric
    return next((k for k, v in result.metrics.items() if isinstance(v, float)), None)


@dataclass
class MatrixResult:
    name: str
    cells: list[CellResult]
    table_options: TableOptions
    dataset_labels: list[str]
    model_labels: list[str]
    source: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"))
    path: Path | None = None  # directory with the matrix files, if saved

    @property
    def failed(self) -> list[CellResult]:
        return [c for c in self.cells if c.status == "failed"]

    @property
    def metrics(self) -> list[str]:
        if self.table_options.metrics:
            return self.table_options.metrics
        found: list[str | None] = [_headline(c.result) for c in self.cells if c.result is not None]
        return list(dict.fromkeys(m for m in found if m is not None))

    def _subsets(self, dataset: str) -> list[str]:
        if not self.table_options.by_subset:
            return [""]
        names: dict[str, None] = {}
        for c in self.cells:
            if c.dataset == dataset and c.result is not None:
                names.update(dict.fromkeys(s for s in c.result.by_subset if s))
        return ["", *names]

    def _columns(self) -> list[tuple[str, str, str]]:
        return [(d, s, m) for d in self.dataset_labels for s in self._subsets(d) for m in self.metrics]

    def _format(self, value: Any) -> str:
        if value is None:
            return NOT_APPLICABLE
        if isinstance(value, float):
            scaled: float = value * 100 if self.table_options.percent and -1 <= value <= 1 else value
            return f"{scaled:.{self.table_options.digits}f}"
        return str(value)

    def _value(self, cell: CellResult, subset: str, metric: str) -> str:
        if cell.status == "failed":
            return "failed"
        if cell.result is None:
            return NOT_APPLICABLE
        values: dict[str, Any] = cell.result.by_subset.get(subset, {}) if subset else cell.result.metrics
        return self._format(_lookup(values, metric))

    def table(self) -> str:
        """The results as a Markdown table: one row per model, one column per dataset (× subset × metric)."""
        columns = self._columns()
        several_metrics: bool = len(self.metrics) > 1
        headers: list[str] = ["Model"] + [" / ".join(p for p in (d, (s or "total") if self.table_options.by_subset
                                                               else "", m if several_metrics else "") if p)
                                          for d, s, m in columns]
        by_key: dict[tuple[str, str], CellResult] = {(c.dataset, c.model): c for c in self.cells}
        rows: list[list[str]] = [[model] + [self._value(by_key[(d, model)], s, m) for d, s, m in columns]
                                 for model in self.model_labels]
        lines: list[str] = ["| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |",
                            "|:--" + "|--:" * len(columns) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        if not several_metrics and self.metrics:
            lines.append(f"\nMetric: {self.metrics[0]}" + (" (%)" if self.table_options.percent else ""))
        for cell in self.failed:
            lines.append(f"\nFailed: {cell.dataset} × {cell.model}: {cell.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        cells: list[dict[str, Any]] = []
        for c in self.cells:
            entry: dict[str, Any] = {"dataset": c.dataset, "model": c.model, "status": c.status}
            if c.reason:
                entry["reason"] = c.reason
            if c.result is not None:
                entry["metrics"] = {m: _lookup(c.result.metrics, m) for m in self.metrics}
                if c.result.by_subset:
                    entry["by_subset"] = {s: {m: _lookup(v, m) for m in self.metrics}
                                          for s, v in c.result.by_subset.items()}
                if c.result.path is not None:
                    entry["result_file"] = c.result.path.name
            cells.append(entry)
        return {"name": self.name, "timestamp": self.timestamp, "seflag_version": __version__,
                "metrics": self.metrics, "cells": cells, "config": self.source}

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "matrix.json").write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
        (directory / "table.md").write_text(self.table() + "\n", encoding="utf-8")
        self.path = directory
        return directory


def run_matrix(matrix: MatrixConfig, fail_fast: bool = False) -> MatrixResult:
    """Runs all applicable cells of a matrix one after the other and collects their results."""
    result = MatrixResult(name=matrix.name, cells=[], table_options=matrix.spec.table,
                          dataset_labels=matrix.dataset_labels, model_labels=matrix.model_labels,
                          source=matrix.source)
    directory: Path | None = None
    if matrix.output_dir is not None:
        stamp: str = result.timestamp.replace(":", "").replace("+0000", "Z")
        directory = matrix.output_dir / re.sub(r"[^\w.-]+", "_", f"{matrix.name}_{stamp}")
    runnable: list[Cell] = [c for c in matrix.cells if c.config is not None]
    for cell in matrix.cells:
        if cell.config is None:
            result.cells.append(CellResult(cell.dataset, cell.model, "not_applicable", cell.reason))
            continue
        logger.info("[%d/%d] %s × %s", runnable.index(cell) + 1, len(runnable), cell.dataset, cell.model)
        config: EvaluationConfig = cell.config.model_copy(update={"output": cell.config.output.model_copy(
            update={"dir": directory / "cells" if directory is not None else None})})
        try:
            result.cells.append(CellResult(cell.dataset, cell.model, "ok", result=run(config)))
        except SeflagError as e:
            if fail_fast:
                raise
            logger.error("%s × %s failed: %s", cell.dataset, cell.model, e)
            result.cells.append(CellResult(cell.dataset, cell.model, "failed", str(e).splitlines()[0]))
    if directory is not None:
        result.save(directory)
        logger.info("Matrix results written to %s", directory)
    return result


def table_from_results(results: list[EvaluationResult], table: TableOptions | None = None,
                       name: str = "results") -> MatrixResult:
    """Arranges saved evaluation results like a matrix (rows: models, columns: datasets), e.g. for a README table.

    Labels come from each result's configuration (dataset repository/path, model name). If several results have the
    same dataset and model, the most recent one is used, with a warning if their settings (task, options) differ.
    """
    from seflag.components import section_label

    latest: dict[tuple[str, str], EvaluationResult] = {}
    for result in sorted(results, key=lambda r: str(r.provenance.get("timestamp", ""))):
        dataset: str = section_label("dataset", result.config.get("dataset")) or "dataset"
        language: str = result.language
        model: str = section_label("model", result.config.get("model")) or result.name
        key = (f"{dataset} ({language})" if language else dataset, model)
        if key in latest:
            older: EvaluationResult = latest[key]
            differing: list[str] = [s for s in ("task", "options", "dataset", "model")
                                    if older.config.get(s) != result.config.get(s)]
            if differing:  # e.g. another label_normalization: the table shows only the newer settings
                logger.warning("%s × %s: %s replaces the older %s, which has different settings (%s)", *key,
                               result.path, older.path, ", ".join(differing))
            else:
                logger.info("%s × %s: using %s instead of the older %s", *key, result.path, older.path)
        latest[key] = result
    dataset_labels: list[str] = sorted({d for d, _ in latest})
    model_labels: list[str] = sorted({m for _, m in latest})
    cells: list[CellResult] = []
    for d in dataset_labels:
        for m in model_labels:
            cells.append(CellResult(d, m, "ok", result=latest[(d, m)]) if (d, m) in latest
                         else CellResult(d, m, "not_applicable", "no result"))
    return MatrixResult(name=name, cells=cells, table_options=table or TableOptions(), dataset_labels=dataset_labels,
                        model_labels=model_labels, source={"results": [str(r.path) for r in latest.values()]})
