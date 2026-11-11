"""Turns an :class:`EvaluationConfig` into an executed evaluation."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spacy.tokens import Doc
from spacy.vocab import Vocab

from seflag import provenance, registry
from seflag.config import EvaluationConfig, validate_component_params
from seflag.backends.embeddings import EmbeddingBackend
from seflag.data.lexical import LexicalDataset
from seflag.errors import ConfigError, DataError
from seflag.tasks.base import LexicalTask
from seflag.normalization import get_pipeline

logger = logging.getLogger("seflag")


@dataclass
class EvaluationResult:
    name: str
    task: str
    language: str
    metrics: dict[str, Any]
    by_subset: dict[str, dict[str, Any]]
    details: dict[str, Any]
    config: dict[str, Any]
    provenance: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)
    path: Path | None = None  # where the result was saved, if at all

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = asdict(self)
        data.pop("path")
        return data

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.path = path
        return path

    def summary(self) -> str:
        """A short human-readable overview of the headline metrics."""
        lines: list[str] = [f"{self.name}  ({self.task}, {self.language})"]
        for key, value in self.metrics.items():
            if (isinstance(value, (int, float)) and not isinstance(value, bool)) or value is None:
                lines.append(f"  {key:<20} {value:.4f}" if isinstance(value, float) else f"  {key:<20} {value}")
        if self.by_subset:
            lines.append("  by subset:")
            for subset, values in self.by_subset.items():
                scores = "  ".join(f"{k}={v:.4f}" for k, v in values.items() if isinstance(v, float))
                lines.append(f"    {subset:<30} {scores}  n={values.get('scored')}")
        return "\n".join(lines)


def _build(kind: str, spec_name: str, params: dict[str, Any], section: str, *args: Any) -> Any:
    component: type = registry.get(kind, spec_name)
    return component(validate_component_params(component, params, section), *args)


def _backend(config: EvaluationConfig):
    component: type = registry.get("backends", config.model.backend)
    params = validate_component_params(component, config.model.params, "model")  # validate before launching
    if config.model.launcher is None:
        return component(params, config)
    if issubclass(component, EmbeddingBackend):
        raise ConfigError("'launcher' is not supported for embedding backends yet; run them in the current "
                          "environment.")
    from seflag.backends.launched import LaunchedBackend
    from seflag.launcher import python_command

    return LaunchedBackend(config, python_command(config.model.launcher))


def _run_tokens(config: EvaluationConfig, task, dataset, backend, timings: dict[str, float]):
    start: float = time.perf_counter()
    gold: list[Doc] = dataset.load_docs(Vocab(), task.required_layers, config.options.max_sentences)
    if not gold:
        raise DataError("The dataset contains no sentences.")
    timings["load_data_s"] = time.perf_counter() - start
    logger.info("Loaded %d sentences (%d tokens)", len(gold), sum(len(d) for d in gold))

    normalize = get_pipeline(config.options.input_normalization)
    inputs: list[list[str]] = [[normalize(t.text) for t in doc] for doc in gold]
    start = time.perf_counter()
    predicted: list[Doc] = backend.predict(inputs)
    timings["predict_s"] = time.perf_counter() - start
    logger.info("Annotated with %s backend in %.1fs", config.model.backend, timings["predict_s"])

    task.bind_backend(backend)  # after prediction, so launched backends can report their label inventories
    return task.evaluate(gold, predicted)


def _run_lexical(config: EvaluationConfig, task, dataset, backend, timings: dict[str, float]):
    start: float = time.perf_counter()
    items: list = dataset.items()[:config.options.max_sentences]
    if not items:
        raise DataError("The dataset contains no items.")
    _ = backend.metadata()  # loads the model, so that loading and evaluation time are reported separately
    timings["load_data_s"] = time.perf_counter() - start
    logger.info("Loaded %d items and the embedding model", len(items))
    start = time.perf_counter()
    result = task.evaluate_lexical(items, backend)
    timings["evaluate_s"] = time.perf_counter() - start
    return result


def run(config: EvaluationConfig) -> EvaluationResult:
    """Runs one evaluation: load gold data, annotate it with the model, compare, record provenance."""
    timings: dict[str, float] = {}
    task = _build("tasks", config.task.type, config.task.params, "task", config.options)
    dataset = _build("datasets", config.dataset.source, config.dataset.params, "dataset", config)
    backend = _backend(config)

    if isinstance(task, LexicalTask) != isinstance(dataset, LexicalDataset) or \
            isinstance(task, LexicalTask) != isinstance(backend, EmbeddingBackend):
        raise ConfigError(
            f"Task '{config.task.type}', dataset source '{config.dataset.source}' and backend "
            f"'{config.model.backend}' do not fit together: word embedding tasks (word_similarity, word_retrieval, "
            "synonym_selection) need a lexical dataset (word_pairs, word_choices) and an embedding backend "
            "(gensim); all other tasks need sentence datasets and annotation backends.")
    if isinstance(task, LexicalTask):
        task_result = _run_lexical(config, task, dataset, backend, timings)
    else:
        task_result = _run_tokens(config, task, dataset, backend, timings)
    result = EvaluationResult(
        name=config.display_name,
        task=config.task.type,
        language=config.language,
        metrics=task_result.metrics,
        by_subset=task_result.by_subset,
        details=task_result.details,
        config=config.model_dump(mode="json"),
        provenance=provenance.collect(dataset.fingerprint(), backend.metadata(),
                                      extra_packages=[str(config.model.params["name"])] if "name" in config.model.params else []),
        timings=timings,
    )
    if config.output.dir is not None:
        stamp: str = result.provenance["timestamp"].replace(":", "").replace("+0000", "Z")
        file_name: str = re.sub(r"[^\w.-]+", "_", f"{result.name}_{stamp}") + ".json"
        result.save(config.resolve_path(config.output.dir) / file_name)
        logger.info("Results written to %s", result.path)
    return result
