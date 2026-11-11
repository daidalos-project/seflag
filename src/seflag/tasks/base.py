from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from spacy.tokens import Doc, Token

from seflag.config import EvaluationOptions
from seflag.metrics import accuracy, top_errors
from seflag.normalization import get_pipeline


@dataclass
class TaskResult:
    metrics: dict[str, Any]
    by_subset: dict[str, dict[str, Any]] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Compares predicted ``Doc`` objects with gold ``Doc`` objects for one kind of annotation."""

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

    description: ClassVar[str] = ""

    def __init__(self, config: Any, options: EvaluationOptions):
        self.config = config
        self.options: EvaluationOptions = options

    @property
    @abstractmethod
    def required_layers(self) -> set[str]:
        """Gold annotation layers (see seflag.data.records.LAYERS) needed by this task."""

    def bind_backend(self, backend: Any) -> None:
        """Called after prediction and before evaluation; lets a task query the backend (e.g. label inventories)."""

    @abstractmethod
    def evaluate(self, gold: list[Doc], predicted: list[Doc]) -> TaskResult:
        ...


class LexicalTask(Task):
    """Base for type-level tasks that evaluate word embeddings on lexical datasets instead of sentences."""

    @property
    def required_layers(self) -> set[str]:
        return set()

    def evaluate(self, gold: list[Doc], predicted: list[Doc]) -> TaskResult:
        raise TypeError("Lexical tasks are evaluated with evaluate_lexical().")

    @abstractmethod
    def evaluate_lexical(self, items: list, model: Any) -> TaskResult:
        """Evaluates an embedding backend on the items of a lexical dataset."""


EMPTY_PREDICTION: str = "<empty>"


class TokenLabelTask(Task):
    """Base for tasks that assign exactly one string label to every token (lemmas, POS tags, ...).

    Tokens with an empty gold label (missing annotation) are not scored. Empty predictions are scored
    as wrong and reported with the label ``<empty>``.
    """

    normalize_labels: ClassVar[bool] = True

    @abstractmethod
    def label(self, token: Token) -> str:
        """Extracts the label to evaluate from a token (used for gold and predicted tokens)."""

    def include(self, gold_token: Token) -> bool:
        """Whether a gold token is scored at all (beyond having a gold label)."""
        return True

    def extra_metrics(self, gold: list[str], pred: list[str]) -> dict[str, Any]:
        return {}

    def collect(self, gold_docs: list[Doc], pred_docs: list[Doc]) -> tuple[list[str], list[str], list[str], dict]:
        normalize = get_pipeline(self.options.label_normalization if self.normalize_labels else [])
        gold_labels: list[str] = []
        pred_labels: list[str] = []
        subsets: list[str] = []
        counts: dict[str, int] = defaultdict(int)
        for gold_doc, pred_doc in zip(gold_docs, pred_docs, strict=True):
            for gold_token, pred_token in zip(gold_doc, pred_doc, strict=True):
                counts["tokens"] += 1
                g: str = self.label(gold_token)
                if not g:
                    counts["unannotated_in_gold"] += 1
                    continue
                if not self.include(gold_token):
                    counts["excluded"] += 1
                    continue
                gold_labels.append(normalize(g))
                pred_labels.append(normalize(self.label(pred_token)) or EMPTY_PREDICTION)
                subsets.append(gold_doc.user_data.get("subset") or "")
        counts["scored"] = len(gold_labels)
        counts["empty_predictions"] = sum(1 for p in pred_labels if p == EMPTY_PREDICTION)
        return gold_labels, pred_labels, subsets, dict(counts)

    def evaluate(self, gold: list[Doc], predicted: list[Doc]) -> TaskResult:
        gold_labels, pred_labels, subsets, counts = self.collect(gold, predicted)
        metrics: dict[str, Any] = {"accuracy": accuracy(gold_labels, pred_labels), **counts}
        metrics.update(self.extra_metrics(gold_labels, pred_labels))
        by_subset: dict[str, dict[str, Any]] = {}
        if len(set(subsets)) > 1:
            for subset in sorted(set(subsets)):
                idx: list[int] = [i for i, s in enumerate(subsets) if s == subset]
                g: list[str] = [gold_labels[i] for i in idx]
                p: list[str] = [pred_labels[i] for i in idx]
                by_subset[subset] = {"accuracy": accuracy(g, p), "scored": len(idx)}
        details: dict[str, Any] = {"top_errors": top_errors(gold_labels, pred_labels, self.options.max_errors_reported)}
        return TaskResult(metrics=metrics, by_subset=by_subset, details=details)
