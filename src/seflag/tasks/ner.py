"""Named entity recognition: entity-level precision/recall/F1 with exact matching (CoNLL / seqeval semantics)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Doc

from seflag.metrics import classification_report
from seflag.tasks.base import Task, TaskResult

OUTSIDE: str = "O"
Entity = tuple[int, int, str]  # start, end (exclusive, token indices), label


def _prf(correct: int, gold: int, predicted: int) -> dict[str, float]:
    precision: float = correct / predicted if predicted else 0.0
    recall: float = correct / gold if gold else 0.0
    f1: float = 2 * correct / (gold + predicted) if gold + predicted else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class _Counts:
    def __init__(self):
        self.c = Counter()

    def add(self, gold: set[Entity], pred: set[Entity]) -> None:
        self.c["gold"] += len(gold)
        self.c["predicted"] += len(pred)
        self.c["correct"] += len(gold & pred)
        self.c["span_correct"] += len({e[:2] for e in gold} & {e[:2] for e in pred})

    def summary(self) -> dict[str, Any]:
        return {**_prf(self.c["correct"], self.c["gold"], self.c["predicted"]), "scored": self.c["gold"]}


class NerTask(Task):
    """Entity-level evaluation: an entity counts as correct only if both its boundaries and its type match.

    Gold and predicted labels usually come from different schemes (e.g. Herodotos PRS/GEO/GRP vs. LatinCy
    PERSON/LOC/NORP). ``gold_labels`` and ``predicted_labels`` map both onto a common scheme; mapping a label to
    ``null`` removes those entities. Also reported: span-only (unlabelled) scores, per-label scores, and
    token-level scores (each token labelled with its entity type or O), the latter for comparison with older
    evaluations.
    """

    description = "Named entity recognition: entity-level P/R/F1 (exact match), per label, unlabelled, token level."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        gold_labels: dict[str, str | None] = Field(
            default_factory=dict, description="Maps gold labels to the common scheme; null drops the label.")
        predicted_labels: dict[str, str | None] = Field(
            default_factory=dict, description="Maps predicted labels to the common scheme; null drops the label.")
        labels: list[str] | None = Field(
            None, description="Evaluate only these (mapped) labels; entities with other labels are ignored.")

    def __init__(self, config: Any, options):
        super().__init__(config, options)
        self.model_labels: set[str] | None = None

    @property
    def required_layers(self) -> set[str]:
        return {"entities"}

    def bind_backend(self, backend: Any) -> None:
        self.model_labels = backend.label_inventory("entities")

    def _entities(self, doc: Doc, mapping: dict[str, str | None], raw: Counter) -> set[Entity]:
        result: set[Entity] = set()
        for ent in doc.ents:
            raw[ent.label_] += 1
            label: str | None = mapping.get(ent.label_, ent.label_)
            if label is None or (self.config.labels is not None and label not in self.config.labels):
                continue
            result.add((ent.start, ent.end, label))
        return result

    @staticmethod
    def _token_labels(doc: Doc, entities: set[Entity]) -> list[str]:
        labels: list[str] = [OUTSIDE] * len(doc)
        for start, end, label in entities:
            for i in range(start, end):
                labels[i] = label
        return labels

    def evaluate(self, gold: list[Doc], predicted: list[Doc]) -> TaskResult:
        total = _Counts()
        by_subset: dict[str, _Counts] = defaultdict(_Counts)
        per_label: dict[str, Counter] = defaultdict(Counter)
        raw_gold: Counter = Counter()
        raw_pred: Counter = Counter()
        errors: Counter = Counter()
        token_gold: list[str] = []
        token_pred: list[str] = []
        skipped: int = 0

        for gold_doc, pred_doc in zip(gold, predicted, strict=True):
            if not gold_doc.user_data.get("entities_annotated", True):
                skipped += 1
                continue
            g: set[Entity] = self._entities(gold_doc, self.config.gold_labels, raw_gold)
            p: set[Entity] = self._entities(pred_doc, self.config.predicted_labels, raw_pred)
            total.add(g, p)
            by_subset[gold_doc.user_data.get("subset") or ""].add(g, p)
            for e in g:
                per_label[e[2]]["gold"] += 1
                per_label[e[2]]["correct"] += e in p
            for e in p:
                per_label[e[2]]["predicted"] += 1
            # classify errors
            pred_spans: dict[tuple[int, int], str] = {e[:2]: e[2] for e in p}
            for start, end, label in g - p:
                if (start, end) in pred_spans:
                    errors[("wrong label", label, pred_spans[(start, end)])] += 1
                elif any(s < end and start < e for s, e, _ in p):
                    errors[("wrong boundaries", label, "")] += 1
                else:
                    errors[("missed", label, "")] += 1
            gold_spans: set[tuple[int, int]] = {e[:2] for e in g}
            for start, end, label in p - g:
                if (start, end) not in gold_spans and not any(s < end and start < e for s, e, _ in g):
                    errors[("spurious", "", label)] += 1
            token_gold += self._token_labels(gold_doc, g)
            token_pred += self._token_labels(pred_doc, p)

        c = total.c
        metrics: dict[str, Any] = {
            **_prf(c["correct"], c["gold"], c["predicted"]),
            "unlabeled_f1": _prf(c["span_correct"], c["gold"], c["predicted"])["f1"],
            "gold_entities": c["gold"], "predicted_entities": c["predicted"], "correct_entities": c["correct"],
        }
        labels_report: dict[str, dict[str, Any]] = {
            label: {**_prf(v["correct"], v["gold"], v["predicted"]), "support": v["gold"], "predicted": v["predicted"]}
            for label, v in sorted(per_label.items())
        }
        metrics["macro_f1"] = (sum(v["f1"] for v in labels_report.values()) / len(labels_report)
                               if labels_report else None)
        metrics["per_label"] = labels_report
        metrics["unlabeled"] = _prf(c["span_correct"], c["gold"], c["predicted"])
        token_report: dict[str, Any] = classification_report(token_gold, token_pred)
        token_report["accuracy"] = (sum(a == b for a, b in zip(token_gold, token_pred)) / len(token_gold)
                                    if token_gold else None)
        metrics["token_level"] = token_report
        if skipped:
            metrics["sentences_without_gold_entities_annotation"] = skipped

        error_types: Counter = Counter()
        for (kind, _, _), count in errors.items():
            error_types[kind] += count
        n: int = self.options.max_errors_reported
        details: dict[str, Any] = {
            "top_errors": [{"type": t, "gold": g, "predicted": p, "count": k}
                           for (t, g, p), k in errors.most_common(n)],
            "error_types": dict(error_types.most_common()),
            "gold_labels_raw": dict(raw_gold.most_common()),
            "predicted_labels_raw": dict(raw_pred.most_common()),
        }
        if self.model_labels is not None:
            details["model_labels"] = sorted(self.model_labels)
        subsets: dict[str, dict[str, Any]] = (
            {s: v.summary() for s, v in sorted(by_subset.items())} if len(by_subset) > 1 else {})
        return TaskResult(metrics=metrics, by_subset=subsets, details=details)
