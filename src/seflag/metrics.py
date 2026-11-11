"""Metrics for token-level evaluation. Pure Python, so results do not depend on optional libraries."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Hashable, Sequence


def _check(gold: Sequence, pred: Sequence) -> None:
    if len(gold) != len(pred):
        raise ValueError(f"Gold and predicted sequences differ in length ({len(gold)} vs. {len(pred)}).")


def accuracy(gold: Sequence[Hashable], pred: Sequence[Hashable]) -> float | None:
    """Share of positions where prediction and gold label are identical; None for empty input."""
    _check(gold, pred)
    if not gold:
        return None
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_report(gold: Sequence[Hashable], pred: Sequence[Hashable]) -> dict[str, Any]:
    """Per-label precision/recall/F1 plus macro and weighted averages.

    Labels are the union of gold and predicted labels; macro averages include labels that were only
    predicted (with recall 0 and support 0), which is the scikit-learn convention.
    """
    _check(gold, pred)
    tp: Counter = Counter(g for g, p in zip(gold, pred) if g == p)
    gold_counts: Counter = Counter(gold)
    pred_counts: Counter = Counter(pred)
    labels: list = sorted(set(gold_counts) | set(pred_counts), key=str)
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        precision: float = _safe_div(tp[label], pred_counts[label])
        recall: float = _safe_div(tp[label], gold_counts[label])
        per_label[str(label)] = {
            "precision": precision, "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall), "support": gold_counts[label],
        }
    total: int = len(gold)
    report: dict[str, Any] = {"per_label": per_label}
    for metric in ("precision", "recall", "f1"):
        values: list[float] = [v[metric] for v in per_label.values()]
        report[f"macro_{metric}"] = _safe_div(sum(values), len(values))
        report[f"weighted_{metric}"] = _safe_div(sum(v[metric] * v["support"] for v in per_label.values()), total)
    return report


def confusion_matrix(gold: Sequence[Hashable], pred: Sequence[Hashable]) -> dict[str, dict[str, int]]:
    """Nested counts ``matrix[gold_label][predicted_label]``."""
    _check(gold, pred)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold, pred):
        matrix[str(g)][str(p)] += 1
    return {g: dict(row) for g, row in sorted(matrix.items())}


def top_errors(gold: Sequence[Hashable], pred: Sequence[Hashable], n: int) -> list[dict[str, Any]]:
    """The n most frequent (gold, predicted) confusions."""
    _check(gold, pred)
    errors: Counter = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [{"gold": g, "predicted": p, "count": c} for (g, p), c in errors.most_common(n)]


def _ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; ties get the average of their ranks (as in scipy.stats.rankdata)."""
    order: list[int] = sorted(range(len(values)), key=lambda i: values[i])
    ranks: list[float] = [0.0] * len(values)
    i: int = 0
    while i < len(order):
        j: int = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson's correlation coefficient; None for fewer than two points or constant input."""
    _check(x, y)
    n: int = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rank correlation (Pearson on average ranks, so ties are handled as in scipy)."""
    _check(x, y)
    return pearson(_ranks(x), _ranks(y))
