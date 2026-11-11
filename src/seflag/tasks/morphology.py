from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Doc, Token

from seflag.errors import ConfigError
from seflag.metrics import top_errors
from seflag.tasks.base import Task, TaskResult

# the features scored by the official CoNLL 2018 shared task evaluation ("UFeats");
# layered features such as Number[psor] and treebank-specific ones such as InflClass are not included
UNIVERSAL_FEATURES: frozenset[str] = frozenset({
    "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Gender", "Animacy", "Number", "Case",
    "Definite", "Degree", "VerbForm", "Mood", "Tense", "Aspect", "Voice", "Evident", "Polarity", "Person", "Polite",
})
NONE: str = "_"  # display value for "no features" / "feature absent"

FeatureSet = dict[str, str]


def _prf(tp: int, n_gold: int, n_pred: int) -> dict[str, float]:
    precision: float = tp / n_pred if n_pred else 0.0
    recall: float = tp / n_gold if n_gold else 0.0
    f1: float = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _as_string(features: FeatureSet) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(features.items())) or NONE


class _Counts:
    """Accumulates token-level morphology statistics for one group of tokens."""

    def __init__(self):
        self.tokens = 0
        self.exact = 0
        self.exact_with_upos = 0
        self.pairs = Counter()  # tp / gold / pred over feature=value pairs
        self.per_feature: dict[str, Counter] = defaultdict(Counter)

    def add(self, gold: FeatureSet, pred: FeatureSet, upos_correct: bool) -> None:
        self.tokens += 1
        exact: bool = gold == pred
        self.exact += exact
        self.exact_with_upos += exact and upos_correct
        for name in gold.keys() | pred.keys():
            g, p = gold.get(name), pred.get(name)
            counts: Counter = self.per_feature[name]
            counts["gold"] += g is not None
            counts["pred"] += p is not None
            counts["tp"] += g is not None and g == p
        self.pairs["gold"] += len(gold)
        self.pairs["pred"] += len(pred)
        self.pairs["tp"] += sum(1 for k, v in gold.items() if pred.get(k) == v)

    def summary(self) -> dict[str, Any]:
        pair: dict[str, float] = _prf(self.pairs["tp"], self.pairs["gold"], self.pairs["pred"])
        return {"accuracy": self.exact / self.tokens if self.tokens else None,
                **{f"pair_{k}": v for k, v in pair.items()}, "scored": self.tokens}


class MorphologyTask(Task):
    """Morphological feature tagging (UD FEATS).

    Reports exact-match accuracy of the whole (filtered) feature set per word, which with the default
    ``features: universal`` equals the CoNLL 2018 UFeats score on gold tokenization; micro precision/recall/F1
    over feature=value pairs; and precision/recall/F1 per feature. Words without gold features are scored
    with the empty feature set, as in CoNLL-U ``_``.
    """

    description = "Morphological features (UD FEATS): full-set accuracy (UFeats), feature=value and per-feature P/R/F1."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        features: Literal["universal", "all", "model"] | list[str] = Field(
            "universal",
            description="Features to evaluate: 'universal' (CoNLL 2018 UFeats set), 'all', 'model' (universal "
                        "features the model can predict, from its label inventory), or an explicit list.")
        exclude_features: list[str] = Field(default_factory=list, description="Features never evaluated.")
        predicted_feature_names: dict[str, str] = Field(
            default_factory=dict,
            description="Renames features in the predictions to the gold scheme, e.g. {Verbform: VerbForm}.")
        predicted_feature_values: dict[str, dict[str, str]] = Field(
            default_factory=dict,
            description="Maps predicted values per (gold-scheme) feature, e.g. {Tense: {Past: Perf}}.")
        with_upos: bool = Field(False, description="Also report accuracy of UPOS + features jointly.")

    @property
    def required_layers(self) -> set[str]:
        return {"morph", "upos"} if self.config.with_upos else {"morph"}

    def __init__(self, config: Any, options):
        super().__init__(config, options)
        self.model_features: set[str] | None = None

    def bind_backend(self, backend: Any) -> None:
        if self.config.features != "model":
            return
        inventory: set[str] | None = backend.label_inventory("morph")
        if inventory is None:
            raise ConfigError("'features: model' needs a backend that reports its feature inventory; "
                              "list the features explicitly instead.")
        renamed: set[str] = {self.config.predicted_feature_names.get(n, n) for n in inventory}
        self.model_features = renamed & UNIVERSAL_FEATURES

    def _selected(self, name: str) -> bool:
        if name in self.config.exclude_features:
            return False
        if self.config.features == "universal":
            return name in UNIVERSAL_FEATURES
        if self.config.features == "all":
            return True
        if self.config.features == "model":
            return name in (self.model_features or set())
        return name in self.config.features

    def features_of(self, token: Token, predicted: bool) -> FeatureSet:
        result: FeatureSet = {}
        for name, value in token.morph.to_dict().items():
            if predicted:
                name = self.config.predicted_feature_names.get(name, name)
                value = self.config.predicted_feature_values.get(name, {}).get(value, value)
            if self._selected(name):
                result[name] = value
        return result

    def evaluate(self, gold: list[Doc], predicted: list[Doc]) -> TaskResult:
        total = _Counts()
        by_subset: dict[str, _Counts] = defaultdict(_Counts)
        full_gold: list[str] = []
        full_pred: list[str] = []
        feature_errors: Counter = Counter()
        for gold_doc, pred_doc in zip(gold, predicted, strict=True):
            subset: str = gold_doc.user_data.get("subset") or ""
            for gold_token, pred_token in zip(gold_doc, pred_doc, strict=True):
                g: FeatureSet = self.features_of(gold_token, predicted=False)
                p: FeatureSet = self.features_of(pred_token, predicted=True)
                upos_correct: bool = self.config.with_upos and gold_token.pos_ == pred_token.pos_
                total.add(g, p, upos_correct)
                by_subset[subset].add(g, p, upos_correct)
                full_gold.append(_as_string(g))
                full_pred.append(_as_string(p))
                for name in g.keys() | p.keys():
                    if g.get(name) != p.get(name):
                        feature_errors[(name, g.get(name, NONE), p.get(name, NONE))] += 1

        per_feature: dict[str, dict[str, Any]] = {
            name: {**_prf(c["tp"], c["gold"], c["pred"]), "support": c["gold"], "predicted": c["pred"]}
            for name, c in sorted(total.per_feature.items())
        }
        metrics: dict[str, Any] = {**total.summary(), "tokens": total.tokens}
        if self.config.with_upos:
            metrics["upos_feats_accuracy"] = total.exact_with_upos / total.tokens if total.tokens else None
        metrics["macro_f1"] = (sum(v["f1"] for v in per_feature.values()) / len(per_feature)) if per_feature else None
        metrics["per_feature"] = per_feature

        n: int = self.options.max_errors_reported
        details: dict[str, Any] = {
            "top_errors": top_errors(full_gold, full_pred, n),
            "top_feature_errors": [{"feature": f, "gold": g, "predicted": p, "count": c}
                                   for (f, g, p), c in feature_errors.most_common(n)],
            "features_never_predicted": sorted(k for k, v in per_feature.items() if v["support"] and not v["predicted"]),
            "features_not_in_gold": sorted(k for k, v in per_feature.items() if v["predicted"] and not v["support"]),
        }
        if self.model_features is not None:
            details["evaluated_features"] = sorted(self.model_features - set(self.config.exclude_features))
        subsets: dict[str, dict[str, Any]] = (
            {s: c.summary() for s, c in sorted(by_subset.items())} if len(by_subset) > 1 else {})
        return TaskResult(metrics=metrics, by_subset=subsets, details=details)
