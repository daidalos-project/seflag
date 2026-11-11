"""Dependency parsing evaluation following the CoNLL 2018 shared task (UAS, LAS, CLAS, MLAS, BLEX).

With gold tokenization, system and gold words align 1:1, so the official metrics reduce to per-word
comparisons. The definitions follow ``conll18_ud_eval.py`` v1.2 (https://universaldependencies.org/conll18/):

* deprel subtypes are ignored (``obl:arg`` -> ``obl``) and only universal features are compared;
* CLAS, MLAS and BLEX are F1 scores over *content* relations: recall over gold content words, precision over
  words the system labelled with a content relation;
* MLAS additionally requires correct UPOS, universal features, and correct functional children
  (aux, cop, mark, det, clf, case, cc) with their relation, UPOS and features;
* BLEX additionally requires the correct lemma (not compared where the gold lemma is ``_``).

Unlike the official script, invalid predicted trees (several roots, cycles) are scored instead of rejected and
counted in the results; ``require_valid_trees: true`` restores the strict behaviour.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Doc, Token

from seflag.errors import DataError
from seflag.normalization import get_pipeline
from seflag.tasks.base import Task, TaskResult
from seflag.tasks.morphology import UNIVERSAL_FEATURES

CONTENT_DEPRELS: frozenset[str] = frozenset({
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated", "advcl", "advmod",
    "discourse", "nmod", "appos", "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list", "parataxis",
    "orphan", "goeswith", "reparandum", "root", "dep",
})
FUNCTIONAL_DEPRELS: frozenset[str] = frozenset({"aux", "cop", "mark", "det", "clf", "case", "cc"})
ALL_METRICS: tuple[str, ...] = ("uas", "las", "clas", "mlas", "blex")


@dataclass(slots=True)
class Word:
    head: int  # CoNLL-U convention: 1-based, 0 = root
    deprel: str  # as compared (possibly without subtype)
    universal_deprel: str  # without subtype, decides content/functional
    upos: str
    feats: str  # universal features only, sorted
    lemma: str


def _f1(correct: int, gold: int, system: int) -> dict[str, float]:
    precision: float = correct / system if system else 0.0
    recall: float = correct / gold if gold else 0.0
    f1: float = 2 * correct / (gold + system) if gold + system else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def tree_problems(heads: list[int]) -> tuple[int, bool]:
    """Returns the number of roots and whether the heads (1-based, 0 = root) contain a cycle."""
    roots: int = sum(1 for h in heads if h == 0)
    for start in range(len(heads)):
        seen: set[int] = set()
        node: int = start
        while heads[node] != 0:
            if node in seen:
                return roots, True
            seen.add(node)
            node = heads[node] - 1
    return roots, False


class _Counts:
    def __init__(self):
        self.c = Counter()

    def summary(self, metrics: tuple[str, ...]) -> dict[str, Any]:
        c = self.c
        out: dict[str, Any] = {}
        if "uas" in metrics:
            out["uas"] = c["uas"] / c["words"] if c["words"] else None
        if "las" in metrics:
            out["las"] = c["las"] / c["words"] if c["words"] else None
            out["label_accuracy"] = c["la"] / c["words"] if c["words"] else None
        for m in ("clas", "mlas", "blex"):
            if m in metrics:
                out[m] = _f1(c[m], c["gold_content"], c["system_content"])["f1"]
        out["scored"] = c["words"]
        return out


class DependencyParsingTask(Task):
    """Dependency parsing with the CoNLL 2018 metrics on gold tokenization and (by default) gold sentences."""

    description = "Dependency parsing: UAS, LAS, CLAS, MLAS, BLEX (CoNLL 2018 definitions), per-relation scores."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        metrics: list[Literal["uas", "las", "clas", "mlas", "blex"]] = Field(
            default_factory=lambda: list(ALL_METRICS),
            description="Metrics to compute. MLAS needs gold UPOS and features, BLEX gold lemmas.")
        deprel_subtypes: Literal["ignore", "keep"] = Field(
            "ignore", description="'ignore' compares universal relations only (official), 'keep' full labels.")
        exclude_deprels: list[str] = Field(
            default_factory=list,
            description="Words whose gold relation is listed here are not scored, e.g. [punct]. Not official.")
        predicted_deprels: dict[str, str] = Field(
            default_factory=dict, description="Maps predicted relation labels to the gold scheme before comparison.")
        predicted_feature_names: dict[str, str] = Field(
            default_factory=dict, description="Renames predicted features for MLAS, e.g. {Verbform: VerbForm}.")
        require_valid_trees: bool = Field(
            False, description="Raise an error if a predicted sentence has not exactly one root or has a cycle.")

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(m for m in ALL_METRICS if m in self.config.metrics)

    @property
    def required_layers(self) -> set[str]:
        layers: set[str] = {"dependencies"}
        if "mlas" in self.config.metrics:
            layers |= {"upos", "morph"}
        if "blex" in self.config.metrics:
            layers.add("lemma")
        return layers

    # --- word extraction -----------------------------------------------------------------------------

    def _word(self, token: Token, predicted: bool, normalize_lemma) -> Word | None:
        deprel: str = token.dep_
        if not predicted and not deprel:
            return None  # no gold annotation
        if predicted:
            deprel = self.config.predicted_deprels.get(deprel, deprel)
        universal: str = deprel.split(":")[0]
        feats: list[str] = []
        for name, value in token.morph.to_dict().items():
            if predicted:
                name = self.config.predicted_feature_names.get(name, name)
            if name in UNIVERSAL_FEATURES:
                feats.append(f"{name}={value}")
        head: int = 0 if token.head.i == token.i else token.head.i - token.doc[0].i + 1
        return Word(head=head, deprel=universal if self.config.deprel_subtypes == "ignore" else deprel,
                    universal_deprel=universal, upos=token.pos_, feats="|".join(sorted(feats)),
                    lemma=normalize_lemma(token.lemma_))

    @staticmethod
    def _functional_children(words: list[Word | None]) -> list[list[tuple]]:
        children: list[list[tuple]] = [[] for _ in words]
        for i, w in enumerate(words):
            if w is not None and w.head and w.universal_deprel in FUNCTIONAL_DEPRELS:
                children[w.head - 1].append((i + 1, w.deprel, w.upos, w.feats))
        return children

    # --- evaluation ----------------------------------------------------------------------------------

    def evaluate(self, gold: list[Doc], predicted: list[Doc]) -> TaskResult:
        normalize = get_pipeline(self.options.label_normalization)
        total = _Counts()
        by_subset: dict[str, _Counts] = defaultdict(_Counts)
        per_deprel: dict[str, Counter] = defaultdict(Counter)
        errors: Counter = Counter()
        validity: Counter = Counter()

        for gold_doc, pred_doc in zip(gold, predicted, strict=True):
            g_words: list[Word | None] = [self._word(t, False, normalize) for t in gold_doc]
            p_words: list[Word] = [self._word(t, True, normalize) for t in pred_doc]
            sent_id = gold_doc.user_data.get("sent_id")
            roots, cycle = tree_problems([w.head for w in p_words])
            validity["sentences"] += 1
            validity["predicted_multiple_roots"] += roots > 1
            validity["predicted_no_root"] += roots == 0
            validity["predicted_cycles"] += cycle
            if self.config.require_valid_trees and (roots != 1 or cycle):
                raise DataError(f"Sentence {sent_id!r}: predicted tree has {roots} roots"
                                f"{' and a cycle' if cycle else ''}.")
            if all(w is not None for w in g_words):
                g_roots, g_cycle = tree_problems([w.head for w in g_words])
                validity["gold_invalid_trees"] += g_roots != 1 or g_cycle

            g_children = self._functional_children(g_words)
            p_children = self._functional_children(p_words)
            counts = by_subset[gold_doc.user_data.get("subset") or ""]
            for i, (g, p) in enumerate(zip(g_words, p_words, strict=True)):
                if g is None:
                    validity["unannotated_in_gold"] += 1
                    continue
                if g.universal_deprel in self.config.exclude_deprels or g.deprel in self.config.exclude_deprels:
                    validity["excluded"] += 1
                    continue
                head_ok: bool = g.head == p.head
                label_ok: bool = g.deprel == p.deprel
                gold_content: bool = g.universal_deprel in CONTENT_DEPRELS
                lemma_ok: bool = (not g.lemma) or g.lemma == p.lemma  # '_' in gold = not compared
                result: dict[str, int] = {
                    "words": 1, "uas": head_ok, "las": head_ok and label_ok, "la": label_ok,
                    "gold_content": gold_content, "system_content": p.universal_deprel in CONTENT_DEPRELS,
                    "clas": gold_content and head_ok and label_ok,
                    "mlas": gold_content and head_ok and label_ok and g.upos == p.upos and g.feats == p.feats
                            and g_children[i] == p_children[i],
                    "blex": gold_content and head_ok and label_ok and lemma_ok,
                }
                total.c.update(result)
                counts.c.update(result)
                per_deprel[g.deprel]["support"] += 1
                per_deprel[g.deprel]["correct"] += head_ok and label_ok
                per_deprel[g.deprel]["head_correct"] += head_ok
                per_deprel[p.deprel]["predicted"] += 1
                per_deprel[p.deprel]["predicted_correct"] += head_ok and label_ok
                if not (head_ok and label_ok):
                    errors[(g.deprel, p.deprel, head_ok)] += 1

        names: tuple[str, ...] = self.metric_names
        metrics: dict[str, Any] = total.summary(names)
        for m in ("clas", "mlas", "blex"):
            if m in names:
                metrics[f"{m}_detail"] = _f1(total.c[m], total.c["gold_content"], total.c["system_content"])
        metrics.update(validity)
        metrics["per_deprel"] = {
            label: {
                **_f1(c["correct"], c["support"], c["predicted"]),  # labelled attachment per relation
                "uas": c["head_correct"] / c["support"] if c["support"] else None,
                "support": c["support"], "predicted": c["predicted"],
            }
            for label, c in sorted(per_deprel.items())
        }
        n: int = self.options.max_errors_reported
        details: dict[str, Any] = {"top_errors": [
            {"gold": g, "predicted": p, "head_correct": h, "count": c} for (g, p, h), c in errors.most_common(n)]}
        subsets: dict[str, dict[str, Any]] = (
            {s: c.summary(names) for s, c in sorted(by_subset.items())} if len(by_subset) > 1 else {})
        return TaskResult(metrics=metrics, by_subset=subsets, details=details)
