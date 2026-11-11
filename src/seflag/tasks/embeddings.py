"""Type-level evaluation of word embeddings: graded similarity, neighbour retrieval, synonym selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from seflag.data.lexical import ChoiceItem, WordPair
from seflag.metrics import pearson, spearman
from seflag.normalization import get_pipeline
from seflag.tasks.base import LexicalTask, TaskResult

OovPolicy = Literal["skip", "subwords"]
OOV_DESCRIPTION: str = ("Words outside the model's vocabulary: 'skip' the item (coverage is reported), or build "
                        "vectors from 'subwords' (fastText models only).")


class _Words:
    """Applies the input normalization and tracks vocabulary coverage."""

    def __init__(self, model: Any, options, oov: str):
        self.model = model
        self.normalize = get_pipeline(options.input_normalization)
        self.subwords: bool = oov == "subwords"
        if self.subwords and not model.supports_subwords:
            from seflag.errors import ConfigError

            raise ConfigError("'oov: subwords' needs a model with subword information (e.g. fastText .bin).")
        self.missing: set[str] = set()
        self.seen: set[str] = set()

    def known(self, word: str) -> bool:
        word = self.normalize(word)
        self.seen.add(word)
        if self.model.in_vocabulary(word) or self.subwords:
            return True
        self.missing.add(word)
        return False

    def stats(self) -> dict[str, Any]:
        return {"unique_words": len(self.seen), "oov_words": len(self.missing),
                "oov_ratio": len(self.missing) / len(self.seen) if self.seen else None}


class WordSimilarityTask(LexicalTask):
    """Correlation between gold similarity/relatedness scores and the cosine similarity of word vectors."""

    description = "Word similarity/relatedness: Spearman and Pearson correlation with gold scores."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        oov: OovPolicy = Field("skip", description=OOV_DESCRIPTION)

    def evaluate_lexical(self, items: list[WordPair], model: Any) -> TaskResult:
        if any(p.score is None for p in items):
            from seflag.errors import DataError

            raise DataError("word_similarity needs gold scores for all pairs (set 'score_field' / a score column).")
        words = _Words(model, self.options, self.config.oov)
        gold, cosine, kept, groups = [], [], [], defaultdict(lambda: ([], []))
        for pair in items:
            if not (words.known(pair.word1) and words.known(pair.word2)):
                continue
            sim: float = model.similarity(words.normalize(pair.word1), words.normalize(pair.word2), words.subwords)
            gold.append(pair.score)
            cosine.append(sim)
            kept.append(pair)
            groups[pair.subset or ""][0].append(pair.score)
            groups[pair.subset or ""][1].append(sim)
        metrics: dict[str, Any] = {
            "spearman": spearman(gold, cosine), "pearson": pearson(gold, cosine),
            "pairs": len(items), "evaluated_pairs": len(kept),
            "coverage": len(kept) / len(items) if items else None, **words.stats(),
        }
        # pairs where model and annotators disagree most (difference of rank percentiles)
        details: dict[str, Any] = {"oov_examples": sorted(words.missing)[:50]}
        if len(kept) > 1:
            from seflag.metrics import _ranks

            rg, rc = _ranks(gold), _ranks(cosine)
            n: int = len(kept)
            order = sorted(range(n), key=lambda i: -abs(rg[i] - rc[i]))[:self.options.max_errors_reported]
            details["largest_disagreements"] = [
                {"word1": kept[i].word1, "word2": kept[i].word2, "gold": gold[i], "cosine": round(cosine[i], 4),
                 "gold_percentile": round(rg[i] / n, 3), "model_percentile": round(rc[i] / n, 3)} for i in order]
        by_subset = ({s: {"spearman": spearman(g, c), "pearson": pearson(g, c), "scored": len(g)}
                      for s, (g, c) in sorted(groups.items())} if len(groups) > 1 else {})
        return TaskResult(metrics=metrics, by_subset=by_subset, details=details)


class WordRetrievalTask(LexicalTask):
    """Is the second word of a related pair among the k nearest neighbours of the first? (hits@k, MRR)

    The search covers the model's vocabulary (limited by the backend's ``max_vocabulary``, if set).
    """

    description = "Neighbour retrieval for related word pairs: hits@k and mean reciprocal rank."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        k: list[int] = Field(default_factory=lambda: [1, 10], description="Neighbourhood sizes to report.")
        directions: Literal["forward", "both"] = Field(
            "forward", description="'forward': word1 -> word2; 'both': also word2 -> word1 (each counted).")
        oov: OovPolicy = Field("skip", description=OOV_DESCRIPTION + " The retrieved word must be in the vocabulary.")

    def evaluate_lexical(self, items: list[WordPair], model: Any) -> TaskResult:
        words = _Words(model, self.options, self.config.oov)
        k_max: int = max(self.config.k)
        queries: list[tuple[str, str]] = []
        for pair in items:
            directions = [(pair.word1, pair.word2)]
            if self.config.directions == "both":
                directions.append((pair.word2, pair.word1))
            for query, answer in directions:
                query_known: bool = words.known(query)
                # the answer must be retrievable, i.e. in the vocabulary (subwords do not help here)
                answer_key: str | None = model.key(words.normalize(answer))
                words.seen.add(words.normalize(answer))
                if answer_key is None:
                    words.missing.add(words.normalize(answer))
                if query_known and answer_key is not None:
                    queries.append((words.normalize(query), answer_key))
        total: int = len(items) * (2 if self.config.directions == "both" else 1)
        hits = {k: 0 for k in self.config.k}
        reciprocal: float = 0.0
        examples: list[dict[str, Any]] = []
        for query, answer in queries:
            neighbours: list[str] = model.neighbours(query, k_max, words.subwords)
            rank: int | None = neighbours.index(answer) + 1 if answer in neighbours else None
            for k in self.config.k:
                hits[k] += rank is not None and rank <= k
            reciprocal += 1 / rank if rank else 0.0
            if len(examples) < 20:
                examples.append({"query": query, "expected": answer, "rank": rank, "neighbours": neighbours[:5]})
        n: int = len(queries)
        metrics: dict[str, Any] = {f"hits_at_{k}": hits[k] / n if n else None for k in self.config.k}
        metrics.update({f"mrr_at_{k_max}": reciprocal / n if n else None, "queries": total, "evaluated_queries": n,
                        "coverage": n / total if total else None, **words.stats()})
        return TaskResult(metrics=metrics, details={"examples": examples, "oov_examples": sorted(words.missing)[:50]})


class SynonymSelectionTask(LexicalTask):
    """TOEFL-style multiple choice: the correct candidate must be strictly the most similar one to the target."""

    description = "Synonym selection (multiple choice): accuracy of picking the most similar candidate."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        oov: OovPolicy = Field("skip", description=OOV_DESCRIPTION)
        missing_candidates: Literal["skip_item", "ignore"] = Field(
            "skip_item", description="If a wrong candidate is out of vocabulary: 'skip_item' (strict), or 'ignore' "
                                     "that candidate, which makes the item easier (behaviour of the original script).")

    def evaluate_lexical(self, items: list[ChoiceItem], model: Any) -> TaskResult:
        words = _Words(model, self.options, self.config.oov)
        correct: int = 0
        evaluated: int = 0
        chance: float = 0.0
        errors: list[dict[str, Any]] = []
        for item in items:
            answer_word: str = item.choices[item.answer]
            known = [words.known(c) for c in item.choices]
            if not (words.known(item.target) and known[item.answer]):
                continue
            if not all(known) and self.config.missing_candidates == "skip_item":
                continue
            candidates = [c for c, ok in zip(item.choices, known) if ok]
            target = words.normalize(item.target)
            sims = {c: model.similarity(target, words.normalize(c), words.subwords) for c in candidates}
            best: float = max(sims.values())
            winners = [c for c, s in sims.items() if s == best]
            evaluated += 1
            chance += 1 / len(candidates)
            if winners == [answer_word]:  # ties count as wrong
                correct += 1
            elif len(errors) < self.options.max_errors_reported:
                errors.append({"target": item.target, "expected": answer_word, "chosen": winners[0],
                               "similarities": {c: round(s, 4) for c, s in sims.items()}})
        metrics: dict[str, Any] = {
            "accuracy": correct / evaluated if evaluated else None,
            "chance_accuracy": chance / evaluated if evaluated else None,
            "items": len(items), "evaluated_items": evaluated,
            "coverage": evaluated / len(items) if items else None, **words.stats(),
        }
        return TaskResult(metrics=metrics, details={"errors": errors, "oov_examples": sorted(words.missing)[:50]})
