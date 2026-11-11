"""Type-level (lexical) datasets for evaluating word embeddings: scored word pairs and multiple-choice items."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from seflag.config import EvaluationConfig
from seflag.data.files import FileResolver, FileSource
from seflag.errors import ConfigError, DataError


@dataclass(slots=True)
class WordPair:
    word1: str
    word2: str
    score: float | None = None  # gold similarity/relatedness; None for pair lists without scores
    subset: str | None = None


@dataclass(slots=True)
class ChoiceItem:
    """A multiple-choice item: which of ``choices`` is most similar to ``target``? ``answer`` indexes ``choices``."""

    target: str
    choices: list[str]
    answer: int = 0
    subset: str | None = None


class LexicalDataset(ABC):
    """Base class for type-level datasets. Subclasses define ``Config`` and implement :meth:`items`."""

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

    description: ClassVar[str] = ""

    def __init__(self, config: Any, evaluation: EvaluationConfig):
        self.config = config
        self.evaluation = evaluation
        self.resolver = FileResolver(config, evaluation)

    @abstractmethod
    def items(self) -> list:
        ...

    def fingerprint(self) -> dict[str, Any]:
        return self.resolver.fingerprint()


def _tabular_rows(path: Path, delimiter: str) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f, delimiter=delimiter) if any(cell.strip() for cell in row)]


class WordPairsDataset(LexicalDataset):
    """Word pairs with optional gold scores, from JSON or delimited text files.

    Example (AGREE, entries like ``{"pair": "κακός, ἀγαθός", "score": 81, ...}``)::

        dataset:
          source: word_pairs
          url: https://zenodo.org/api/records/8027490/files/2_agree_task2.json/content
          checksum: md5:3f037ae1ad530f9098eda1e64e19243a
          format: json
          records_key: pairs
          pair_field: pair
          score_field: score
    """

    description = "Word pairs (with optional similarity scores) from JSON, TSV or CSV files."

    class Config(FileSource):
        format: Literal["json", "tsv", "csv"] = Field(description="File format.")
        # JSON
        records_key: str | None = Field(None, description="json: key of the list of entries (null: top-level list).")
        pair_field: str | None = Field(None, description="json: field holding both words, e.g. 'κακός, ἀγαθός'.")
        pair_separator: str = Field(",", description="json: separator between the two words in 'pair_field'.")
        word_fields: list[str] | None = Field(None, description="json: two fields holding the words separately.")
        score_field: str | None = Field(None, description="json: field with the gold score (omit for unscored pairs).")
        where: dict[str, Any] = Field(default_factory=dict,
                                      description="json: keep only entries whose fields have these values.")
        # TSV / CSV
        header: bool = Field(False, description="tsv/csv: the first row names the columns.")
        columns: dict[Literal["word1", "word2", "score"], str | int] = Field(
            default_factory=lambda: {"word1": 0, "word2": 1},
            description="tsv/csv: column (name or 0-based index) of word1, word2 and optionally score.")
        subset_by_file: bool = Field(False, description="Report results per file in addition to the total.")

        @model_validator(mode="after")
        def _json_fields(self):
            if self.format == "json" and (self.pair_field is None) == (self.word_fields is None):
                raise ValueError("json needs exactly one of 'pair_field' and 'word_fields'")
            if self.word_fields is not None and len(self.word_fields) != 2:
                raise ValueError("'word_fields' needs exactly two field names")
            return self

    def _json_pairs(self, path: Path, subset: str | None) -> list[WordPair]:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data[self.config.records_key] if self.config.records_key else data
        if not isinstance(entries, list):
            raise DataError(f"{path}: expected a list of entries under {self.config.records_key!r}")
        pairs: list[WordPair] = []
        for entry in entries:
            if any(entry.get(k) != v for k, v in self.config.where.items()):
                continue
            if self.config.pair_field:
                words = [w.strip() for w in str(entry[self.config.pair_field]).split(self.config.pair_separator)]
            else:
                words = [str(entry[f]).strip() for f in self.config.word_fields]
            if len(words) != 2 or not all(words):
                raise DataError(f"{path}: cannot read two words from {entry}")
            score = float(entry[self.config.score_field]) if self.config.score_field else None
            pairs.append(WordPair(words[0], words[1], score, subset))
        return pairs

    def _tabular_pairs(self, path: Path, subset: str | None) -> list[WordPair]:
        rows = _tabular_rows(path, "\t" if self.config.format == "tsv" else ",")
        header: list[str] | None = rows.pop(0) if self.config.header and rows else None
        index: dict[str, int] = {}
        for field, column in self.config.columns.items():
            if isinstance(column, int):
                index[field] = column
            elif header is None or column not in header:
                raise ConfigError(f"{path}: column {column!r} not found (header: {header})")
            else:
                index[field] = header.index(column)
        if "word1" not in index or "word2" not in index:
            raise ConfigError("'columns' needs word1 and word2")
        return [WordPair(row[index["word1"]].strip(), row[index["word2"]].strip(),
                         float(row[index["score"]]) if "score" in index else None, subset) for row in rows]

    def items(self) -> list[WordPair]:
        pairs: list[WordPair] = []
        for path in self.resolver.files():
            subset: str | None = path.stem if self.config.subset_by_file else None
            pairs += self._json_pairs(path, subset) if self.config.format == "json" else self._tabular_pairs(path, subset)
        return pairs


class WordChoicesDataset(LexicalDataset):
    """Multiple-choice items, one per line: target word, then the candidate words.

    Example (Latin synonym selection benchmark: target, synonym, three distractors)::

        dataset:
          source: word_choices
          url: https://zenodo.org/api/records/4022818/files/syn-selection-benchmark-Latin.tsv/content
          checksum: md5:6c0d8edd89abbf9db10654579e59f567
    """

    description = "Multiple-choice word items (target + candidates, e.g. synonym selection) from TSV/CSV files."

    class Config(FileSource):
        delimiter: str = Field("\t", description="Column separator.")
        header: bool = Field(False, description="The first row names the columns (it is skipped).")
        target_column: int = Field(0, ge=0, description="Column of the target word.")
        choices_start: int = Field(1, ge=0, description="First column of the candidate words (up to the end).")
        answer: int = Field(0, ge=0, description="Position of the correct candidate among the candidates.")

    def items(self) -> list[ChoiceItem]:
        items: list[ChoiceItem] = []
        for path in self.resolver.files():
            rows = _tabular_rows(path, self.config.delimiter)
            for row in rows[1:] if self.config.header else rows:
                choices = [c.strip() for c in row[self.config.choices_start:]]
                if len(choices) <= self.config.answer or len(choices) < 2:
                    raise DataError(f"{path}: item {row} has too few candidates")
                items.append(ChoiceItem(row[self.config.target_column].strip(), choices, self.config.answer))
        return items
