from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from collections.abc import Iterable, Iterator

from pydantic import BaseModel, ConfigDict
from spacy.tokens import Doc
from spacy.vocab import Vocab

from seflag.config import EvaluationConfig
from seflag.data.records import SentenceRecord, check_layers_present, present_layers, record_to_doc


class DatasetAdapter(ABC):
    """Reads a dataset in some source format and yields :class:`SentenceRecord` objects.

    Subclasses define a pydantic ``Config`` class for their configuration fields and implement
    :meth:`records`. Everything else (conversion to spaCy ``Doc``, sentence limits) is shared.
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

    description: ClassVar[str] = ""

    def __init__(self, config: Any, evaluation: EvaluationConfig):
        self.config = config
        self.evaluation: EvaluationConfig = evaluation

    @abstractmethod
    def records(self) -> Iterator[SentenceRecord]:
        """Yields the dataset's sentences in their original order."""

    def fingerprint(self) -> dict[str, Any]:
        """Information that identifies the exact version of the data, stored with the results."""
        return {}

    def load_docs(self, vocab: Vocab, layers: set[str], max_sentences: int | None = None) -> list[Doc]:
        records: Iterator[SentenceRecord] = self.records()
        if max_sentences is not None:
            records = itertools.islice(records, max_sentences)
        loaded: list[SentenceRecord] = list(records)
        check_layers_present(loaded, layers)
        return [record_to_doc(vocab, r, layers) for r in loaded]

    def docs(self, layers: Iterable[str] | None = None, max_sentences: int | None = None,
             vocab: Vocab | None = None) -> list[Doc]:
        """The gold sentences as spaCy ``Doc`` objects, e.g. for use outside of an evaluation.

        ``layers``: annotation layers to include (see :data:`seflag.data.records.LAYERS`); default: all layers
        annotated in the data. Sentence ids, texts and subsets are kept in ``Doc.user_data``.
        """
        records: list[SentenceRecord] = list(itertools.islice(self.records(), max_sentences))
        chosen: set[str] = set(layers) if layers is not None else present_layers(records)
        check_layers_present(records, chosen)
        vocab = vocab or Vocab()
        return [record_to_doc(vocab, r, chosen) for r in records]
