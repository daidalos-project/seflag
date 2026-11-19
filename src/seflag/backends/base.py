from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from spacy.tokens import Doc

from seflag.config import EvaluationConfig
from seflag.errors import PredictionError


class Backend(ABC):
    """Wraps an NLP platform (spaCy, flair, Transformers, ...) and turns its output into spaCy ``Doc`` objects.

    With gold tokenization, :meth:`predict` receives pre-tokenized sentences and must return one ``Doc``
    per sentence with exactly the same tokens, so predictions align 1:1 with the gold standard.
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

    description: ClassVar[str] = ""

    def __init__(self, config: Any, evaluation: EvaluationConfig):
        self.config = config
        self.evaluation: EvaluationConfig = evaluation

    @abstractmethod
    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        """Annotates pre-tokenized sentences."""

    def label_inventory(self, layer: str) -> set[str] | None:
        """Labels the model can predict for an annotation layer (for 'morph': feature names), if known."""
        return None

    def metadata(self) -> dict[str, Any]:
        """Model name, version etc., stored with the results."""
        return {}


def check_predictions(sentences: list[list[str]], predicted: list[Doc], backend: str) -> None:
    """Raises :class:`PredictionError` unless ``predicted`` has one ``Doc`` per sentence with exactly its tokens.

    The runner calls this after every prediction; backend authors can use it in their own tests.
    """
    if len(predicted) != len(sentences):
        raise PredictionError(
            f"Backend '{backend}' returned {len(predicted)} documents for {len(sentences)} sentences.")
    for number, (words, doc) in enumerate(zip(sentences, predicted, strict=True), start=1):
        tokens: list[str] = [token.text for token in doc]
        if tokens == words:
            continue
        position: int = next((i for i, (a, b) in enumerate(zip(words, tokens, strict=False)) if a != b),
                             min(len(words), len(tokens)))
        expected: str = repr(words[position]) if position < len(words) else "(end of sentence)"
        found: str = repr(tokens[position]) if position < len(tokens) else "(end of sentence)"
        raise PredictionError(
            f"Backend '{backend}' changed the tokens of sentence {number}: {len(words)} tokens given, {len(tokens)} "
            f"returned; first difference at token {position + 1}: expected {expected}, found {found}. Backends must "
            "return exactly the given tokens, so that predictions align with the gold standard.")
