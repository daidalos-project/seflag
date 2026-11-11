from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from spacy.tokens import Doc

from seflag.config import EvaluationConfig


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
