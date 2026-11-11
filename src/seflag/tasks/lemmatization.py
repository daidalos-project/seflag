from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Token

from seflag.tasks.base import TokenLabelTask


class LemmatizationTask(TokenLabelTask):
    """Token-level lemma accuracy on gold tokenization."""

    description = "Lemmatization: accuracy of predicted lemmas per syntactic word."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        exclude_upos: list[str] = Field(
            default_factory=list,
            description="Gold UPOS tags whose tokens are not scored, e.g. [PUNCT, NUM]. Requires UPOS in the gold data.")

    @property
    def required_layers(self) -> set[str]:
        return {"lemma", "upos"} if self.config.exclude_upos else {"lemma"}

    def label(self, token: Token) -> str:
        return token.lemma_

    def include(self, gold_token: Token) -> bool:
        return gold_token.pos_ not in self.config.exclude_upos
