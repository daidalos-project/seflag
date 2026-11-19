from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Token

from seflag.enums import Layer, Tagset
from seflag.metrics import classification_report, confusion_matrix
from seflag.tasks.base import TokenLabelTask


class PosTaggingTask(TokenLabelTask):
    """Part-of-speech tagging with UPOS (Universal Dependencies) or treebank-specific XPOS tags."""

    description = "Part-of-speech tagging: accuracy, per-tag precision/recall/F1 and confusion matrix."
    normalize_labels = False  # string normalizers are meant for lemmas/forms, not for tag sets

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tagset: Tagset = Field(Tagset.UPOS, description="Which tag layer to compare.")

    @property
    def required_layers(self) -> set[Layer]:
        return {Layer.UPOS if self.config.tagset == Tagset.UPOS else Layer.XPOS}

    def label(self, token: Token) -> str:
        return token.pos_ if self.config.tagset == Tagset.UPOS else token.tag_

    def extra_metrics(self, gold: list[str], pred: list[str]) -> dict[str, Any]:
        report: dict[str, Any] = classification_report(gold, pred)
        report["confusion_matrix"] = confusion_matrix(gold, pred)
        return report
