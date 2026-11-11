from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Token

from seflag.metrics import classification_report, confusion_matrix
from seflag.tasks.base import TokenLabelTask


class PosTaggingTask(TokenLabelTask):
    """Part-of-speech tagging with UPOS (Universal Dependencies) or treebank-specific XPOS tags."""

    description = "Part-of-speech tagging: accuracy, per-tag precision/recall/F1 and confusion matrix."
    normalize_labels = False  # string normalizers are meant for lemmas/forms, not for tag sets

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        tagset: Literal["upos", "xpos"] = Field("upos", description="Which tag layer to compare.")

    @property
    def required_layers(self) -> set[str]:
        return {self.config.tagset}

    def label(self, token: Token) -> str:
        return token.pos_ if self.config.tagset == "upos" else token.tag_

    def extra_metrics(self, gold: list[str], pred: list[str]) -> dict[str, Any]:
        report: dict[str, Any] = classification_report(gold, pred)
        report["confusion_matrix"] = confusion_matrix(gold, pred)
        return report
