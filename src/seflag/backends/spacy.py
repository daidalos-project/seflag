from __future__ import annotations

from typing import Any

import spacy
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.tokens import Doc

from seflag.backends.base import Backend
from seflag.errors import MissingDependencyError, SeflagError

# installation commands for spaCy pipelines that are not distributed via PyPI
KNOWN_MODELS: dict[str, str] = {
    **{f"la_core_web_{size}": f"pip install https://huggingface.co/latincy/la_core_web_{size}/resolve/main/"
                              f"la_core_web_{size}-any-py3-none-any.whl" for size in ("sm", "md", "lg", "trf")},
    **{name: f"pip install grecy && python -m grecy install {name}"
       for name in ("grc_proiel_sm", "grc_proiel_lg", "grc_proiel_trf", "grc_perseus_sm", "grc_perseus_lg",
                    "grc_perseus_trf", "grc_odycy_joint_sm", "grc_odycy_joint_trf")},
}


def install_hint(name: str) -> str:
    command: str | None = KNOWN_MODELS.get(name)
    if command:
        return f"Install it with:\n    {command}"
    return "Install the model package, or set 'name' to a local model directory."


class SpacyBackend(Backend):
    """spaCy pipelines, e.g. LatinCy (la_core_web_*) and greCy (grc_*)."""

    description = "spaCy pipelines such as LatinCy (la_core_web_*) and greCy (grc_*)."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str = Field(description="Installed pipeline package name or path to a model directory.")
        exclude: list[str] = Field(default_factory=list, description="Pipeline components not to load.")
        disable: list[str] = Field(default_factory=list, description="Pipeline components to load but not run.")
        batch_size: int = Field(64, ge=1)
        gpu: bool = Field(False, description="Use the GPU if available (spacy.prefer_gpu).")

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            if self.config.gpu:
                spacy.prefer_gpu()
            # transformer-based greCy pipelines expect this extension to exist
            if not Doc.has_extension("trf_data"):
                Doc.set_extension("trf_data", default=None)
            try:
                self._nlp = spacy.load(self.config.name, exclude=self.config.exclude, disable=self.config.disable)
            except OSError as e:
                raise MissingDependencyError(
                    f"spaCy model '{self.config.name}' could not be loaded. {install_hint(self.config.name)}") from e
        return self._nlp

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        nlp: Language = self.nlp
        inputs: list[Doc] = [Doc(nlp.vocab, words=words) for words in sentences]
        if self.evaluation.options.sentence_segmentation == "gold":
            # spaCy parsers respect preset sentence boundaries: one input = one sentence
            for doc in inputs:
                for token in doc:
                    token.is_sent_start = token.i == 0
        outputs: list[Doc] = list(nlp.pipe(inputs, batch_size=self.config.batch_size))
        for words, doc in zip(sentences, outputs):
            if len(doc) != len(words):
                raise SeflagError(
                    f"Pipeline '{self.config.name}' changed the tokenization ({len(words)} -> {len(doc)} tokens). "
                    "Exclude retokenizing components (e.g. merge_entities) via 'exclude'.")
            for token in doc:
                # spaCy labels roots 'ROOT'; UD uses 'root'
                if token.dep_ == "ROOT":
                    token.dep_ = "root"
        return outputs

    def label_inventory(self, layer: str) -> set[str] | None:
        if layer == "entities":
            labels: set[str] = set()
            for _, pipe in self.nlp.pipeline:
                if pipe.__class__.__name__ in ("EntityRecognizer", "EntityRuler", "SpanRuler"):
                    labels |= set(pipe.labels)
            return labels or None
        if layer != "morph":
            return None
        names: set[str] = set()
        found: bool = False
        for _, pipe in self.nlp.pipeline:
            if pipe.__class__.__name__ == "Morphologizer":
                found = True
                # labels look like "POS=NOUN|Case=Nom|Number=Sing"; POS is not a morphological feature
                names |= {kv.split("=", 1)[0] for label in pipe.labels for kv in label.split("|") if "=" in kv}
        names.discard("POS")
        return names if found else None

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = self.nlp.meta
        return {
            "name": f"{meta.get('lang')}_{meta.get('name')}",
            "version": meta.get("version"),
            "spacy_version": meta.get("spacy_version"),
            "pipeline": self.nlp.pipe_names,
            "disabled": list(self.nlp.disabled),
        }
