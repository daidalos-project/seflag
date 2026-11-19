from __future__ import annotations

from typing import Any

import spacy
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.tokens import Doc

from seflag.enums import Layer, SentenceSegmentation
from seflag.backends.base import Backend
from seflag.errors import MissingDependencyError, SeflagError

# Wheels of spaCy pipelines that are not distributed via PyPI. pip only accepts wheel file names with a concrete
# version (PEP 427), so the URLs name one, and they are pinned to a repository revision or release, because the
# maintainers delete old wheels from their main branches. tests/test_readme.py checks the file names (and, with
# network access, that every URL exists).
LATINCY_WHEELS: dict[str, tuple[str, str]] = {  # size: (version, revision of latincy/la_core_web_<size>)
    "sm": ("3.9.8", "c433837a66f74643ba49bf853b87927954c7d701"),
    "md": ("3.9.8", "45c9ac84afecfeb54bc0f6abacbc0aaf9ce1850e"),
    "lg": ("3.9.8", "eef57207424f80d2e3806315bdc08554f432f740"),
    "trf": ("3.9.9", "8e4accad1fd30bc962703d00ac3e7c0374125af0"),
}
# greCy's GitHub release with the model versions of SEFLAG's published results (the Hub repositories are gone)
GRECY_RELEASE = "3.7.5"
GRECY_MODELS = ("grc_proiel_sm", "grc_proiel_lg", "grc_proiel_trf", "grc_perseus_sm", "grc_perseus_lg",
                "grc_perseus_trf", "grc_ner_trf")
# odyCy; grc_odycy_joint_sm is only published under an invalid wheel name that pip rejects
ODYCY_WHEELS: dict[str, tuple[str, str]] = {
    "grc_odycy_joint_trf": ("0.7.0", "83046e93fa6b5dee122c3ca7cff1377512928143"),
}

KNOWN_MODELS: dict[str, str] = {
    **{f"la_core_web_{size}": f"pip install https://huggingface.co/latincy/la_core_web_{size}/resolve/{revision}/"
                              f"la_core_web_{size}-{version}-py3-none-any.whl"
       for size, (version, revision) in LATINCY_WHEELS.items()},
    **{name: f"pip install https://github.com/jmyerston/greCy/releases/download/v{GRECY_RELEASE}/"
             f"{name}-{GRECY_RELEASE}-py3-none-any.whl" for name in GRECY_MODELS},
    **{name: f"pip install https://huggingface.co/chcaa/{name}/resolve/{revision}/{name}-{version}-py3-none-any.whl"
       for name, (version, revision) in ODYCY_WHEELS.items()},
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
        if self.evaluation.options.sentence_segmentation == SentenceSegmentation.GOLD:
            # spaCy parsers respect preset sentence boundaries: one input = one sentence
            for doc in inputs:
                for token in doc:
                    token.is_sent_start = token.i == 0
        outputs: list[Doc] = list(nlp.pipe(inputs, batch_size=self.config.batch_size))
        for words, doc in zip(sentences, outputs, strict=True):
            if len(doc) != len(words):
                culprit = self._retokenizing_component(words)
                raise SeflagError(
                    f"Pipeline '{self.config.name}' changed the tokenization ({len(words)} -> {len(doc)} tokens)"
                    + (f" in its component '{culprit}'. Add it to 'exclude' in the model section."
                       if culprit else ". Exclude retokenizing components (e.g. merge_entities) via 'exclude'.")
                    + f" Sentence: {' '.join(words)}")
            for token in doc:
                # spaCy labels roots 'ROOT'; UD uses 'root'
                if token.dep_ == "ROOT":
                    token.dep_ = "root"
        return outputs

    def _retokenizing_component(self, words: list[str]) -> str | None:
        """The first pipeline component that changes the number of tokens of a sentence (for error messages)."""
        doc: Doc = Doc(self.nlp.vocab, words=words)
        for name, component in self.nlp.pipeline:
            doc = component(doc)
            if len(doc) != len(words):
                return name
        return None

    def label_inventory(self, layer: Layer) -> set[str] | None:
        if layer == Layer.ENTITIES:
            labels: set[str] = set()
            for _, pipe in self.nlp.pipeline:
                if pipe.__class__.__name__ in ("EntityRecognizer", "EntityRuler", "SpanRuler"):
                    labels |= set(pipe.labels)
            return labels or None
        if layer != Layer.MORPH:
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
