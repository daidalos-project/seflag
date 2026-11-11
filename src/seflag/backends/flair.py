"""flair sequence taggers (NER, POS) on pre-tokenized input."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from spacy.parts_of_speech import IDS as SPACY_POS_IDS
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from seflag.backends.base import Backend
from seflag.data.tags import TagScheme
from seflag.errors import MissingDependencyError, SeflagError

MODEL_FILES: tuple[str, ...] = ("pytorch_model.bin", "final-model.pt", "best-model.pt")


def _require_flair():
    try:
        import flair
    except ImportError as e:
        raise MissingDependencyError("The flair backend needs flair: pip install 'seflag[flair]'") from e
    return flair


class FlairBackend(Backend):
    """flair ``SequenceTagger`` models, e.g. UGARIT/flair_grc_bert_ner.

    Hub models are downloaded with ``huggingface_hub`` so that a ``revision`` can be pinned; local model files
    (``.pt``) are loaded directly. Entity taggers produce spans, POS taggers token labels.
    """

    description = "flair SequenceTagger models (NER spans, or UPOS/XPOS token labels)."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str = Field(description="Hub repository id, or path to a local model file.")
        revision: str | None = Field(None, description="Hub revision; pin a commit hash for reproducible results.")
        filename: str | None = Field(None, description=f"Model file in the repository (default: first of {MODEL_FILES}).")
        layer: Literal["auto", "entities", "upos", "xpos"] = Field(
            "auto", description="What the tagger predicts; 'auto': entities for span taggers, else from tag type.")
        tag_type: str | None = Field(None, description="flair label type (default: the tagger's own tag type).")
        batch_size: int = Field(32, ge=1)

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        self._tagger = None
        self._model_path: Path | None = None

    def _resolve_model_path(self) -> Path:
        local = Path(self.config.name).expanduser()
        if local.is_file():
            return local
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as e:
            raise MissingDependencyError("Loading flair models from the Hub needs huggingface_hub") from e
        filename: str | None = self.config.filename
        if filename is None:
            files: list[str] = list_repo_files(self.config.name, revision=self.config.revision)
            filename = next((f for f in MODEL_FILES if f in files), None)
            if filename is None:
                raise SeflagError(f"No flair model file ({', '.join(MODEL_FILES)}) in {self.config.name}; "
                                  "set 'filename'.")
        return Path(hf_hub_download(self.config.name, filename, revision=self.config.revision))

    @property
    def tagger(self):
        if self._tagger is None:
            _require_flair()
            from flair.models import SequenceTagger

            self._model_path = self._resolve_model_path()
            self._tagger = SequenceTagger.load(str(self._model_path))
        return self._tagger

    @property
    def tag_type(self) -> str:
        return self.config.tag_type or self.tagger.tag_type

    @property
    def layer(self) -> str:
        if self.config.layer != "auto":
            return self.config.layer
        if getattr(self.tagger, "predict_spans", False) or "ner" in self.tag_type.lower():
            return "entities"
        return "upos" if self.tag_type.lower() in ("upos", "pos") else "xpos"

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        _require_flair()
        from flair.data import Sentence

        flair_sentences = [Sentence(words) for words in sentences]
        for words, sentence in zip(sentences, flair_sentences):
            if len(sentence) != len(words):
                raise SeflagError(f"flair changed the tokenization of {words[:5]}...")
        self.tagger.predict(flair_sentences, mini_batch_size=self.config.batch_size)
        vocab = Vocab()
        docs: list[Doc] = []
        layer, tag_type = self.layer, self.tag_type
        for words, sentence in zip(sentences, flair_sentences):
            if layer == "entities":
                doc = Doc(vocab, words=words)
                doc.ents = [Span(doc, span.tokens[0].idx - 1, span.tokens[-1].idx, label=span.get_label(tag_type).value)
                            for span in sentence.get_spans(tag_type)]
            else:
                labels: list[str] = [t.get_label(tag_type).value if t.get_labels(tag_type) else "" for t in sentence]
                if layer == "upos":
                    doc = Doc(vocab, words=words, pos=[t if t in SPACY_POS_IDS else "" for t in labels])
                else:
                    doc = Doc(vocab, words=words, tags=labels)
            docs.append(doc)
        return docs

    def label_inventory(self, layer: str) -> set[str] | None:
        if layer != self.layer:
            return None
        labels: list[str] = [lab for lab in self.tagger.label_dictionary.get_items() if lab not in ("<unk>", "O")]
        if layer == "entities":
            scheme = TagScheme()
            return {p[1] for p in map(scheme.parse, labels) if p is not None and p[1] not in ("<START>", "<STOP>")}
        return set(labels) - {"<START>", "<STOP>"}

    def metadata(self) -> dict[str, Any]:
        flair = _require_flair()
        tagger = self.tagger
        resolved = self._model_path.parent.name if self._model_path and "snapshots" in self._model_path.parts else None
        return {"name": self.config.name, "revision": self.config.revision, "resolved_revision": resolved,
                "file": self._model_path.name if self._model_path else None, "tag_type": self.tag_type,
                "layer": self.layer, "flair_version": flair.__version__}
