"""Hugging Face Transformers token-classification models (NER or POS tagging) on pre-tokenized input."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from spacy.parts_of_speech import IDS as SPACY_POS_IDS
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from seflag.backends.base import Backend
from seflag.data.tags import TagScheme, decode
from seflag.errors import MissingDependencyError, SeflagError


def _require_transformers():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise MissingDependencyError("The transformers backend needs: pip install transformers torch") from e
    return transformers, torch


class TransformersBackend(Backend):
    """``AutoModelForTokenClassification`` models, e.g. UGARIT/grc-ner-bert.

    Words are passed to the tokenizer as pre-tokenized input; each word gets the label predicted for its first
    subword. Sentences longer than the model's maximum input are split into windows of whole words.
    """

    description = "Hugging Face Transformers token-classification models (NER via BIO labels, or UPOS/XPOS)."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str = Field(description="Hub model id or local directory.")
        revision: str | None = Field(None, description="Hub revision; pin a commit hash for reproducible results.")
        layer: Literal["entities", "upos", "xpos"] = Field(
            "entities", description="What the labels are: BIO entity tags, UPOS tags or XPOS tags.")
        tag_scheme: TagScheme = Field(default_factory=TagScheme, description="Encoding of entity labels.")
        batch_size: int = Field(16, ge=1)
        max_length: int | None = Field(None, ge=8, description="Maximum subwords per window (default: model limit).")
        device: str = Field("auto", description="'auto' (GPU if available), 'cpu', 'cuda', 'cuda:1', ...")

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        transformers, torch = _require_transformers()
        kwargs: dict[str, Any] = {"revision": self.config.revision}
        model_config = transformers.AutoConfig.from_pretrained(self.config.name, **kwargs)
        tokenizer_kwargs: dict[str, Any] = dict(kwargs)
        if model_config.model_type in {"roberta", "gpt2", "bart", "longformer", "deberta"}:
            tokenizer_kwargs["add_prefix_space"] = True  # needed for pre-tokenized input with BPE tokenizers
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.name, **tokenizer_kwargs)
        if not self._tokenizer.is_fast:
            raise SeflagError(f"Model '{self.config.name}' has no fast tokenizer, which is needed for word alignment.")
        self._model = transformers.AutoModelForTokenClassification.from_pretrained(self.config.name, **kwargs)
        device: str = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model.to(self._device).eval()

    @property
    def labels(self) -> dict[int, str]:
        self._load()
        return {int(k): v for k, v in self._model.config.id2label.items()}

    def _limit(self) -> int:
        limit: int = self.config.max_length or min(self._tokenizer.model_max_length, 4096)
        return limit - self._tokenizer.num_special_tokens_to_add()

    def _windows(self, words: list[str], lengths: dict[str, int]) -> list[tuple[int, int]]:
        """Splits a sentence into word ranges whose subwords fit into the model."""
        limit: int = self._limit()
        windows: list[tuple[int, int]] = []
        start, size = 0, 0
        for i, word in enumerate(words):
            n: int = max(1, lengths[word])
            if size + n > limit and i > start:
                windows.append((start, i))
                start, size = i, 0
            size += n
        windows.append((start, len(words)))
        return windows

    def _predict_labels(self, sentences: list[list[str]]) -> list[list[str]]:
        _, torch = _require_transformers()
        self._load()
        unique: set[str] = {w for s in sentences for w in s}
        lengths: dict[str, int] = {w: len(self._tokenizer(w, add_special_tokens=False)["input_ids"]) for w in unique}
        chunks: list[tuple[int, int, int]] = []  # sentence index, word start, word end
        for si, words in enumerate(sentences):
            chunks += [(si, a, b) for a, b in self._windows(words, lengths)]
        results: list[list[str]] = [[""] * len(s) for s in sentences]
        id2label: dict[int, str] = self.labels
        for offset in range(0, len(chunks), self.config.batch_size):
            batch = chunks[offset:offset + self.config.batch_size]
            inputs = self._tokenizer([sentences[si][a:b] for si, a, b in batch], is_split_into_words=True,
                                     truncation=True, max_length=self._limit() + self._tokenizer.num_special_tokens_to_add(),
                                     padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**{k: v.to(self._device) for k, v in inputs.items()}).logits
            predictions = logits.argmax(-1).cpu().tolist()
            for row, (si, a, _) in enumerate(batch):
                seen: set[int] = set()
                for position, word_id in enumerate(inputs.word_ids(row)):
                    if word_id is None or word_id in seen:
                        continue  # special tokens and non-first subwords
                    seen.add(word_id)
                    results[si][a + word_id] = id2label[predictions[row][position]]
        return results

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        labels: list[list[str]] = self._predict_labels(sentences)
        vocab = Vocab()
        docs: list[Doc] = []
        for words, tags in zip(sentences, labels):
            if self.config.layer == "upos":
                doc = Doc(vocab, words=words, pos=[t if t in SPACY_POS_IDS else "" for t in tags])
            elif self.config.layer == "xpos":
                doc = Doc(vocab, words=words, tags=tags)
            else:
                doc = Doc(vocab, words=words)
                doc.ents = [Span(doc, e.first - 1, e.last, label=e.label) for e in decode(tags, self.config.tag_scheme)]
            docs.append(doc)
        return docs

    def label_inventory(self, layer: str) -> set[str] | None:
        if layer != self.config.layer:
            return None
        if layer == "entities":
            parsed = (self.config.tag_scheme.parse(t) for t in self.labels.values())
            return {p[1] for p in parsed if p is not None}
        return set(self.labels.values())

    def metadata(self) -> dict[str, Any]:
        transformers, _ = _require_transformers()
        self._load()
        return {"name": self.config.name, "revision": self.config.revision,
                "resolved_revision": getattr(self._model.config, "_commit_hash", None),
                "transformers_version": transformers.__version__, "labels": sorted(self.labels.values()),
                "device": str(self._device)}
