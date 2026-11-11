"""Static word embeddings (word2vec, fastText, ...) loaded with gensim."""

from __future__ import annotations

import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from seflag.config import EvaluationConfig
from seflag.data.files import FileResolver, FileSource
from seflag.errors import ConfigError, MissingDependencyError


class EmbeddingBackend(ABC):
    """Type-level models: one vector per word form (or lemma)."""

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

    description: ClassVar[str] = ""

    def __init__(self, config: Any, evaluation: EvaluationConfig):
        self.config = config
        self.evaluation = evaluation

    @property
    @abstractmethod
    def supports_subwords(self) -> bool:
        """Whether vectors can be built for words outside the vocabulary (e.g. fastText character n-grams)."""

    @abstractmethod
    def key(self, word: str) -> str | None:
        """The vocabulary entry a word maps to (e.g. ignoring case), or None if it is not in the vocabulary."""

    def in_vocabulary(self, word: str) -> bool:
        return self.key(word) is not None

    @abstractmethod
    def vector(self, word: str, subwords: bool = False) -> np.ndarray | None:
        """The word's vector; None if it is not in the vocabulary (and subwords are not used or not available)."""

    @abstractmethod
    def neighbours(self, word: str, k: int, subwords: bool = False) -> list[str]:
        """The vocabulary entries of the k nearest words by cosine similarity, excluding the word itself."""

    def similarity(self, word1: str, word2: str, subwords: bool = False) -> float | None:
        v1, v2 = self.vector(word1, subwords), self.vector(word2, subwords)
        if v1 is None or v2 is None:
            return None
        norm: float = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.dot(v1, v2) / norm) if norm else 0.0

    def metadata(self) -> dict[str, Any]:
        return {}


class GensimBackend(EmbeddingBackend):
    """Embeddings in the formats gensim reads: fastText ``.bin``, word2vec binary/text (``.bin``/``.vec``),
    and gensim's own ``KeyedVectors`` files. The file can be local, a URL (e.g. Zenodo, with checksum) or in a
    Hub repository; ``archive_member`` picks a file out of a ``.zip`` download (e.g. NLPL repository models).
    """

    description = "Static word embeddings via gensim: fastText .bin, word2vec .bin/.vec, gensim KeyedVectors."

    class Config(FileSource):
        format: Literal["fasttext", "word2vec", "word2vec_text", "gensim"] = Field(
            description="fasttext (.bin with subwords), word2vec (binary), word2vec_text (.vec/.txt), gensim (.kv).")
        archive_member: str | None = Field(None, description="File inside a .zip archive to load.")
        max_vocabulary: int | None = Field(
            None, ge=1, description="Use only the N most frequent words (gensim's evaluate_word_pairs uses 300000 "
                                    "by default). Default: the whole vocabulary.")
        case_insensitive: bool = Field(
            False, description="Match words regardless of case (as gensim's evaluate_word_pairs does by default; "
                               "the most frequent entry wins).")

    def __init__(self, config: Any, evaluation: EvaluationConfig):
        super().__init__(config, evaluation)
        self.resolver = FileResolver(config, evaluation)
        self._kv = None
        self._path: Path | None = None
        self._metadata: dict[str, Any] | None = None
        self._keys: dict[str, str] | None = None

    def _model_file(self) -> Path:
        files: list[Path] = self.resolver.files()
        if len(files) != 1:
            raise ConfigError(f"The embedding model must be exactly one file, found {len(files)}: {files[:5]}")
        path: Path = files[0]
        if self.config.archive_member:
            target: Path = path.parent / f"{path.name}.extracted" / self.config.archive_member
            if not target.is_file():
                with zipfile.ZipFile(path) as archive:
                    archive.extract(self.config.archive_member, target.parent.parent / f"{path.name}.extracted")
            path = target
        return path

    @property
    def kv(self):
        if self._kv is None:
            try:
                from gensim.models import KeyedVectors
                from gensim.models.fasttext import load_facebook_vectors
            except ImportError as e:
                raise MissingDependencyError("Word embeddings need gensim: pip install 'seflag[embeddings]'") from e
            self._path = self._model_file()
            fmt: str = self.config.format
            if fmt == "fasttext":
                self._kv = load_facebook_vectors(str(self._path))
            elif fmt == "gensim":
                self._kv = KeyedVectors.load(str(self._path))
            else:
                self._kv = KeyedVectors.load_word2vec_format(str(self._path), binary=fmt == "word2vec")
        return self._kv

    @property
    def supports_subwords(self) -> bool:
        return self.config.format == "fasttext"

    @property
    def keys(self) -> dict[str, str]:
        """Lookup form -> vocabulary entry, restricted to ``max_vocabulary`` (gensim keeps words by frequency)."""
        if self._keys is None:
            entries: list[str] = list(self.kv.index_to_key[:self.config.max_vocabulary])
            if self.config.case_insensitive:
                # reversed, so that the most frequent of several entries with the same upper-case form wins
                self._keys = {w.upper(): w for w in reversed(entries)}
            else:
                self._keys = {w: w for w in entries}
        return self._keys

    def key(self, word: str) -> str | None:
        # not `word in kv`: fastText vectors claim every word, because they can build vectors from n-grams
        return self.keys.get(word.upper() if self.config.case_insensitive else word)

    def vector(self, word: str, subwords: bool = False) -> np.ndarray | None:
        key: str | None = self.key(word)
        if key is not None:
            return self.kv[key]
        if subwords and self.supports_subwords:
            return self.kv[word]
        return None

    def neighbours(self, word: str, k: int, subwords: bool = False) -> list[str]:
        key: str | None = self.key(word)
        if key is not None:
            query: Any = key  # gensim excludes the query entry itself
        elif subwords and self.supports_subwords:
            query = self.kv[word]
        else:
            return []
        found = self.kv.most_similar(positive=[query], topn=k + 1, restrict_vocab=self.config.max_vocabulary)
        return [w for w, _ in found if w != key][:k]

    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata = self._collect_metadata()
        return self._metadata

    def _collect_metadata(self) -> dict[str, Any]:
        import gensim

        kv = self.kv
        source: dict[str, Any] = {k: getattr(self.config, k) for k in ("url", "checksum", "repo", "revision", "path")
                                  if getattr(self.config, k) not in (None, ".")}
        if self.config.checksum is None and self._path is not None:  # identify the exact file
            from seflag.data.download import file_checksum

            source["sha256"] = file_checksum(self._path, "sha256")
        return {"file": self._path.name if self._path else None, "format": self.config.format,
                "source": source, "vocabulary_size": len(kv.key_to_index),
                "used_vocabulary_size": len(self.keys), "max_vocabulary": self.config.max_vocabulary,
                "case_insensitive": self.config.case_insensitive,
                "dimensions": kv.vector_size, "subwords": self.supports_subwords, "gensim_version": gensim.__version__}
