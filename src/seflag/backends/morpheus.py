"""Morpheus (Perseus morphological analyser) as a lemmatizer for Ancient Greek and Latin.

Morpheus is an external program: see https://github.com/perseids-tools/morpheus-perseids for installation.
It analyses words out of context and may return several lemmas; ``select`` decides which one is used.
"""

from __future__ import annotations

import os
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Doc
from spacy.vocab import Vocab

from seflag.backends.base import Backend
from seflag.errors import ConfigError, MissingDependencyError, SeflagError


def _betacode():
    try:
        import betacode.conv
    except ImportError as e:
        raise MissingDependencyError("Morpheus for Greek needs the betacode package: pip install betacode") from e
    return betacode.conv


class MorpheusBackend(Backend):
    """Lemmatization with Morpheus; Greek words are converted to and from beta code automatically."""

    description = "Morpheus morphological analyser (external program) as lemmatizer for Greek and Latin."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str | None = Field(
            None, description="Morpheus installation directory (with bin/ and stemlib/); default: $MORPHEUS_PATH.")
        language: Literal["auto", "grc", "la"] = Field("auto", description="'auto': the evaluation's language.")
        select: Literal["first", "last"] = Field("first", description="Which lemma to use if Morpheus returns several.")
        unknown: Literal["empty", "form"] = Field(
            "empty", description="Lemma for words Morpheus cannot analyse: empty (scored as wrong) or the word itself.")
        non_words: Literal["form", "empty"] = Field(
            "form", description="Lemma for tokens without letters (punctuation, numbers), which are not sent to "
                                "Morpheus: the token itself (as in UD) or empty.")
        batch_size: int = Field(5000, ge=1, description="Words per Morpheus process.")
        timeout: int = Field(600, ge=1, description="Seconds per Morpheus process.")

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        language: str = evaluation.language if config.language == "auto" else config.language
        if language not in ("grc", "la"):
            raise ConfigError(f"Morpheus supports Ancient Greek (grc) and Latin (la), not '{language}'.")
        self.language: str = language
        self.stats: dict[str, int] = {"words": 0, "unknown": 0, "ambiguous": 0}

    @property
    def root(self) -> Path:
        path: str | None = self.config.path or os.environ.get("MORPHEUS_PATH")
        if not path:
            raise MissingDependencyError("Morpheus not configured: set 'path' or the MORPHEUS_PATH variable. "
                                         "Installation: https://github.com/perseids-tools/morpheus-perseids")
        root = Path(path).expanduser()
        if not (root / "bin" / "morpheus").is_file():
            raise MissingDependencyError(f"No Morpheus program at {root / 'bin' / 'morpheus'}")
        return root

    def _to_input(self, word: str) -> str:
        if self.language == "la":
            return word
        beta: str = _betacode().uni_to_beta(word)
        return beta.replace("s1", "s").replace("s2", "s").replace("s3", "s")

    def _to_output(self, lemma: str) -> str:
        if self.language == "la":
            return lemma
        # betacode produces accents with oxia (e.g. U+1F77); NFC turns them into the canonically equivalent
        # tonos forms (U+03AF) used in UD and most Unicode Greek text
        return unicodedata.normalize("NFC", _betacode().beta_to_uni(lemma))

    def _analyse(self, words: list[str]) -> list[list[str]]:
        """Returns the lemmas Morpheus proposes for each word (empty list = unknown)."""
        args: list[str] = [str(self.root / "bin" / "morpheus")] + (["-L"] if self.language == "la" else [])
        env: dict[str, str] = {**os.environ, "MORPHLIB": "stemlib"}
        results: list[list[str]] = []
        for offset in range(0, len(words), self.config.batch_size):
            batch: list[str] = words[offset:offset + self.config.batch_size]
            # one word per line; whitespace inside a word would break the line protocol
            payload: str = "\n".join(self._to_input(w).replace(" ", "").replace("\t", "") or "_" for w in batch) + "\n"
            done = subprocess.run(args, input=payload, capture_output=True, text=True, env=env, cwd=self.root,
                                  timeout=self.config.timeout)
            if done.returncode != 0:
                raise SeflagError(f"Morpheus failed ({done.returncode}): {done.stderr[:500]}")
            root = ET.fromstring(done.stdout)
            items = [el for el in root if el.tag in ("word", "unknown")]
            # Morpheus echoes each input (<form> or <unknown>), but skips some inputs entirely (e.g. numbers):
            # align by the echoed text instead of by position
            inputs: list[str] = payload.split("\n")[:-1]
            position: int = 0
            for text in inputs:
                lemmas: list[str] = []
                # Morpheus lowercases the echo of some inputs (e.g. unknown words)
                if position < len(items) and self._echo(items[position]).lower() == text.lower():
                    for hdwd in items[position].iter("hdwd"):
                        if hdwd.text and hdwd.text not in lemmas:
                            lemmas.append(hdwd.text)
                    position += 1
                results.append(lemmas)
            if position != len(items):
                raise SeflagError(f"Could not align Morpheus output: {len(items) - position} analyses left over.")
        return results

    @staticmethod
    def _echo(element: ET.Element) -> str:
        if element.tag == "unknown":
            return element.text or ""
        form = element.find("form")
        return form.text if form is not None and form.text else ""

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        all_words: set[str] = {w for s in sentences for w in s}
        lemma_of: dict[str, str] = {w: (w if self.config.non_words == "form" else "")
                                    for w in all_words if not any(c.isalpha() for c in w)}
        unique: list[str] = sorted(all_words - lemma_of.keys())
        for word, lemmas in zip(unique, self._analyse(unique)):
            self.stats["words"] += 1
            if not lemmas:
                self.stats["unknown"] += 1
                lemma_of[word] = word if self.config.unknown == "form" else ""
                continue
            self.stats["ambiguous"] += len(lemmas) > 1
            lemma_of[word] = self._to_output(lemmas[0] if self.config.select == "first" else lemmas[-1])
        vocab = Vocab()
        return [Doc(vocab, words=words, lemmas=[lemma_of[w] for w in words]) for words in sentences]

    def metadata(self) -> dict[str, Any]:
        info: dict[str, Any] = {"name": "morpheus", "path": str(self.root), "language": self.language,
                                "unique_words": self.stats["words"], "unknown_words": self.stats["unknown"],
                                "ambiguous_words": self.stats["ambiguous"]}
        head = self.root / ".git" / "HEAD"
        if head.is_file():  # record the Morpheus version if it is a git checkout
            ref: str = head.read_text().strip()
            ref_file = self.root / ".git" / ref.removeprefix("ref: ") if ref.startswith("ref: ") else None
            info["git_commit"] = ref_file.read_text().strip() if ref_file and ref_file.is_file() else ref
        return info
