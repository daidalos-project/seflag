"""Lamon (lamonpy) as a lemmatizer, POS tagger and morphological tagger for Latin.

Lamon (https://github.com/bab2min/lamonpy, MIT license) is a BiLSTM tagger trained on the Perseus Latin
Dependency Treebank and self-trained on Latin text. Its last release, lamonpy 0.2.0 (2020), has wheels only up to
Python 3.8, and its source distribution no longer builds with current compilers. By default this backend therefore
runs lamonpy in a separate Python 3.8 environment that `uv <https://docs.astral.sh/uv/>`_ creates on first use
(uv downloads Python 3.8 and the 42 MB wheel once and caches them). There, the small script
:mod:`seflag.backends.lamon_bridge` tags the sentences; that environment needs lamonpy only, not SEFLAG.
``command`` replaces uv with any interpreter that has lamonpy installed.

Lamon tokenizes the text itself. The backend gives it each sentence as the gold tokens joined by spaces and maps
Lamon's tokens back to the gold tokens by their character offsets; if Lamon splits a gold token (e.g. '...'), the
first of its parts counts, and tokens without letters or digits keep their form as lemma.

Lamon's lemmas follow its training data: 'v' and 'j' (*venio*, *Julius*), numbered homonyms (*cum2*), and
``[NUM]`` for numbers. Its 9-character Perseus tags are converted to UD (:func:`perseus_to_ud`), following the
conventions of most UD Latin treebanks (see ``FEATURES``). Perseus has no PROPN, AUX, DET, PART or SCONJ, and no UD
equivalent for the deponent voice, so these POS tags and that feature cannot be matched against UD gold data.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Doc
from spacy.vocab import Vocab

from seflag.backends.base import Backend
from seflag.enums import FallbackLemma, Language, Layer
from seflag.errors import ConfigError, MissingDependencyError, SeflagError

BRIDGE: Path = Path(__file__).with_name("lamon_bridge.py")
NUMBER_LEMMA: str = "[NUM]"

# Perseus positional tags: 1 part of speech, 2 person, 3 number, 4 tense, 5 mood, 6 voice, 7 gender, 8 case,
# 9 degree; '-' = not applicable
UPOS: dict[str, str] = {"n": "NOUN", "v": "VERB", "a": "ADJ", "d": "ADV", "c": "CCONJ", "r": "ADP", "p": "PRON",
                        "m": "NUM", "i": "INTJ", "e": "INTJ", "u": "PUNCT"}
# The UD features follow the harmonized conventions of the UD Latin treebanks (CIRCSE, ITTB, LLCT, Perseus, UDante):
# every finite form has an aspect (present system Imp, perfect system Perf), non-finite forms have an aspect but
# no tense, the positive degree is unmarked and the superlative is 'Abs'. UD PROIEL differs (no aspect in the
# present system, tense on participles, Degree=Pos and Sup).
FEATURES: list[dict[str, dict[str, str]]] = [
    {},  # part of speech
    {p: {"Person": p} for p in "123"},
    {"s": {"Number": "Sing"}, "p": {"Number": "Plur"}},
    {"p": {"Tense": "Pres", "Aspect": "Imp"}, "i": {"Tense": "Past", "Aspect": "Imp"},
     "f": {"Tense": "Fut", "Aspect": "Imp"}, "r": {"Tense": "Past", "Aspect": "Perf"},
     "l": {"Tense": "Pqp", "Aspect": "Perf"}, "t": {"Tense": "Fut", "Aspect": "Perf"}},
    {"i": {"VerbForm": "Fin", "Mood": "Ind"}, "s": {"VerbForm": "Fin", "Mood": "Sub"},
     "m": {"VerbForm": "Fin", "Mood": "Imp"}, "n": {"VerbForm": "Inf"}, "p": {"VerbForm": "Part"},
     "d": {"VerbForm": "Ger"}, "g": {"VerbForm": "Gdv"}},
    {"a": {"Voice": "Act"}, "p": {"Voice": "Pass"}},  # 'd' (deponent) has no UD value
    {"m": {"Gender": "Masc"}, "f": {"Gender": "Fem"}, "n": {"Gender": "Neut"}},
    {"n": {"Case": "Nom"}, "g": {"Case": "Gen"}, "d": {"Case": "Dat"}, "a": {"Case": "Acc"}, "v": {"Case": "Voc"},
     "b": {"Case": "Abl"}, "l": {"Case": "Loc"}},
    {"c": {"Degree": "Cmp"}, "s": {"Degree": "Abs"}},  # positive: unmarked
]
# infinitives and participles: aspect only, from the Perseus tense (future participle: prospective)
NON_FINITE_ASPECT: dict[str, str] = {"p": "Imp", "i": "Imp", "r": "Perf", "l": "Perf", "t": "Perf", "f": "Prosp"}
FEATURE_NAMES: set[str] = {name for position in FEATURES for values in position.values() for name in values}
_WHITESPACE = re.compile(r"\s")


def perseus_to_ud(tag: str, lemma: str, form: str) -> tuple[str, str]:
    """A Perseus tag as UD (UPOS, FEATS), e.g. 'v3spia---' -> ('VERB', 'Mood=Ind|Number=Sing|Person=3|...')."""
    pos: str = tag[:1]
    if pos in UPOS:
        upos: str = UPOS[pos]
    elif lemma == NUMBER_LEMMA:  # Lamon tags numbers (26, XII) and punctuation with '-' only
        upos = "NUM"
    elif not any(c.isalnum() for c in form):
        upos = "PUNCT"
    else:
        upos = "X"
    features: dict[str, str] = {}
    for position, value in zip(FEATURES[1:], tag[1:], strict=False):
        features.update(position.get(value, {}))
    if features.get("VerbForm") in ("Inf", "Part"):
        features.pop("Tense", None)
        features.pop("Aspect", None)
        if tag[3:4] in NON_FINITE_ASPECT:
            features["Aspect"] = NON_FINITE_ASPECT[tag[3]]
    elif features.get("VerbForm") in ("Ger", "Gdv"):  # no tense or aspect in UD Latin
        features.pop("Tense", None)
        features.pop("Aspect", None)
    return upos, "|".join(f"{name}={value}" for name, value in sorted(features.items()))


class LamonBackend(Backend):
    """Latin lemmas, POS tags and morphological features from Lamon, run in its own Python environment."""

    description = "Lamon (lamonpy) Latin lemmatizer and tagger, run in a separate Python 3.8 environment via uv."

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        requirement: str = Field("lamonpy==0.2.0", description="lamonpy version that uv installs.")
        python: str = Field("3.8", description="Python version of the environment that uv creates.")
        command: list[str] | None = Field(
            None, min_length=1,
            description="Command that starts a Python interpreter with lamonpy installed, e.g. [conda, run, -n, "
                        "lamon, python]; replaces uv, 'python' and 'requirement'.")
        dict_path: str | None = Field(
            None, description="Lamon dictionary file (e.g. dict.large.bin of the large model); default: built in.")
        tagger_path: str | None = Field(
            None, description="Lamon tagger file (e.g. tagger.large.bin of the large model); default: built in.")
        beam_size: int | None = Field(None, ge=1, description="Beam size of the tagger; default: lamonpy's.")
        workers: int = Field(1, ge=1, description="Threads that tag sentences in parallel.")
        numbers: FallbackLemma = Field(
            FallbackLemma.FORM,
            description=f"Lemma for tokens that Lamon marks as numbers ({NUMBER_LEMMA}, e.g. 26 or XII): the token "
                        f"itself (as in UD) or empty.")
        timeout: int = Field(3600, ge=1, description="Seconds for the Lamon process, including the first-time setup.")

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        if evaluation.language != Language.LATIN:
            raise ConfigError(f"Lamon only supports Latin ('la'), not '{evaluation.language}'.")
        self.stats: dict[str, int] = {"tokens": 0, "unanalysed_tokens": 0, "split_tokens": 0}
        self.versions: dict[str, str | None] = {}

    def command(self) -> list[str]:
        """The command that runs the bridge script in lamonpy's environment."""
        if self.config.command is not None:
            return [*self.config.command, str(BRIDGE)]
        uv: str | None = shutil.which("uv")
        if uv is None:
            raise MissingDependencyError(
                "The Lamon backend runs lamonpy in a Python 3.8 environment created by uv, which is not installed. "
                "Install it (https://docs.astral.sh/uv/getting-started/installation/), or set 'command' to a "
                "Python interpreter with lamonpy installed.")
        return [uv, "run", "--no-project", "--isolated", "--python", self.config.python,
                "--with", self.config.requirement, "python", str(BRIDGE)]

    def _model_path(self, path: str | None) -> str | None:
        return str(self.evaluation.resolve_path(path)) if path else None

    def _tag(self, texts: list[str]) -> list[list[list]]:
        request: dict[str, Any] = {"texts": texts, "dict_path": self._model_path(self.config.dict_path),
                                   "tagger_path": self._model_path(self.config.tagger_path),
                                   "beam_size": self.config.beam_size, "workers": self.config.workers}
        command: list[str] = self.command()
        try:
            done = subprocess.run(command, input=json.dumps(request).encode("utf-8"), capture_output=True,
                                  env=os.environ, timeout=self.config.timeout, check=False)
        except FileNotFoundError as e:
            raise MissingDependencyError(f"Could not start Lamon with {command}: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise SeflagError(f"Lamon did not finish within {self.config.timeout} seconds; increase "
                              f"'model.timeout' (the first run also downloads Python and lamonpy).") from e
        if done.returncode != 0:
            stderr: str = done.stderr.decode("utf-8", errors="replace").strip()
            raise SeflagError(f"Lamon failed (exit code {done.returncode}): {stderr[-1500:]}")
        response: dict[str, Any] = json.loads(done.stdout.decode("utf-8"))
        self.versions = {"lamonpy": response.get("lamonpy"), "python": response.get("python")}
        results: list[list[list]] = response["results"]
        if len(results) != len(texts):
            raise SeflagError(f"Lamon returned {len(results)} results for {len(texts)} sentences.")
        return results

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        texts: list[str] = []
        starts: list[list[int]] = []
        for words in sentences:
            # whitespace inside a gold token would make Lamon split it; '_' keeps the character offsets
            texts.append(" ".join(_WHITESPACE.sub("_", w) for w in words))
            offsets: list[int] = []
            position: int = 0
            for w in words:
                offsets.append(position)
                position += len(w) + 1
            starts.append(offsets)
        vocab = Vocab()
        docs: list[Doc] = []
        for words, offsets, tagged in zip(sentences, starts, self._tag(texts) if texts else [], strict=True):
            analyses: list[list | None] = [None] * len(words)
            parts: list[int] = [0] * len(words)
            for start, _end, lemma, tag in tagged:
                i: int = bisect.bisect_right(offsets, start) - 1
                if i < 0 or start >= offsets[i] + len(words[i]):
                    continue  # a token of Lamon's on a separating space, which does not happen in practice
                parts[i] += 1
                if analyses[i] is None:
                    analyses[i] = [lemma, tag]
            lemmas: list[str] = []
            pos: list[str] = []
            morphs: list[str] = []
            for word, analysis, n in zip(words, analyses, parts, strict=True):
                self.stats["tokens"] += 1
                self.stats["split_tokens"] += n > 1
                if analysis is None:
                    self.stats["unanalysed_tokens"] += 1
                    lemmas.append("")
                    pos.append("X")
                    morphs.append("")
                    continue
                lemma, tag = analysis
                upos, feats = perseus_to_ud(tag, lemma, word)
                if lemma == NUMBER_LEMMA:
                    lemma = word if self.config.numbers == FallbackLemma.FORM else ""
                elif not any(c.isalnum() for c in word):  # punctuation, which Lamon may split ('...' -> 3 x '.')
                    lemma = word
                lemmas.append(lemma)
                pos.append(upos)
                morphs.append(feats)
            docs.append(Doc(vocab, words=words, lemmas=lemmas, pos=pos, morphs=morphs))
        return docs

    def label_inventory(self, layer: Layer) -> set[str] | None:
        if layer == Layer.UPOS:
            return {*UPOS.values(), "X"}
        if layer == Layer.MORPH:
            return set(FEATURE_NAMES)
        return None

    def metadata(self) -> dict[str, Any]:
        info: dict[str, Any] = {"name": "lamon", **self.versions, "beam_size": self.config.beam_size,
                                "dict_path": self._model_path(self.config.dict_path),
                                "tagger_path": self._model_path(self.config.tagger_path), **self.stats}
        if self.config.command is not None:
            info["command"] = self.config.command
        else:
            info.update(requirement=self.config.requirement, environment_python=self.config.python)
        return info
