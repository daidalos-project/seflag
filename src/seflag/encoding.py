"""Character-level inspection: how the strings of a dataset (or an embedding vocabulary) are spelled in Unicode.

All resources SEFLAG uses are UTF-8 and mostly NFC, but they differ in characters that NFC does not unify: elision
marks, sigma variants, u/v and i/j, case (see ENCODING.md). ``seflag inspect-encoding`` reports these properties so
that datasheets and model cards can state them from measurements and update them when a resource changes::

    seflag inspect-encoding examples/grecy_lemma_ud.yaml --by-subset
    seflag inspect-encoding examples/agree_similarity_grc_fasttext.yaml --model --json
"""

from __future__ import annotations

import unicodedata as ud
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from seflag.normalization import ELISION_MARKS, elision_position

GREEK_PUNCTUATION: str = "··;;‧⋅⸱"
INVISIBLE: str = " ​‌‍﻿­\t  "
APOSTROPHE_LIKE: str = ELISION_MARKS + "‘ʽ`´ʹ′῾"
MAX_EXAMPLES: int = 5

# string-level properties: key -> description, in report order
FEATURES: dict[str, str] = {
    "capitalized": "starts with an upper-case letter",
    "mixed_greek_latin": "contains both Greek and Latin letters (homoglyphs, e.g. Latin C as sigma)",
    "other_script": "contains letters of another script (e.g. Cyrillic)",
    "elided": "elided Greek word (elision mark at the end, incl. ῤ for ρ + U+0313)",
    "medial_sigma_at_end": "Greek word ending in σ instead of ς",
    "final_sigma_inside": "ς inside a word",
    "lunate_sigma": "contains lunate sigma ϲ/Ϲ",
    "iota_subscript": "contains iota subscript (ypogegrammeni)",
    "prosgegrammeni": "contains U+1FBE GREEK PROSGEGRAMMENI (NFC turns it into ι)",
    "grave_accent": "contains a grave accent",
    "unaccented_greek": "Greek word of 2+ letters without acute, grave or circumflex",
    "v": "contains v or V",
    "j": "contains j or J",
    "macron_or_breve": "contains a macron or breve",
    "ligature": "contains æ, œ, Æ or Œ",
    "letters_and_digits": "contains letters and digits (e.g. homonym numbers)",
}
# character-level inventories: key -> description
CHARACTERS: dict[str, str] = {
    "changed_by_nfc": "characters that NFC replaces (e.g. oxia -> tonos)",
    "uncomposed_combining": "combining marks left after NFC (no precomposed character exists)",
    "apostrophe_like": "apostrophe-like characters (elision marks, breathings written as spacing characters)",
    "greek_punctuation": "middle dot, ano teleia, semicolon, Greek question mark",
    "invisible": "spaces, tabs, zero-width characters, soft hyphens",
    "other_non_ascii": "other non-ASCII characters outside the Greek blocks",
}


def _script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    name: str = ud.name(ch, "")
    if name.startswith(("GREEK", "COPTIC")):
        return "Greek"
    return "Latin" if name.startswith("LATIN") else "other"


def normalization_class(text: str) -> str:
    """ascii, nfc, nfd, mixed (neither NFC nor NFD), or no_decomposable (non-ASCII, but NFC == NFD)."""
    if text.isascii():
        return "ascii"
    nfc, nfd = ud.normalize("NFC", text), ud.normalize("NFD", text)
    if nfc == nfd:
        return "no_decomposable"
    return "nfc" if text == nfc else "nfd" if text == nfd else "mixed"


def char_label(ch: str) -> str:
    return f"U+{ord(ch):04X} {ud.name(ch, repr(ch))}"


@dataclass
class EncodingStats:
    """Counts over the analysed strings (occurrences, not types), with examples."""

    strings: int = 0
    empty: int = 0
    normalization: Counter = field(default_factory=Counter)
    features: Counter = field(default_factory=Counter)
    characters: dict[str, Counter] = field(default_factory=lambda: {k: Counter() for k in CHARACTERS})
    examples: dict[str, list[str]] = field(default_factory=dict)

    def _example(self, key: str, text: str) -> None:
        found: list[str] = self.examples.setdefault(key, [])
        if len(found) < MAX_EXAMPLES and text not in found:
            found.append(text)

    def add(self, text: str | None) -> None:
        if not text:
            self.empty += 1
            return
        self.strings += 1
        cls: str = normalization_class(text)
        self.normalization[cls] += 1
        if cls in ("nfd", "mixed"):
            self._example(f"normalization:{cls}", text)
        nfc: str = ud.normalize("NFC", text)
        nfd: str = ud.normalize("NFD", text)
        for ch in text:
            category: str | None = None
            if not ud.combining(ch) and ud.normalize("NFC", ch) != ch:
                category = "changed_by_nfc"
            elif ch in APOSTROPHE_LIKE:
                category = "apostrophe_like"
            elif ch in GREEK_PUNCTUATION:
                category = "greek_punctuation"
            elif ch in INVISIBLE:
                category = "invisible"
            elif not ch.isascii() and not ud.combining(ch) and _script(ud.normalize("NFD", ch)[0]) != "Greek":
                category = "other_non_ascii"
            if category:
                self.characters[category][ch] += 1
                self._example(f"{category}:{ch}", text)
        for ch in nfc:
            if ud.combining(ch) and ch not in APOSTROPHE_LIKE:
                self.characters["uncomposed_combining"][ch] += 1
                self._example(f"uncomposed_combining:{ch}", text)
        letters: list[str] = [ch for ch in nfd if ch.isalpha()]
        scripts: set[str] = {s for s in map(_script, letters) if s}
        greek: bool = "Greek" in scripts
        flags: dict[str, bool] = {
            "capitalized": text[0].isupper(),
            "mixed_greek_latin": {"Greek", "Latin"} <= scripts,
            "other_script": "other" in scripts,
            "elided": elision_position(text) is not None,
            "medial_sigma_at_end": greek and len(letters) > 1 and text[-1] == "σ",
            "final_sigma_inside": "ς" in text[:-1] and any(c.isalpha() for c in text[text.index("ς") + 1:]),
            "lunate_sigma": "ϲ" in text or "Ϲ" in text,
            "iota_subscript": "ͅ" in nfd,
            "prosgegrammeni": "ι" in text,
            "grave_accent": greek and "̀" in nfd,
            "unaccented_greek": greek and len(letters) > 1 and not any(m in nfd for m in "̀́͂"),
            "v": "v" in text or "V" in text,
            "j": "j" in text or "J" in text,
            "macron_or_breve": "̄" in nfd or "̆" in nfd,
            "ligature": any(ch in text for ch in "æœÆŒ"),
            "letters_and_digits": bool(letters) and any(ch.isdigit() for ch in text),
        }
        for key, value in flags.items():
            if value:
                self.features[key] += 1
                self._example(key, text)

    def to_dict(self) -> dict[str, Any]:
        return {"strings": self.strings, "empty": self.empty, "normalization": dict(self.normalization),
                "features": dict(self.features),
                "characters": {k: {char_label(ch): n for ch, n in c.most_common()} for k, c in self.characters.items()
                               if c},
                "examples": self.examples}

    def format(self, indent: str = "  ") -> list[str]:
        """Human-readable lines; only properties that occur are listed."""
        lines: list[str] = []
        not_nfc: int = self.normalization["nfd"] + self.normalization["mixed"]
        verdict: str = "all NFC" if not not_nfc else f"{not_nfc} not NFC"
        parts: str = ", ".join(f"{k} {v}" for k, v in sorted(self.normalization.items()))
        lines.append(f"{indent}{self.strings} strings ({self.empty} empty); normalization: {parts} [{verdict}]")
        for cls in ("nfd", "mixed"):
            if f"normalization:{cls}" in self.examples:
                lines.append(f"{indent}  {cls} e.g. {', '.join(self.examples[f'normalization:{cls}'])}")
        for key, description in FEATURES.items():
            if self.features[key]:
                share: float = self.features[key] / self.strings
                lines.append(f"{indent}{key}: {self.features[key]} ({share:.1%}) - {description}; e.g. "
                             f"{', '.join(self.examples.get(key, []))}")
        for key, description in CHARACTERS.items():
            if self.characters[key]:
                lines.append(f"{indent}{key} - {description}:")
                for ch, n in self.characters[key].most_common(10):
                    lines.append(f"{indent}  {n:>7}  {char_label(ch)}; e.g. "
                                 f"{', '.join(self.examples.get(f'{key}:{ch}', []))}")
        return lines


def analyse(strings: Iterable[str | None]) -> EncodingStats:
    stats = EncodingStats()
    for text in strings:
        stats.add(text)
    return stats


def inspect_dataset(dataset: Any, by_subset: bool = False, max_sentences: int | None = None) -> dict[str, dict]:
    """Statistics per field ('form', 'lemma'; 'words' for lexical datasets), overall and optionally per subset.

    Returns {subset: {field: EncodingStats}}; the overall statistics are under the subset name ''.
    """
    from seflag.data.lexical import ChoiceItem, LexicalDataset, WordPair

    result: dict[str, dict[str, EncodingStats]] = {}

    def add(subset: str, name: str, value: str | None) -> None:
        for key in ("", subset) if by_subset and subset else ("",):
            result.setdefault(key, {}).setdefault(name, EncodingStats()).add(value)

    if isinstance(dataset, LexicalDataset):
        for item in dataset.items()[:max_sentences]:
            words: list[str] = [item.word1, item.word2] if isinstance(item, WordPair) else \
                [item.target, *item.choices] if isinstance(item, ChoiceItem) else []
            for word in words:
                add(item.subset or "", "words", word)
        return result
    for count, record in enumerate(dataset.records()):
        if max_sentences is not None and count >= max_sentences:
            break
        for token in record.tokens:
            add(record.subset or "", "form", token.form)
            add(record.subset or "", "lemma", token.lemma)
    # drop fields without any annotation (e.g. lemmas of an NER dataset)
    return {subset: {name: s for name, s in fields.items() if s.strings} for subset, fields in result.items()}
