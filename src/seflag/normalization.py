"""Named string normalizers that can be referenced from configuration files.

Normalization is a frequent source of hidden differences between published evaluation results
(e.g. u/v spelling in Latin, Unicode composition in Greek). In SEFLAG, every normalization step
is explicit and recorded in the results.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

Normalizer = Callable[[str], str]


def _strip_diacritics(text: str) -> str:
    decomposed: str = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


# characters used to mark elision in Greek texts, e.g. δ’ (U+2019), ἀλλ̓ (U+0313, UD Perseus), ἐπ᾿ (U+1FBF, UD PTNK),
# δ᾽ (U+1FBD), δʼ (U+02BC), δ' (U+0027); see ENCODING.md and elision_position()
ELISION_MARKS: str = "\u2019\u1fbd\u1fbf\u02bc\u0027\u0313"
_GREEK_LETTER = r"\u0370-\u03ff\u1f00-\u1fff"
_SPACING_ELISION = re.compile(f"(?<=[{_GREEK_LETTER}\u0300-\u036f])[\u2019\u1fbd\u1fbf\u02bc\u0027]$")
_FINAL_SIGMA = re.compile(f"(?<=[{_GREEK_LETTER}\u0300-\u036f])σ$")
_LATIN_LETTERS = re.compile(r"[A-Za-z]")


def elision_position(text: str) -> int | None:
    """Index of the elision mark of an elided Greek word (e.g. δ’, δι’, ἀλλ̓), or None.

    Only a mark at the end of a word directly after a Greek letter counts, so Latin apostrophes ("l'") and marks
    standing alone (quotation marks) are left alone. U+0313 only counts if NFC cannot compose it with the letter
    before: otherwise it is a smooth breathing in decomposed text (NFD 'οὐ'). A final ῤ (U+1FE4) after another
    letter is an elided ρ whose mark U+0313 NFC has merged into it (UD Perseus 'παῤ' for παρά).
    """
    if len(text) < 2:
        return None
    if _SPACING_ELISION.search(text):
        return len(text) - 1
    if text[-1] == "\u0313" and unicodedata.normalize("NFC", text)[-1] == "\u0313" and \
            re.search(f"[{_GREEK_LETTER}\u0300-\u036f]", text[-2]):
        return len(text) - 1
    if text[-1] == "\u1fe4" and text[-2].isalpha():
        return len(text) - 1
    return None


def elision_to(mark: str) -> Normalizer:
    """Normalizer that writes the elision mark of every elided Greek word as ``mark``."""
    def normalize(text: str) -> str:
        position: int | None = elision_position(text)
        if position is None:
            return text
        return text[:position] + ("ρ" if text[position] == "\u1fe4" else "") + mark

    return normalize


def _final_sigma(text: str) -> str:
    return _FINAL_SIGMA.sub("ς", text)


def _lunate_sigma(text: str) -> str:
    """ϲ/Ϲ -> σ/ς/Σ; also a Latin C/c used as lunate sigma inside an otherwise Greek word (e.g. 'διόC')."""
    greek_word: bool = bool(re.search(f"[{_GREEK_LETTER}]", text)) and \
        not _LATIN_LETTERS.search(text.replace("C", "").replace("c", ""))
    letters: list[int] = [i for i, ch in enumerate(text) if ch.isalpha()]
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == "\u03f9" or (greek_word and ch == "C" and i == letters[0] and len(letters) > 1):
            out.append("Σ")
        elif ch in "\u03f2" or (greek_word and ch in "Cc"):
            out.append("ς" if i == letters[-1] and len(letters) > 1 else "σ")
        else:
            out.append(ch)
    return "".join(out)


NORMALIZERS: dict[str, tuple[Normalizer, str]] = {
    "nfc": (lambda s: unicodedata.normalize("NFC", s), "Unicode canonical composition (NFC)"),
    "nfd": (lambda s: unicodedata.normalize("NFD", s), "Unicode canonical decomposition (NFD)"),
    "nfkc": (lambda s: unicodedata.normalize("NFKC", s), "Unicode compatibility composition (NFKC)"),
    "lowercase": (str.lower, "Lowercase all characters"),
    "v_to_u": (lambda s: s.replace("v", "u").replace("V", "U"), "Latin: replace v/V with u/U"),
    "j_to_i": (lambda s: s.replace("j", "i").replace("J", "I"), "Latin: replace j/J with i/I"),
    "strip_diacritics": (_strip_diacritics, "Remove combining marks (accents, breathings, macrons)"),
    "elision_to_u02bc": (elision_to("\u02bc"), "Greek: elision marks of elided words -> U+02BC MODIFIER LETTER "
                                                "APOSTROPHE (what greCy expects)"),
    "elision_to_u2019": (elision_to("\u2019"), "Greek: elision marks of elided words -> U+2019 RIGHT SINGLE "
                                                "QUOTATION MARK (Morpheus, Ancient-Greek-BERT)"),
    "elision_to_apostrophe": (elision_to("'"), "Greek: elision marks of elided words -> U+0027 APOSTROPHE "
                                               "(Greek fastText vectors)"),
    "final_sigma": (_final_sigma, "Greek: σ at the end of a word -> ς"),
    "lunate_sigma": (_lunate_sigma,
                     "Greek: lunate sigma \u03f2/\u03f9 (and Latin C/c used as one in Greek words) -> σ/ς/Σ"),
    "strip_homonym_digits": (
        lambda s: re.sub(r"(?<=\D)#?\d+$", "", s),
        "Remove trailing homonym numbers from lemmas, e.g. 'sum1' or 'sum#1' (Morpheus) -> 'sum'",
    ),
}


def get_pipeline(names: list[str]) -> Normalizer:
    """Composes the named normalizers (applied in the given order) into a single function."""
    unknown: list[str] = [n for n in names if n not in NORMALIZERS]
    if unknown:
        raise ValueError(f"Unknown normalizer(s) {unknown}. Available: {', '.join(NORMALIZERS)}")
    functions: list[Normalizer] = [NORMALIZERS[n][0] for n in names]

    def pipeline(text: str) -> str:
        for fn in functions:
            text = fn(text)
        return text

    return pipeline
