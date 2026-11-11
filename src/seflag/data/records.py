"""The interchange format between dataset adapters and the evaluation core.

Every dataset adapter yields one :class:`SentenceRecord` per sentence, holding one :class:`TokenRecord`
per syntactic word (CoNLL-U semantics). Records hold everything a CoNLL-U sentence contains (comments,
multiword tokens, empty nodes, DEPS, MISC), so CoNLL-U -> records -> CoNLL-U is lossless.
The same structure, serialized with :meth:`SentenceRecord.to_dict`, is the layout of the Parquet datasets
created by :mod:`seflag.data.convert`. Records are turned into spaCy ``Doc`` objects, which are SEFLAG's
internal data model.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from spacy.parts_of_speech import IDS as SPACY_POS_IDS
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from seflag.errors import DataError


@dataclass(slots=True)
class TokenRecord:
    """One syntactic word with its gold annotations; ``None`` = not annotated (``_`` in CoNLL-U)."""

    form: str
    lemma: str | None = None
    upos: str | None = None
    xpos: str | None = None
    feats: str | None = None  # CoNLL-U FEATS string, e.g. "Case=Nom|Number=Sing"
    head: int | None = None  # CoNLL-U convention: 1-based, 0 = root
    deprel: str | None = None
    deps: str | None = None  # enhanced dependencies, CoNLL-U DEPS string
    misc: str | None = None  # CoNLL-U MISC string, e.g. "SpaceAfter=No|Gloss=and"


@dataclass(slots=True)
class MultiwordToken:
    """A surface token spanning several syntactic words, e.g. crasis κἀγώ = καί + ἐγώ (CoNLL-U line ``9-10``)."""

    first: int  # 1-based index of the first word
    last: int  # 1-based index of the last word (inclusive)
    form: str
    misc: str | None = None


@dataclass(slots=True)
class EmptyNode:
    """An empty node of the enhanced dependency graph (CoNLL-U line ``8.1``)."""

    id: str  # e.g. "8.1": inserted after word 8
    form: str | None = None
    lemma: str | None = None
    upos: str | None = None
    xpos: str | None = None
    feats: str | None = None
    deps: str | None = None
    misc: str | None = None


@dataclass(slots=True)
class EntitySpan:
    """A named entity spanning words ``first`` to ``last`` (1-based, inclusive), e.g. a person name."""

    first: int
    last: int
    label: str


TOKEN_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TokenRecord))
_COMMENT = re.compile(r"^#\s*(?P<key>[^=]*?)\s*(?:=\s?(?P<value>.*))?$")


def parse_comment(line: str) -> tuple[str, str | None]:
    """'# sent_id = 12' -> ('sent_id', '12'); '# newpar' -> ('newpar', None)."""
    match = _COMMENT.match(line)
    if not match:
        return line.lstrip("#").strip(), None
    value: str | None = match.group("value")
    return match.group("key"), value.strip() if value is not None else None


@dataclass(slots=True)
class SentenceRecord:
    tokens: list[TokenRecord]
    sent_id: str | None = None
    text: str | None = None
    subset: str | None = None  # e.g. the source treebank; results are additionally reported per subset
    comments: list[str] = field(default_factory=list)  # raw comment lines in original order, incl. sent_id/text
    multiword_tokens: list[MultiwordToken] = field(default_factory=list)
    empty_nodes: list[EmptyNode] = field(default_factory=list)
    entities: list[EntitySpan] | None = None  # None = not annotated for named entities; [] = no entities

    @property
    def metadata(self) -> dict[str, str | None]:
        """Comments as a key -> value mapping (``# newpar`` -> ``{"newpar": None}``)."""
        return dict(parse_comment(c) for c in self.comments)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SentenceRecord:
        def build(item_cls, items):
            return [i if isinstance(i, item_cls) else item_cls(**i) for i in (items or [])]

        return cls(tokens=build(TokenRecord, data["tokens"]), sent_id=data.get("sent_id"), text=data.get("text"),
                   subset=data.get("subset"), comments=list(data.get("comments") or []),
                   multiword_tokens=build(MultiwordToken, data.get("multiword_tokens")),
                   empty_nodes=build(EmptyNode, data.get("empty_nodes")),
                   entities=None if data.get("entities") is None else build(EntitySpan, data["entities"]))


# annotation layers a task may require, mapped to the TokenRecord fields they are read from
LAYERS: dict[str, tuple[str, ...]] = {
    "lemma": ("lemma",),
    "upos": ("upos",),
    "xpos": ("xpos",),
    "morph": ("feats",),
    "dependencies": ("head", "deprel"),
    "entities": (),  # sentence-level: SentenceRecord.entities
}


def record_to_doc(vocab: Vocab, record: SentenceRecord, layers: set[str]) -> Doc:
    """Builds a gold-standard ``Doc`` holding the requested annotation layers.

    Missing annotations (``None``) are stored as empty strings; tasks exclude such tokens from scoring.
    """
    unknown: set[str] = layers - LAYERS.keys()
    if unknown:
        raise ValueError(f"Unknown annotation layer(s) {unknown}")
    if not record.tokens:
        raise DataError(f"Sentence {record.sent_id!r} has no tokens.")
    tokens: list[TokenRecord] = record.tokens
    kwargs: dict[str, Any] = {}
    if "lemma" in layers:
        kwargs["lemmas"] = [t.lemma or "" for t in tokens]
    if "upos" in layers:
        upos: list[str] = [t.upos or "" for t in tokens]
        invalid: set[str] = {v for v in upos if v not in SPACY_POS_IDS}
        if invalid:
            raise DataError(f"Sentence {record.sent_id!r}: invalid UPOS tag(s) {sorted(invalid)}.")
        kwargs["pos"] = upos
    if "xpos" in layers:
        kwargs["tags"] = [t.xpos or "" for t in tokens]
    if "morph" in layers:
        kwargs["morphs"] = [t.feats or "" for t in tokens]
    if "dependencies" in layers:
        # spaCy convention: absolute token indices, the root points to itself. Words without a gold head
        # also point to themselves and get an empty label, so tasks can tell them apart and skip them.
        heads: list[int] = []
        deps: list[str] = []
        for i, t in enumerate(tokens):
            if t.head is None or t.deprel is None:
                heads.append(i)
                deps.append("")
            elif not 0 <= t.head <= len(tokens):
                raise DataError(f"Sentence {record.sent_id!r}: HEAD {t.head} of word {i + 1} is outside the sentence.")
            else:
                heads.append(i if t.head == 0 else t.head - 1)
                deps.append(t.deprel)
        kwargs["heads"] = heads
        kwargs["deps"] = deps
    doc = Doc(vocab, words=[t.form for t in tokens], **kwargs)
    if "entities" in layers:
        spans = []
        for e in sorted(record.entities or [], key=lambda e: e.first):
            if not 1 <= e.first <= e.last <= len(tokens):
                raise DataError(f"Sentence {record.sent_id!r}: entity {e} lies outside the sentence.")
            if spans and spans[-1].end > e.first - 1:
                raise DataError(f"Sentence {record.sent_id!r}: overlapping entities are not supported ({e}).")
            spans.append(Span(doc, e.first - 1, e.last, label=e.label))
        doc.ents = spans
        doc.user_data["entities_annotated"] = record.entities is not None
    doc.user_data["sent_id"] = record.sent_id
    doc.user_data["subset"] = record.subset
    doc.user_data["text"] = record.text
    return doc


def check_layers_present(records: list[SentenceRecord], layers: set[str]) -> None:
    """Raises if a required layer is not annotated for any token (e.g. a dataset without lemmas)."""
    for layer in sorted(layers):
        if layer == "entities":
            if not any(r.entities is not None for r in records):
                raise DataError("The dataset has no 'entities' annotations, which this task requires.")
            continue
        names: tuple[str, ...] = LAYERS[layer]
        if not any(getattr(t, n) is not None for r in records for t in r.tokens for n in names):
            raise DataError(f"The dataset has no '{layer}' annotations, which this task requires.")
