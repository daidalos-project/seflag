from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import Field

from seflag.data.base import DatasetAdapter
from seflag.data.files import FileResolver, FileSource, collect_files, file_hashes  # noqa: F401 (re-exported)
from seflag.data.records import EmptyNode, MultiwordToken, SentenceRecord, TokenRecord, parse_comment
from seflag.errors import DataError

logger = logging.getLogger("seflag")

UD_SPLIT_PATTERN = re.compile(r"-(train|dev|test)$")
_WORD_ID = re.compile(r"^[1-9]\d*$")
_RANGE_ID = re.compile(r"^([1-9]\d*)-([1-9]\d*)$")
_EMPTY_ID = re.compile(r"^(0|[1-9]\d*)\.[1-9]\d*$")


def split_from_filename(path: Path, default: str = "test") -> str:
    """Infers the split from UD-style file names, e.g. 'la_proiel-ud-dev.conllu' -> 'validation'."""
    match = UD_SPLIT_PATTERN.search(path.stem)
    if not match:
        return default
    return {"dev": "validation"}.get(match.group(1), match.group(1))


def _opt(value: str) -> str | None:
    return None if value == "_" else value


def _out(value: object) -> str:
    return "_" if value is None else str(value)


def lines_to_record(comments: list[str], rows: list[tuple[int, list[str]]], source: str,
                    subset: str | None = None) -> SentenceRecord:
    """Builds a record from the comment lines and the (line number, 10 columns) rows of one sentence."""
    record = SentenceRecord(tokens=[], subset=subset, comments=comments)
    for line_no, c in rows:
        where = f"{source}:{line_no}"
        if _WORD_ID.match(c[0]):
            if int(c[0]) != len(record.tokens) + 1:
                raise DataError(f"{where}: word ID {c[0]} out of sequence")
            try:
                head: int | None = None if c[6] == "_" else int(c[6])
            except ValueError as e:
                raise DataError(f"{where}: invalid HEAD {c[6]!r}") from e
            # a lemma '_' is only meaningful for the form '_'
            lemma: str | None = c[2] if c[1] == "_" and c[2] == "_" else _opt(c[2])
            record.tokens.append(TokenRecord(
                form=c[1], lemma=lemma, upos=_opt(c[3]), xpos=_opt(c[4]), feats=_opt(c[5]), head=head,
                deprel=_opt(c[7]), deps=_opt(c[8]), misc=_opt(c[9])))
        elif match := _RANGE_ID.match(c[0]):
            if any(v != "_" for v in c[2:9]):
                logger.warning("%s: annotations of multiword token %s other than FORM/MISC are dropped", where, c[0])
            record.multiword_tokens.append(
                MultiwordToken(first=int(match.group(1)), last=int(match.group(2)), form=c[1], misc=_opt(c[9])))
        elif _EMPTY_ID.match(c[0]):
            if c[6] != "_" or c[7] != "_":
                logger.warning("%s: HEAD/DEPREL of empty node %s are dropped", where, c[0])
            record.empty_nodes.append(EmptyNode(
                id=c[0], form=_opt(c[1]), lemma=_opt(c[2]), upos=_opt(c[3]), xpos=_opt(c[4]), feats=_opt(c[5]),
                deps=_opt(c[8]), misc=_opt(c[9])))
        else:
            raise DataError(f"{where}: invalid ID {c[0]!r}")
    for comment in comments:
        key, value = parse_comment(comment)
        if key == "sent_id" and record.sent_id is None:
            record.sent_id = value
        elif key == "text" and record.text is None:
            record.text = value
    return record


def parse_conllu_lines(lines: Iterable[str], source: str = "<string>",
                       subset: str | None = None) -> Iterator[SentenceRecord]:
    """Parses CoNLL-U lines (without or with trailing newlines) into records, keeping all information."""
    comments: list[str] = []
    rows: list[tuple[int, list[str]]] = []
    line_no: int = 0
    for line_no, line in enumerate(lines, start=1):
        line = line.rstrip("\r\n")
        if not line.strip():
            if rows:
                yield lines_to_record(comments, rows, source, subset)
            elif comments:
                raise DataError(f"{source}:{line_no}: comment lines without a sentence")
            comments, rows = [], []
        elif line.startswith("#"):
            if rows:
                raise DataError(f"{source}:{line_no}: comment line inside a sentence")
            comments.append(line)
        else:
            columns: list[str] = line.split("\t")
            if len(columns) != 10:
                raise DataError(f"{source}:{line_no}: expected 10 tab-separated columns, found {len(columns)}")
            rows.append((line_no, columns))
    if rows:
        yield lines_to_record(comments, rows, source, subset)
    elif comments:
        raise DataError(f"{source}:{line_no}: comment lines without a sentence")


def read_conllu_files(paths: list[Path], subset_from_file: bool = True) -> Iterator[SentenceRecord]:
    for path in paths:
        with path.open(encoding="utf-8") as f:
            yield from parse_conllu_lines(f, str(path), path.stem if subset_from_file else None)


def record_to_conllu(record: SentenceRecord) -> str:
    """Serializes a record as one CoNLL-U sentence block, including the terminating empty line."""
    ranges: dict[int, list[MultiwordToken]] = defaultdict(list)
    for mwt in record.multiword_tokens:
        ranges[mwt.first].append(mwt)
    empties: dict[int, list[EmptyNode]] = defaultdict(list)
    for node in record.empty_nodes:
        empties[int(node.id.split(".")[0])].append(node)

    def empty_lines(position: int) -> list[str]:
        return ["\t".join([n.id, _out(n.form), _out(n.lemma), _out(n.upos), _out(n.xpos), _out(n.feats), "_", "_",
                           _out(n.deps), _out(n.misc)]) for n in empties.get(position, [])]

    lines: list[str] = list(record.comments) + empty_lines(0)
    for i, t in enumerate(record.tokens, start=1):
        lines += ["\t".join([f"{m.first}-{m.last}", m.form, "_", "_", "_", "_", "_", "_", "_", _out(m.misc)])
                  for m in ranges.get(i, [])]
        lines.append("\t".join([str(i), t.form, _out(t.lemma), _out(t.upos), _out(t.xpos), _out(t.feats),
                                _out(t.head), _out(t.deprel), _out(t.deps), _out(t.misc)]))
        lines += empty_lines(i)
    return "\n".join(lines) + "\n\n"


def write_conllu(records: Iterable[SentenceRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(record_to_conllu(record))


class ConlluDataset(DatasetAdapter):
    """CoNLL-U files (e.g. Universal Dependencies treebanks), local or in a Hugging Face Hub repository."""

    description = "CoNLL-U file(s), local or in a Hub repository (repo: ...), e.g. Universal Dependencies treebanks."

    class Config(FileSource):
        pattern: str = Field("*.conllu", description="Glob pattern for files inside directories.")
        subset_by_file: bool = Field(True, description="Report results per file in addition to the total.")

    def __init__(self, config: Any, evaluation):
        super().__init__(config, evaluation)
        self.resolver = FileResolver(config, evaluation)

    def files(self) -> list[Path]:
        return self.resolver.files()

    def records(self) -> Iterator[SentenceRecord]:
        return read_conllu_files(self.files(), self.config.subset_by_file)

    def fingerprint(self) -> dict[str, Any]:
        return self.resolver.fingerprint()
