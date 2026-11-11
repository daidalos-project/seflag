"""Conversion of local datasets into seflag's Parquet layout for the Hugging Face Hub.

The conversion is lossless (comments, multiword tokens, empty nodes, DEPS and MISC are kept) and is verified
by converting the Parquet files back to CoNLL-U. Hub repositories may also simply contain CoNLL-U files
(seflag reads them directly); Parquet adds the dataset viewer and fast loading via ``datasets``. Example::

    seflag convert-conllu data/ud/*.conllu --output build/ud_latin
    # inspect build/ud_latin, add a dataset card (README.md), then upload, e.g. with
    # huggingface-cli upload daidalos-project/<name> build/ud_latin --repo-type dataset
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from seflag.data.conllu import UD_SPLIT_PATTERN, read_conllu_files, record_to_conllu, split_from_filename
from seflag.data.huggingface import _require_datasets
from seflag.data.records import SentenceRecord

__all__ = ["conllu_to_dataset_dict", "features", "split_from_filename", "subset_from_filename",
           "verify_roundtrip", "write_parquet"]


def subset_from_filename(path: Path) -> str:
    """'la_proiel-ud-test.conllu' -> 'la_proiel'."""
    return re.sub(r"-ud$", "", UD_SPLIT_PATTERN.sub("", path.stem))


def features():
    """Arrow schema: one row per sentence; tokens, multiword tokens and empty nodes are lists of structs."""
    datasets = _require_datasets()
    string, int32 = datasets.Value("string"), datasets.Value("int32")
    token = {"form": string, "lemma": string, "upos": string, "xpos": string, "feats": string, "head": int32,
             "deprel": string, "deps": string, "misc": string}
    multiword_token = {"first": int32, "last": int32, "form": string, "misc": string}
    empty_node = {"id": string, "form": string, "lemma": string, "upos": string, "xpos": string, "feats": string,
                  "deps": string, "misc": string}
    # Python lists (not datasets.Sequence) keep these as lists of structs
    return datasets.Features({
        "sent_id": string, "text": string, "subset": string, "tokens": [token],
        "multiword_tokens": [multiword_token], "empty_nodes": [empty_node], "comments": [string],
        "entities": [{"first": int32, "last": int32, "label": string}],  # null = not annotated
    })


def conllu_to_dataset_dict(paths: list[Path], default_split: str = "test"):
    """Reads CoNLL-U files into a ``datasets.DatasetDict`` (splits inferred from UD file names).

    Files are processed in sorted order; rows keep the sentence order of the files.
    """
    datasets = _require_datasets()
    by_split: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(paths):
        split: str = split_from_filename(path, default_split)
        for record in read_conllu_files([path], subset_from_file=False):
            record.subset = subset_from_filename(path)
            by_split[split].append(record.to_dict())
    return datasets.DatasetDict({
        split: datasets.Dataset.from_list(rows, features=features()) for split, rows in by_split.items()})


def records_to_dataset(records: list[SentenceRecord]):
    """Builds a ``datasets.Dataset`` in seflag's Parquet layout from any records."""
    return _require_datasets().Dataset.from_list([r.to_dict() for r in records], features=features())


def verify_records(records: list[SentenceRecord], parquet_file: Path) -> list[str]:
    """Checks that a Parquet file read back with pyarrow yields exactly the given records."""
    import pyarrow.parquet as pq

    rows: list[dict] = pq.read_table(parquet_file).to_pylist()
    problems: list[str] = []
    if len(rows) != len(records):
        problems.append(f"{len(rows)} rows in Parquet, {len(records)} records")
    for i, (row, record) in enumerate(zip(rows, records)):
        if SentenceRecord.from_dict(row) != record:
            problems.append(f"row {i} ({record.sent_id!r}) differs from the source record")
            if len(problems) >= 10:
                break
    return problems


def write_parquet(dataset_dict, output_dir: Path) -> list[Path]:
    """Writes one parquet file per split into ``output_dir/data/``, the layout the Hub detects automatically."""
    data_dir: Path = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split, ds in dataset_dict.items():
        target: Path = data_dir / f"{split}.parquet"
        ds.to_parquet(str(target))
        written.append(target)
    return written


def verify_roundtrip(paths: list[Path], parquet_files: dict[str, Path], default_split: str = "test") -> list[str]:
    """Checks that the Parquet files reproduce the CoNLL-U sources byte for byte; returns a list of problems.

    The Parquet files are read with pyarrow directly, independently of the ``datasets`` loading logic.
    """
    import pyarrow.parquet as pq

    problems: list[str] = []
    sources: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(paths):
        sources[split_from_filename(path, default_split)].append(path)
    for split, files in sources.items():
        rows: list[dict] = pq.read_table(parquet_files[split]).to_pylist()
        offset: int = 0
        for path in files:
            expected: str = path.read_text(encoding="utf-8")
            n_sentences: int = sum(1 for _ in read_conllu_files([path]))
            records = [SentenceRecord.from_dict(r) for r in rows[offset:offset + n_sentences]]
            offset += n_sentences
            actual: str = "".join(record_to_conllu(r) for r in records)
            if actual != expected:
                exp_lines, act_lines = expected.split("\n"), actual.split("\n")
                line = next((i for i, (a, b) in enumerate(zip(exp_lines, act_lines)) if a != b),
                            min(len(exp_lines), len(act_lines)))
                problems.append(f"{path.name}: differs from line {line + 1}: "
                                f"{exp_lines[line] if line < len(exp_lines) else '<end>'!r} vs. "
                                f"{act_lines[line] if line < len(act_lines) else '<end>'!r}")
        if offset != len(rows):
            problems.append(f"split {split}: {len(rows)} rows in Parquet, {offset} sentences in sources")
    return problems
