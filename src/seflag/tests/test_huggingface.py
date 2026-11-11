import shutil
from pathlib import Path

import pytest

pytest.importorskip("datasets")

from seflag import load_config, run  # noqa: E402
from seflag.data.convert import (conllu_to_dataset_dict, split_from_filename, verify_roundtrip,  # noqa: E402
                                 write_parquet)
from seflag.data.records import SentenceRecord  # noqa: E402


def evaluate(backend, **dataset):
    return run(load_config({"language": "la", "task": "lemmatization", "model": {"backend": backend},
                            "output": {"dir": None}, "dataset": {"source": "huggingface", **dataset}}))


def test_split_from_filename(tmp_path):
    assert split_from_filename(tmp_path / "la_proiel-ud-dev.conllu") == "validation"
    assert split_from_filename(tmp_path / "anything.conllu") == "test"


def test_convert_and_evaluate_roundtrip(conllu_dir, fake_backend, tmp_path):
    dd = conllu_to_dataset_dict(sorted(conllu_dir.glob("*.conllu")))
    assert set(dd) == {"test", "validation"}
    row = dd["test"][0]
    assert row["subset"] == "la_test"
    assert row["tokens"][1] == {"form": "virum", "lemma": "vir", "upos": "NOUN", "xpos": None, "feats": "Case=Acc",
                                "head": 4, "deprel": "obj", "deps": None, "misc": None}
    assert row["multiword_tokens"] == [{"first": 2, "last": 3, "form": "virumque", "misc": "SpaceAfter=No"}]
    files = write_parquet(dd, tmp_path / "hub")
    assert verify_roundtrip(sorted(conllu_dir.glob("*.conllu")), {f.stem: f for f in files}) == []
    result = evaluate(fake_backend, repo="parquet", data_files={"test": str(files[0])})
    assert result.metrics["scored"] == 6
    assert result.metrics["accuracy"] == pytest.approx(4 / 6)


def test_parallel_columns_with_class_labels(fake_backend, tmp_path):
    import datasets

    upos = datasets.ClassLabel(names=["NOUN", "VERB", "_"])
    ds = datasets.Dataset.from_dict(
        {"words": [["Arma", "cano"]], "lemmas": [["arma", "_"]], "upos": [[0, 1]]},
        features=datasets.Features({"words": datasets.Sequence(datasets.Value("string")),
                                    "lemmas": datasets.Sequence(datasets.Value("string")),
                                    "upos": datasets.Sequence(upos)}))
    ds.to_parquet(str(tmp_path / "t.parquet"))
    result = evaluate(fake_backend, repo="parquet", data_files={"test": str(tmp_path / "t.parquet")},
                      columns={"form": "words"})
    assert result.metrics["scored"] == 1 and result.metrics["unannotated_in_gold"] == 1
    pos = run(load_config({"language": "la", "task": "pos_tagging", "model": {"backend": fake_backend},
                           "output": {"dir": None}, "dataset": {"source": "huggingface", "repo": "parquet",
                           "data_files": {"test": str(tmp_path / "t.parquet")}, "columns": {"form": "words"}}}))
    assert pos.metrics["confusion_matrix"] == {"NOUN": {"NOUN": 1}, "VERB": {"NOUN": 1}}


def test_conllu_files_in_hub_repo(conllu_dir, fake_backend, tmp_path, monkeypatch):
    import huggingface_hub

    snapshot = tmp_path / "snapshots" / "abc123"
    shutil.copytree(conllu_dir, snapshot)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda *a, **k: ["README.md", "x-ud-test.conllu"])
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snapshot))
    result = evaluate(fake_backend, repo="someone/treebanks")
    assert result.metrics["scored"] == 6  # only the test split file
    assert result.provenance["dataset"]["format"] == "conllu"
    assert result.provenance["dataset"]["resolved_revision"] == "abc123"
    assert list(result.provenance["dataset"]["files"]) == ["la_test-ud-test.conllu"]
    assert evaluate(fake_backend, repo="someone/treebanks", split="validation").metrics["scored"] == 6


def test_roundtrip_detects_differences(conllu_dir, tmp_path):
    source = conllu_dir / "la_test-ud-test.conllu"
    files = write_parquet(conllu_to_dataset_dict([source]), tmp_path / "hub")
    source.write_text(source.read_text(encoding="utf-8").replace("Gloss=of-Troy", "Gloss=of-Rome"), encoding="utf-8")
    problems = verify_roundtrip([source], {f.stem: f for f in files})
    assert len(problems) == 1 and "Gloss=of-Rome" in problems[0]


def test_seflag_layout_keeps_everything_when_loading(conllu_dir, tmp_path):
    from seflag.data.huggingface import HuggingFaceDataset

    files = write_parquet(conllu_to_dataset_dict(sorted(conllu_dir.glob("*.conllu"))), tmp_path / "hub")
    config = load_config({"language": "la", "task": "lemmatization", "model": {"backend": "spacy", "name": "x"},
                          "dataset": {"source": "huggingface", "repo": "parquet"}})
    ds = HuggingFaceDataset(HuggingFaceDataset.Config(repo="parquet", data_files={"test": str(files[0])}), config)
    second: SentenceRecord = list(ds.records())[1]
    assert second.comments[0] == "# source = Vergil, Aeneid 1.1"
    assert second.empty_nodes[0].id == "3.1"
    assert second.tokens[2].misc == "SpaceAfter=No"


REPO_DATA = sorted((p for p in (Path(__file__).parents[3] / "data").glob("**/*.conllu")), key=str) \
    if (Path(__file__).parents[3] / "data").is_dir() else []


@pytest.mark.skipif(not REPO_DATA, reason="repository data not available")
@pytest.mark.parametrize("source", REPO_DATA, ids=lambda p: p.name)
def test_roundtrip_on_repository_treebanks(source, tmp_path):
    files = write_parquet(conllu_to_dataset_dict([source]), tmp_path)
    assert verify_roundtrip([source], {f.stem: f for f in files}) == []


def test_ner_tag_column_and_export_roundtrip(fake_backend, tmp_path):
    import datasets

    from seflag.data.convert import records_to_dataset, verify_records
    from seflag.data.huggingface import HuggingFaceDataset

    tags = datasets.ClassLabel(names=["O", "B-PER", "I-PER", "B-LOC"])
    ds = datasets.Dataset.from_dict(
        {"tokens": [["Gaius", "Iulius", "Romam", "venit"]], "ner_tags": [[1, 2, 3, 0]]},
        features=datasets.Features({"tokens": datasets.Sequence(datasets.Value("string")),
                                    "ner_tags": datasets.Sequence(tags)}))
    ds.to_parquet(str(tmp_path / "ner.parquet"))
    config = load_config({"language": "la", "task": "ner", "model": {"backend": "spacy", "name": "x"},
                          "dataset": {"source": "huggingface", "repo": "parquet"}})
    adapter = HuggingFaceDataset(HuggingFaceDataset.Config(
        repo="parquet", data_files={"test": str(tmp_path / "ner.parquet")}), config)
    records = list(adapter.records())
    assert [(e.first, e.last, e.label) for e in records[0].entities] == [(1, 2, "PER"), (3, 3, "LOC")]
    # export to SEFLAG's layout and read back: entities survive, None (= not annotated) stays None
    records.append(SentenceRecord.from_dict({"tokens": [{"form": "x"}]}))
    records_to_dataset(records).to_parquet(str(tmp_path / "out.parquet"))
    assert verify_records(records, tmp_path / "out.parquet") == []
