import json
import logging

import pytest
import yaml

from seflag import EvaluationResult, load_backend, load_config, load_dataset, run, table_from_results
from seflag.cli import main
from seflag.errors import ConfigError, DataError
from seflag.runner import RESULT_FORMAT_VERSION


def section(conllu_dir):
    return {"source": "conllu", "path": str(conllu_dir)}


def test_load_dataset_from_section_and_docs(conllu_dir):
    dataset = load_dataset(section(conllu_dir), language="la")
    assert sum(1 for _ in dataset.records()) == 4
    docs = dataset.docs()
    assert len(docs) == 4 and docs[0][0].lemma_ == "arma" and docs[0][0].pos_ == "NOUN"
    assert str(docs[0][0].morph) == "Case=Acc|Number=Plur" and docs[0][0].dep_ == "obj"
    assert docs[0].user_data["sent_id"] == "o1"  # files in name order: la_other-ud-dev first
    lemmas_only = dataset.docs(layers=["lemma"], max_sentences=1)
    assert len(lemmas_only) == 1 and lemmas_only[0][0].pos_ == ""
    with pytest.raises(ConfigError, match="needs the language"):
        load_dataset(section(conllu_dir))


def test_load_dataset_and_backend_from_config(tmp_path, conllu_dir, fake_backend):
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({"language": "la", "task": "lemmatization", "dataset": section(conllu_dir),
                                    "model": {"backend": fake_backend}}), encoding="utf-8")
    assert len(load_dataset(path).docs(max_sentences=2)) == 2
    assert [t.lemma_ for t in load_backend(path).predict([["Arma", "Cano"]])[0]] == ["arma", "cano"]
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"language": "la", "task": "lemmatization", "dataset": section(conllu_dir),
                                      "matrix": {"models": [{"model": {"backend": fake_backend}}]}}))
    with pytest.raises(ConfigError, match="matrix configuration"):
        load_dataset(matrix)


def evaluate(conllu_dir, backend, out, **options):
    return run(load_config({"language": "la", "task": "lemmatization", "dataset": section(conllu_dir),
                            "model": {"backend": backend}, "output": {"dir": str(out)}, "options": options}))


def test_result_round_trip_and_format_checks(tmp_path, conllu_dir, fake_backend):
    result = evaluate(conllu_dir, fake_backend, tmp_path / "out")
    saved = json.loads(result.path.read_text(encoding="utf-8"))
    assert saved["format_version"] == RESULT_FORMAT_VERSION
    loaded = EvaluationResult.load(result.path)
    assert loaded.metrics == result.metrics and loaded.by_subset == result.by_subset and loaded.path == result.path
    (tmp_path / "other.json").write_text(json.dumps({"cells": []}))
    with pytest.raises(DataError, match="not a SEFLAG result file"):
        EvaluationResult.load(tmp_path / "other.json")
    (tmp_path / "future.json").write_text(json.dumps({**saved, "format_version": RESULT_FORMAT_VERSION + 1}))
    with pytest.raises(DataError, match="upgrade seflag"):
        EvaluationResult.load(tmp_path / "future.json")


def test_table_from_saved_results(tmp_path, conllu_dir, fake_backend, caplog, capsys):
    out = tmp_path / "out"
    evaluate(conllu_dir, fake_backend, out)
    newer = evaluate(conllu_dir, fake_backend, out, max_sentences=1)  # same dataset and model, other settings
    newer.provenance["timestamp"] = "9999"  # make sure it counts as the most recent one
    newer.save(newer.path)
    (out / "matrix.json").write_text(json.dumps({"cells": []}))  # skipped when searching a directory
    results = [EvaluationResult.load(p) for p in sorted(out.glob("*_*.json"))]
    with caplog.at_level(logging.WARNING, logger="seflag"):
        table = table_from_results(results).table()
    assert "different settings (options)" in caplog.text
    assert f"| {fake_backend} | 0.7500 |" in table  # the newer result: first sentence of each file
    assert main(["table", str(out), "--percent", "--digits", "1"]) == 0
    assert f"| {fake_backend} | 75.0 |" in capsys.readouterr().out
    assert main(["table", str(tmp_path / "missing")]) == 1
