import json

import pytest
import yaml

from seflag import load_matrix, run_matrix
from seflag.errors import ConfigError
from seflag.matrix import is_matrix


def matrix_config(conllu_dir, backend, **table):
    return {
        "name": "demo",
        "task": "lemmatization",
        "language": "la",
        "output": {"dir": str(conllu_dir.parent / "out")},
        "matrix": {
            "datasets": [
                {"label": "UD Latin", "dataset": {"source": "conllu", "path": str(conllu_dir)}},
                {"label": "UD Greek", "language": "grc", "dataset": {"source": "conllu", "path": str(conllu_dir)}},
            ],
            "models": [
                {"model": {"backend": backend}},  # label defaults to the backend name
                {"label": "first only", "language": "la", "model": {"backend": backend},
                 "options": {"max_sentences": 1}},
                {"label": "broken", "model": {"backend": "nope"}},
            ],
            "table": table,
        },
    }


def test_load_matrix_merges_entries_and_skips_language_mismatches(conllu_dir, fake_backend):
    matrix = load_matrix(matrix_config(conllu_dir, fake_backend))
    assert matrix.dataset_labels == ["UD Latin", "UD Greek"]
    assert matrix.model_labels == [fake_backend, "first only", "broken"]
    cells = {(c.dataset, c.model): c for c in matrix.cells}
    assert cells[("UD Greek", fake_backend)].config.language == "grc"  # dataset entry overrides the shared language
    assert cells[("UD Greek", "first only")].config is None  # la model on grc dataset
    first = cells[("UD Latin", "first only")].config
    assert first.options.max_sentences == 1 and first.task.type == "lemmatization"
    assert first.name == "demo-UD Latin-first only"


def test_matrix_label_and_section_errors(conllu_dir, fake_backend):
    data = matrix_config(conllu_dir, fake_backend)
    data["matrix"]["models"][1]["label"] = fake_backend
    with pytest.raises(ConfigError, match="labels must be unique"):
        load_matrix(data)
    data = matrix_config(conllu_dir, fake_backend)
    data["matrix"]["models"][0]["options"] = {"no_such_option": 1}
    with pytest.raises(ConfigError, match="cell 'UD Latin' × 'lowercase'"):
        load_matrix(data)
    data = matrix_config(conllu_dir, fake_backend)
    data["matrix"]["models"][0]["output"] = {"dir": None}  # output is shared, not per cell
    with pytest.raises(ConfigError, match="Invalid 'matrix' section"):
        load_matrix(data)


def test_run_matrix_table_and_files(conllu_dir, fake_backend):
    result = run_matrix(load_matrix(matrix_config(conllu_dir, fake_backend)))
    status = {(c.dataset, c.model): c.status for c in result.cells}
    assert status == {("UD Latin", "lowercase"): "ok", ("UD Latin", "first only"): "ok",
                      ("UD Latin", "broken"): "failed", ("UD Greek", "lowercase"): "ok",
                      ("UD Greek", "first only"): "not_applicable", ("UD Greek", "broken"): "failed"}
    table = result.table().splitlines()
    assert table[0] == "| Model | UD Latin | UD Greek |"
    assert table[2] == "| lowercase | 0.6667 | 0.6667 |"
    assert table[3] == "| first only | 0.7500 | – |"
    assert table[4] == "| broken | failed | failed |"
    assert "Metric: accuracy" in result.table() and "Failed: UD Latin × broken: Unknown backend" in result.table()
    assert len(result.failed) == 2

    saved = json.loads((result.path / "matrix.json").read_text(encoding="utf-8"))
    ok = next(c for c in saved["cells"] if c["status"] == "ok")
    assert ok["metrics"] == {"accuracy": pytest.approx(8 / 12)}
    assert (result.path / "cells" / ok["result_file"]).is_file()
    assert saved["config"]["matrix"]["models"][2]["label"] == "broken"
    assert (result.path / "table.md").read_text(encoding="utf-8").startswith("| Model |")


def test_matrix_by_subset_percent_and_several_metrics(conllu_dir, fake_backend):
    data = matrix_config(conllu_dir, fake_backend, by_subset=True, percent=True, digits=1,
                         metrics=["accuracy", "scored"])
    data["matrix"]["datasets"].pop()
    data["matrix"]["models"] = data["matrix"]["models"][:1]
    data["output"] = {"dir": None}
    result = run_matrix(load_matrix(data))
    assert result.path is None
    header, _, row = result.table().splitlines()[:3]
    assert header == ("| Model | UD Latin / total / accuracy | UD Latin / total / scored | "
                      "UD Latin / la_other-ud-dev / accuracy | UD Latin / la_other-ud-dev / scored | "
                      "UD Latin / la_test-ud-test / accuracy | UD Latin / la_test-ud-test / scored |")
    assert row == "| lowercase | 66.7 | 12 | 66.7 | 6 | 66.7 | 6 |"


def test_matrix_with_shared_dataset(conllu_dir, fake_backend):
    data = matrix_config(conllu_dir, fake_backend)
    data["dataset"] = data["matrix"].pop("datasets")[0]["dataset"]
    data["matrix"]["models"] = data["matrix"]["models"][:2]
    data["output"] = {"dir": None}
    result = run_matrix(load_matrix(data))
    assert result.dataset_labels == [str(conllu_dir)]
    with pytest.raises(ConfigError, match=r"'matrix\.models' or a shared 'model'"):
        load_matrix({"task": "lemmatization", "language": "la", "dataset": data["dataset"], "matrix": {}})


def test_cli_runs_matrix_files(conllu_dir, fake_backend, capsys):
    from seflag.cli import main

    path = conllu_dir.parent / "matrix.yaml"
    path.write_text(yaml.safe_dump(matrix_config(conllu_dir, fake_backend)), encoding="utf-8")
    assert is_matrix(path)
    assert main(["run", str(path), "--no-save"]) == 1  # the 'broken' model failed
    out = capsys.readouterr().out
    assert "| first only | 0.7500 | – |" in out and "->" not in out
    assert not (conllu_dir.parent / "out").exists()
    with pytest.raises(ConfigError, match="Unknown backend"):
        run_matrix(load_matrix(path, output={"dir": None}), fail_fast=True)
