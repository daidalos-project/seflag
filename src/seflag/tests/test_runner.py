import json

import pytest

from seflag import load_config, run
from seflag.errors import ConfigError
from seflag.tests.conftest import LowercaseBackend


def config(conllu_dir, backend, task="lemmatization", **extra):
    return load_config({"language": "la", "task": task, "dataset": {"source": "conllu", "path": str(conllu_dir)},
                        "model": {"backend": backend}, "output": {"dir": str(conllu_dir.parent / "out")}, **extra})


def test_lemmatization_end_to_end(conllu_dir, fake_backend):
    result = run(config(conllu_dir, fake_backend))
    # per file: 7 words, 1 without gold lemma; lowercase is wrong for 'virum' and 'Troiae'
    assert result.metrics["scored"] == 12
    assert result.metrics["unannotated_in_gold"] == 2
    assert result.metrics["accuracy"] == pytest.approx(8 / 12)
    assert set(result.by_subset) == {"la_test-ud-test", "la_other-ud-dev"}
    assert result.details["top_errors"][0]["count"] == 2
    saved = json.loads(result.path.read_text(encoding="utf-8"))
    assert saved["metrics"]["accuracy"] == pytest.approx(8 / 12)
    assert saved["provenance"]["model"] == {"name": "lowercase"}
    assert saved["config"]["dataset"]["source"] == "conllu"


def test_label_normalization_and_exclusion(conllu_dir, fake_backend):
    cfg = config(conllu_dir, fake_backend, task={"type": "lemmatization", "exclude_upos": ["PROPN"]},
                 options={"label_normalization": ["lowercase"], "max_sentences": 2})
    result = run(cfg)
    assert result.metrics["excluded"] == 1
    assert result.metrics["accuracy"] == pytest.approx(4 / 5)  # "virum" -> "vir" stays wrong
    assert result.config["options"]["label_normalization"] == ["lowercase"]


def test_pos_tagging(conllu_dir, fake_backend):
    result = run(config(conllu_dir, fake_backend, task="pos_tagging"))
    assert result.metrics["accuracy"] == pytest.approx(4 / 14)
    assert result.metrics["per_label"]["NOUN"]["recall"] == 1.0
    assert result.metrics["confusion_matrix"]["VERB"] == {"NOUN": 2}


def test_unknown_components(conllu_dir, fake_backend):
    with pytest.raises(ConfigError, match="Unknown backend"):
        run(config(conllu_dir, "nope"))
    with pytest.raises(ConfigError, match="section 'task'"):
        run(config(conllu_dir, fake_backend, task={"type": "pos_tagging", "tagset": "bad"}))


def test_cache_info_and_purge(tmp_path, monkeypatch, capsys):
    from seflag.cli import main

    monkeypatch.setenv("SEFLAG_HOME", str(tmp_path))
    assert main(["cache", "purge", "--yes"]) == 0
    assert "nothing to delete" in capsys.readouterr().out
    downloads = tmp_path / "downloads" / "abc"
    downloads.mkdir(parents=True)
    (downloads / "model.bin").write_bytes(b"x" * 2048)
    tool = tmp_path / "tools" / "morpheus-1.0.3" / "bin"
    tool.mkdir(parents=True)
    (tool / "morpheus").write_bytes(b"x" * 1024)
    assert main(["cache", "info"]) == 0
    out = capsys.readouterr().out
    assert "1 file(s), 2.0 KB" in out and "model.bin" in out and "1.0 KB  morpheus-1.0.3" in out
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["cache", "purge"]) == 1  # no confirmation possible -> nothing deleted
    assert (downloads / "model.bin").exists()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["cache", "purge"]) == 0 and (downloads / "model.bin").exists()
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert main(["cache", "purge"]) == 0 and not (tmp_path / "downloads").exists()
    assert not (tmp_path / "tools").exists()
    assert "1 downloaded file(s) and morpheus-1.0.3, freed 3.0 KB" in capsys.readouterr().out


def test_relative_output_dir_is_relative_to_working_directory(tmp_path, conllu_dir, fake_backend, monkeypatch):
    import yaml

    from seflag.matrix import load_matrix, run_matrix

    configs = tmp_path / "configs"  # e.g. the examples inside an installed (read-only) package
    configs.mkdir()
    data = {"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": "../data"},
            "model": {"backend": fake_backend}, "output": {"dir": "results"}}
    (configs / "eval.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    (configs / "matrix.yaml").write_text(yaml.safe_dump({**data, "matrix": {}}), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    result = run(load_config(configs / "eval.yaml"))  # dataset path: relative to the config file
    assert result.path.parent == work / "results"
    matrix = run_matrix(load_matrix(configs / "matrix.yaml"))
    assert matrix.path.parent == work / "results"
    assert not (configs / "results").exists()


class MisalignedBackend(LowercaseBackend):
    """Test backend that breaks gold tokenization: lowercases the tokens themselves, or drops the last sentence."""

    drop_sentence = False

    def predict(self, sentences):
        if self.drop_sentence:
            return super().predict(sentences[:-1])
        return super().predict([[w.lower() for w in words] for words in sentences])


def test_misaligned_predictions_are_reported(conllu_dir, monkeypatch):
    from seflag import registry
    from seflag.errors import PredictionError

    monkeypatch.setitem(registry.BUILTINS["backends"], "misaligned", "seflag.tests.test_runner:MisalignedBackend")
    with pytest.raises(PredictionError, match=r"sentence 1: .* token 1: expected 'Arma', found 'arma'"):
        run(config(conllu_dir, "misaligned", output={"dir": None}))
    monkeypatch.setattr(MisalignedBackend, "drop_sentence", True)
    with pytest.raises(PredictionError, match="returned 3 documents for 4 sentences"):
        run(config(conllu_dir, "misaligned", output={"dir": None}))
