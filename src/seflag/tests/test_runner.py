import json

import pytest

from seflag import load_config, run
from seflag.errors import ConfigError


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
    assert main(["cache", "info"]) == 0
    out = capsys.readouterr().out
    assert "1 file(s), 2.0 KB" in out and "model.bin" in out
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["cache", "purge"]) == 1  # no confirmation possible -> nothing deleted
    assert (downloads / "model.bin").exists()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["cache", "purge"]) == 0 and (downloads / "model.bin").exists()
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert main(["cache", "purge"]) == 0 and not (tmp_path / "downloads").exists()
    assert "freed 2.0 KB" in capsys.readouterr().out
