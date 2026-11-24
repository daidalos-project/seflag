"""The Lamon backend with a fake lamonpy: the bridge script runs for real, in a subprocess of the current Python."""

import ast
import json
import sys

import pytest

from seflag import load_config, run
from seflag.backends import lamon
from seflag.errors import ConfigError, MissingDependencyError, SeflagError

FAKE_LAMONPY = '''
import json, os, re

LEXICON = {"Arma": ("arma", "n-p---na-"), "virum": ("vir", "n-s---ma-"), "que": ("que", "c--------"),
           "cano": ("cano", "v1spia---"), "Troiae": ("Troia", "n-s---fg-"), "qui": ("qui", "p-s---mn-"),
           "primus": ("primus", "a-s---mnp"), "fuit": ("sum", "v3sria---"), "cum": ("cum2", "r--------"),
           "XII": ("[NUM]", "---------"), "venit": ("venio", "v3sria---"), "a_b": ("ab", "r--------")}


class Lamon:
    def __init__(self, dict_path=None, tagger_path=None):
        log = os.environ.get("FAKE_LAMON_LOG")
        if log:
            with open(log, "w") as f:
                json.dump({"dict_path": dict_path, "tagger_path": tagger_path}, f)

    def tag(self, text, tag_style="perseus", beam_size=5, bidirection=False):
        assert tag_style == "perseus"
        tokens = []
        for match in re.finditer(r"\\.|[^\\s.]+", text):  # like Lamon, splits '...' into three tokens
            word = match.group()
            lemma, tag = LEXICON.get(word, (word.lower(), "---------"))
            tokens.append((match.start(), match.end(), lemma, tag))
        return [(-1.0, tokens), (-2.0, [])]

    def tag_multi(self, texts, num_workers=1, **options):
        return (self.tag(text, **options) for text in texts)
'''


@pytest.fixture
def fake_lamonpy(tmp_path, monkeypatch):
    (tmp_path / "fake" / "lamonpy").mkdir(parents=True)
    (tmp_path / "fake" / "lamonpy" / "__init__.py").write_text(FAKE_LAMONPY, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake"))
    return tmp_path


def lamon_config(conllu_dir, task="lemmatization", **params):
    return load_config({"language": "la", "task": task, "output": {"dir": None},
                        "dataset": {"source": "conllu", "path": str(conllu_dir)},
                        "model": {"backend": "lamon", "command": [sys.executable], **params}})


def backend(conllu_dir, **params):
    config = lamon_config(conllu_dir, **params)
    return lamon.LamonBackend(lamon.LamonBackend.Config(**config.model.params), config)


def test_perseus_to_ud():
    # UD Latin conventions: aspect on every finite form, aspect but no tense on participles and infinitives
    assert lamon.perseus_to_ud("v3spia---", "sum", "est") == (
        "VERB", "Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act")
    assert lamon.perseus_to_ud("v3slia---", "venio", "venerat")[1].startswith("Aspect=Perf|Mood=Ind")
    assert lamon.perseus_to_ud("v-srppfn-", "divido", "divisa") == (
        "VERB", "Aspect=Perf|Case=Nom|Gender=Fem|Number=Sing|VerbForm=Part|Voice=Pass")
    assert lamon.perseus_to_ud("v-sfpamn-", "venio", "venturus")[1].startswith("Aspect=Prosp|")
    assert lamon.perseus_to_ud("v--pna---", "venio", "venire") == ("VERB", "Aspect=Imp|VerbForm=Inf|Voice=Act")
    assert lamon.perseus_to_ud("v-spgpfn-", "laudo", "laudanda")[1] == \
        "Case=Nom|Gender=Fem|Number=Sing|VerbForm=Gdv|Voice=Pass"
    # positive degree unmarked, superlative 'Abs'
    assert lamon.perseus_to_ud("a-s---mnc", "magnus", "maior") == ("ADJ", "Case=Nom|Degree=Cmp|Gender=Masc|Number=Sing")
    assert lamon.perseus_to_ud("a-s---mns", "magnus", "maximus")[1] == "Case=Nom|Degree=Abs|Gender=Masc|Number=Sing"
    assert lamon.perseus_to_ud("a-s---mnp", "magnus", "magnus")[1] == "Case=Nom|Gender=Masc|Number=Sing"
    assert lamon.perseus_to_ud("v3spid---", "loquor", "loquitur")[1].count("Voice") == 0  # deponent: no UD value
    assert lamon.perseus_to_ud("---------", "[NUM]", "XII") == ("NUM", "")
    assert lamon.perseus_to_ud("---------", ".", ".") == ("PUNCT", "")
    assert lamon.perseus_to_ud("---------", "x", "x") == ("X", "")
    assert lamon.perseus_to_ud("", "x", "x") == ("X", "")


def test_predict_aligns_lamon_tokens(conllu_dir, fake_lamonpy):
    model = backend(conllu_dir)
    docs = model.predict([["Arma", "virum", "que", "cano", "..."], ["cum", "XII", "a b", "venit"]])
    assert [t.lemma_ for t in docs[0]] == ["arma", "vir", "que", "cano", "..."]  # '...' split by Lamon, kept
    assert [t.pos_ for t in docs[0]] == ["NOUN", "NOUN", "CCONJ", "VERB", "PUNCT"]
    assert str(docs[0][0].morph) == "Case=Acc|Gender=Neut|Number=Plur"
    # numbered homonyms stay (strip_homonym_digits); numbers get their form; whitespace inside a token is kept
    assert [t.lemma_ for t in docs[1]] == ["cum2", "XII", "ab", "venio"]
    assert [t.text for t in docs[1]] == ["cum", "XII", "a b", "venit"]
    info = model.metadata()
    assert info["tokens"] == 9 and info["split_tokens"] == 1 and info["unanalysed_tokens"] == 0
    assert info["python"] == sys.version.split()[0] and info["lamonpy"] is None  # the fake has no metadata
    assert info["command"] == [sys.executable]
    assert model.predict([]) == []


def test_numbers_option(conllu_dir, fake_lamonpy):
    assert backend(conllu_dir, numbers="empty").predict([["XII"]])[0][0].lemma_ == ""


def test_model_paths_are_resolved(conllu_dir, fake_lamonpy, monkeypatch):
    log = fake_lamonpy / "log.json"
    monkeypatch.setenv("FAKE_LAMON_LOG", str(log))
    backend(conllu_dir, dict_path="dict.large.bin", workers=2).predict([["cano"]])
    paths = json.loads(log.read_text(encoding="utf-8"))
    assert paths["dict_path"].endswith("dict.large.bin") and paths["tagger_path"] is None


def test_evaluation_with_lamon(conllu_dir, fake_lamonpy):
    result = run(lamon_config(conllu_dir, "lemmatization"))
    assert result.metrics["scored"] == 12 and result.metrics["accuracy"] == 1.0  # 2 files of 6 lemmatized words
    assert result.provenance["model"]["name"] == "lamon"
    # 'features: model' uses the backend's feature inventory: the UD features the Perseus tags map to
    morph = run(lamon_config(conllu_dir, {"type": "morphology", "features": "model"}))
    assert morph.metrics["scored"] > 0 and "Case" in morph.metrics["per_feature"]
    assert "InflClass" not in morph.metrics["per_feature"]  # in the gold data, but not predicted by Lamon


def test_lamonpy_missing(conllu_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # no lamonpy there or in SEFLAG's test environment
    with pytest.raises(SeflagError, match="lamonpy is not installed"):
        backend(conllu_dir).predict([["cano"]])


def test_uv_command(conllu_dir, monkeypatch):
    config = load_config({"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": "x"},
                          "model": {"backend": "lamon"}})
    model = lamon.LamonBackend(lamon.LamonBackend.Config(**config.model.params), config)
    monkeypatch.setattr(lamon.shutil, "which", lambda name: "/usr/bin/uv")
    assert model.command() == ["/usr/bin/uv", "run", "--no-project", "--isolated", "--python", "3.8",
                               "--with", "lamonpy==0.2.0", "python", str(lamon.BRIDGE)]
    assert model.metadata()["requirement"] == "lamonpy==0.2.0"
    monkeypatch.setattr(lamon.shutil, "which", lambda name: None)
    with pytest.raises(MissingDependencyError, match="uv"):
        model.command()


def test_only_latin_and_valid_options():
    with pytest.raises(ConfigError, match="only supports Latin"):
        run(load_config({"language": "grc", "task": "lemmatization", "dataset": {"source": "conllu", "path": "x"},
                         "model": {"backend": "lamon"}}))
    with pytest.raises(ConfigError):
        load_config({"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": "x"},
                     "model": {"backend": "lamon", "command": []}})


def test_bridge_runs_on_python_38():
    """The bridge runs in lamonpy's Python 3.8 environment, so it must not use newer syntax."""
    ast.parse(lamon.BRIDGE.read_text(encoding="utf-8"), feature_version=(3, 8))
