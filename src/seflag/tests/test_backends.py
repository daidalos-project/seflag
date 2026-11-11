import sys
import unicodedata

import pytest
import spacy

from seflag import load_config, run
from seflag.backends.morpheus import MorpheusBackend
from seflag.errors import MissingDependencyError

FAKE_MORPHEUS = '''#!{python}
"""Imitates Morpheus: no output for numbers, lowercased echo of unknown words, beta code lemmas."""
import sys
LEXICON = {{"amabat": ["amo"], "cano": ["cano", "canus"], "est": ["edo#1", "sum#1"],
           "lo/gos": ["lo/gos"], "kai\\\\": ["kai/"]}}
print("<words>")
for line in sys.stdin.read().split("\\n")[:-1]:
    if line.isdigit():
        continue
    if line in LEXICON:
        print("<word><form>" + line + "</form>" + "".join(
            "<entry><dict><hdwd>" + h + "</hdwd></dict></entry>" for h in LEXICON[line]) + "</word>")
    else:
        print("<unknown>" + line.lower() + "</unknown>")
print("</words>")
'''


@pytest.fixture
def morpheus(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    program = tmp_path / "bin" / "morpheus"
    program.write_text(FAKE_MORPHEUS.format(python=sys.executable), encoding="utf-8")
    program.chmod(0o755)
    monkeypatch.setenv("MORPHEUS_PATH", str(tmp_path))
    return tmp_path


def morpheus_backend(language, **params):
    config = load_config({"language": language, "task": "lemmatization", "model": {"backend": "morpheus"},
                          "dataset": {"source": "conllu", "path": "x"}})
    return MorpheusBackend(MorpheusBackend.Config(**params), config)


def test_morpheus_latin_alignment_and_options(morpheus):
    backend = morpheus_backend("la")
    docs = backend.predict([["Amabat", "26", "cano", ",", "Caesar", "est"]])
    assert [t.lemma_ for t in docs[0]] == ["", "26", "cano", ",", "", "edo#1"]  # 'Amabat' unknown: case matters
    assert backend.metadata()["unknown_words"] == 2  # Amabat, Caesar (numbers/punctuation are not sent)
    docs = morpheus_backend("la", select="last", unknown="form", non_words="empty").predict([["est", "Caesar", "26"]])
    assert [t.lemma_ for t in docs[0]] == ["sum#1", "Caesar", ""]


def test_morpheus_greek_betacode_and_nfc(morpheus):
    pytest.importorskip("betacode")
    docs = morpheus_backend("grc").predict([["λόγος", "καὶ"]])
    lemmas = [t.lemma_ for t in docs[0]]
    assert lemmas == ["λόγος", "καί"]
    assert all(unicodedata.is_normalized("NFC", lemma) for lemma in lemmas)
    assert "ί" not in lemmas[1]  # no oxia: tonos as in UD


def test_morpheus_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("MORPHEUS_PATH", str(tmp_path))
    with pytest.raises(MissingDependencyError, match="No Morpheus program"):
        morpheus_backend("la").predict([["x"]])


# --- transformers -------------------------------------------------------------------------------------------

@pytest.fixture
def tiny_token_classifier(tmp_path):
    transformers = pytest.importorskip("transformers")
    words = ["Arma", "virum", "que", "cano", "Troiae", "qui", "primus", "ab", "oris"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [w.lower() for w in words] + ["##s"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    labels = ["O", "B-PER", "I-PER", "B-LOC"]
    config = transformers.BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                                     num_attention_heads=1, intermediate_size=8, max_position_embeddings=64,
                                     id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    model = transformers.BertForTokenClassification(config)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    return tmp_path


def test_transformers_backend_aligns_words_and_windows(tiny_token_classifier):
    from seflag.backends.transformers import TransformersBackend

    config = load_config({"language": "la", "task": "ner", "model": {"backend": "transformers", "name": "x"},
                          "dataset": {"source": "conllu", "path": "x"}})
    words = ["Arma", "virumque", "cano", "Troiae", "qui", "primus", "ab", "oris"] * 3
    backend = TransformersBackend(TransformersBackend.Config(name=str(tiny_token_classifier), max_length=8), config)
    labels = backend._predict_labels([words, words[:2]])
    assert [len(x) for x in labels] == [24, 2]
    assert all(label in {"O", "B-PER", "I-PER", "B-LOC"} for x in labels for label in x)  # every word labelled
    lengths = {w: len(backend._tokenizer(w, add_special_tokens=False)["input_ids"]) for w in words}
    windows = backend._windows(words, lengths)
    assert len(windows) > 1 and windows[0][0] == 0 and windows[-1][1] == 24
    assert all(sum(lengths[w] for w in words[a:b]) <= backend._limit() for a, b in windows)
    docs = backend.predict([words])
    assert all(ent.label_ in {"PER", "LOC"} for ent in docs[0].ents)
    assert backend.label_inventory("entities") == {"PER", "LOC"}


# --- isolated execution -------------------------------------------------------------------------------------

@pytest.fixture
def rule_model(tmp_path):
    nlp = spacy.blank("la")
    nlp.add_pipe("attribute_ruler").add([[{"ORTH": "Arma"}]], {"LEMMA": "arma", "POS": "NOUN"})
    nlp.add_pipe("entity_ruler").add_patterns([{"label": "LOC", "pattern": "Troiae"}])
    nlp.to_disk(tmp_path / "model")
    return tmp_path / "model"


def test_launched_backend_matches_in_process(rule_model, conllu_dir):
    data = {"language": "la", "task": "lemmatization", "output": {"dir": None},
            "dataset": {"source": "conllu", "path": str(conllu_dir)},
            "model": {"backend": "spacy", "name": str(rule_model)}}
    direct = run(load_config(data))
    # a separate process with the current interpreter exercises the whole worker protocol
    launched = run(load_config({**data, "model": {**data["model"], "launcher": {"command": [sys.executable]}}}))
    assert launched.metrics == direct.metrics
    assert launched.provenance["model"]["launcher"]["command"] == [sys.executable]
    assert launched.provenance["model"]["pipeline"] == ["attribute_ruler", "entity_ruler"]


def test_launched_backend_failure_is_reported(rule_model, conllu_dir):
    from seflag.errors import SeflagError

    config = load_config({"language": "la", "task": "lemmatization", "output": {"dir": None},
                          "dataset": {"source": "conllu", "path": str(conllu_dir)},
                          "model": {"backend": "spacy", "name": str(rule_model / "missing"),
                                    "launcher": {"command": [sys.executable]}}})
    with pytest.raises(SeflagError, match="exit code"):
        run(config)


def test_uv_command(monkeypatch):
    from seflag import launcher

    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(launcher, "seflag_requirement", lambda: ("/src/seflag", True))
    spec = launcher.LauncherSpec(requirements=["flair==0.15.1"], python="3.11")
    assert launcher.python_command(spec) == [
        "/usr/bin/uv", "run", "--no-project", "--isolated", "--python", "3.11",
        "--with-editable", "/src/seflag", "--with", "flair==0.15.1", "python"]
    monkeypatch.setattr(launcher, "seflag_requirement", lambda: ("seflag==0.1.0", False))
    assert launcher.python_command(launcher.LauncherSpec(requirements=[]))[4:6] == ["--with", "seflag==0.1.0"]
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    with pytest.raises(MissingDependencyError, match="uv"):
        launcher.python_command(spec)
    command = ["conda", "run", "-n", "x", "python"]
    assert launcher.python_command(launcher.LauncherSpec(command=command)) == command  # no uv needed


@pytest.mark.parametrize("launcher", [{}, {"command": []}, {"command": ["python"], "requirements": ["x"]},
                                      {"unknown": 1}])
def test_invalid_launcher(launcher):
    from seflag.errors import ConfigError

    with pytest.raises(ConfigError):
        load_config({"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": "x"},
                     "model": {"backend": "spacy", "name": "x", "launcher": launcher}})
