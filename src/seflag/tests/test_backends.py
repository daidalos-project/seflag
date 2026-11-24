import hashlib
import sys
import tarfile
import unicodedata

import pytest
import spacy

from seflag import load_config, run
from seflag.backends import morpheus_install
from seflag.backends.morpheus import MorpheusBackend
from seflag.errors import MissingDependencyError, SeflagError

FAKE_MORPHEUS = '''#!{python}
"""Imitates Morpheus: no output for numbers, lowercased echo of unknown words, beta code lemmas."""
import sys
LEXICON = {{"amabat": ["amo"], "amo": ["amo"], "cano": ["cano", "canus"], "est": ["edo#1", "sum#1"],
           "lo/gos": ["lo/gos"], "kai\\\\": ["kai/"], "d'": ["de/"], "par'": ["para/"]}}
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


def skip_on_windows():
    """The fake Morpheus runs through its shebang line and its fake build needs make: neither works on Windows."""
    if sys.platform == "win32":
        pytest.skip("Morpheus runs on Linux and macOS only")


@pytest.fixture
def morpheus(tmp_path, monkeypatch):
    skip_on_windows()
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
    assert "\u1f77" not in lemmas[1]  # no oxia: tonos as in UD


def test_morpheus_greek_input_nfc_and_elision(morpheus):
    pytest.importorskip("betacode")
    # NFD input and every elision mark (incl. UD Perseus' U+0313 and its ρ + U+0313 merged into ῤ by NFC)
    words = [unicodedata.normalize("NFD", "λόγος"), "δ\u0313", "δ\u1fbf", "δ\u1fbd", "δ\u02bc", "πα\u1fe4"]
    assert [t.lemma_ for t in morpheus_backend("grc").predict([words])[0]] == ["λόγος", "δέ", "δέ", "δέ", "δέ", "παρά"]


def test_morpheus_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("MORPHEUS_PATH", str(tmp_path))
    with pytest.raises(MissingDependencyError, match="No Morpheus program"):
        morpheus_backend("la").predict([["x"]])


def test_morpheus_install_never(monkeypatch):
    monkeypatch.delenv("MORPHEUS_PATH", raising=False)
    with pytest.raises(MissingDependencyError, match="install: auto"):
        morpheus_backend("la", install="never").predict([["x"]])


@pytest.fixture
def morpheus_source(tmp_path, monkeypatch):
    """A fake Morpheus source release whose makefiles 'build' the fake Morpheus script."""
    skip_on_windows()
    top = tmp_path / "src" / f"morpheus-perseids-{morpheus_install.VERSION}"
    (top / "stemlib" / "Latin").mkdir(parents=True)
    for directory, target in morpheus_install.TARGETS:
        (top / "src" / directory).mkdir(parents=True)
        recipe = "cp fake.py $@ && chmod +x $@" if target == "morpheus" else "touch $@"
        (top / "src" / directory / "makefile").write_text(f"{target}:\n\t{recipe}\n")
    (top / "src" / "anal" / "fake.py").write_text(FAKE_MORPHEUS.format(python=sys.executable))
    archive = tmp_path / "morpheus.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(top, arcname=top.name)
    monkeypatch.setattr(morpheus_install, "URL", archive.as_uri())
    monkeypatch.setattr(morpheus_install, "CHECKSUM", "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setenv("SEFLAG_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MORPHEUS_PATH", raising=False)
    return archive


def test_morpheus_auto_install(morpheus_source, monkeypatch):
    backend = morpheus_backend("la")
    assert [t.lemma_ for t in backend.predict([["cano", "est"]])[0]] == ["cano", "edo#1"]
    metadata = backend.metadata()
    assert metadata["installed_by"] == "seflag" and metadata["version"] == morpheus_install.VERSION
    root = morpheus_install.install_dir()
    assert metadata["path"] == str(root) and not (root / "src").exists()
    # the installation is reused: no second download
    monkeypatch.setattr(morpheus_install, "download", lambda *a: pytest.fail("downloaded again"))
    assert morpheus_backend("la").predict([["cano"]])[0][0].lemma_ == "cano"


def test_morpheus_auto_install_needs_build_tools(morpheus_source, monkeypatch):
    monkeypatch.setattr(morpheus_install.shutil, "which", lambda tool: None if tool == "make" else "/usr/bin/" + tool)
    with pytest.raises(MissingDependencyError, match="needs make"):
        morpheus_backend("la").predict([["x"]])
    assert morpheus_install.installed() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_morpheus_auto_install_windows(monkeypatch, tmp_path):
    monkeypatch.setenv("SEFLAG_HOME", str(tmp_path))
    monkeypatch.delenv("MORPHEUS_PATH", raising=False)
    with pytest.raises(MissingDependencyError, match="WSL"):
        morpheus_backend("la").predict([["x"]])


# --- flair ------------------------------------------------------------------------------------------------

class FakeTagger:
    """Stands in for a flair SequenceTagger: labels sentences through flair's own API, without loading a model."""

    def __init__(self, tag_type, spans):
        from flair.data import Dictionary

        self.tag_type, self.predict_spans = tag_type, spans
        self.label_dictionary = Dictionary(add_unk=True)
        for label in (["O", "S-PER", "B-LOC", "E-LOC", "<START>", "<STOP>"] if spans else ["NOUN", "VERB", "X"]):
            self.label_dictionary.add_item(label)

    def predict(self, sentences, mini_batch_size):
        for sentence in sentences:
            if self.predict_spans:
                sentence[0:1].add_label(self.tag_type, "PER")
                sentence[1:3].add_label(self.tag_type, "LOC")
            else:
                for i, token in enumerate(sentence):
                    token.add_label(self.tag_type, "NOUN" if i % 2 == 0 else "VERB")


def flair_backend(tagger, **params):
    from seflag.backends.flair import FlairBackend

    config = load_config({"language": "grc", "task": "ner", "model": {"backend": "flair", "name": "x"},
                          "dataset": {"source": "conllu", "path": "x"}})
    backend = FlairBackend(FlairBackend.Config(name="x", **params), config)
    backend._tagger = tagger
    return backend


def test_flair_backend_entities_and_tags():
    pytest.importorskip("flair")
    words = ["Σωκράτης", "ἐν", "Ἀθήναις", "ἦν"]
    backend = flair_backend(FakeTagger("ner", spans=True))
    doc = backend.predict([words])[0]
    assert [(e.text, e.label_) for e in doc.ents] == [("Σωκράτης", "PER"), ("ἐν Ἀθήναις", "LOC")]
    assert backend.label_inventory("entities") == {"PER", "LOC"}
    assert backend.label_inventory("upos") is None
    tags = flair_backend(FakeTagger("upos", spans=False))
    doc = tags.predict([words])[0]
    assert [t.pos_ for t in doc] == ["NOUN", "VERB", "NOUN", "VERB"]
    assert tags.label_inventory("upos") == {"NOUN", "VERB", "X"}
    xpos = flair_backend(FakeTagger("pos-fine", spans=False), layer="xpos")
    assert [t.tag_ for t in xpos.predict([words])[0]] == ["NOUN", "VERB", "NOUN", "VERB"]


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
                                     id2label=dict(enumerate(labels)),
                                     label2id={label: i for i, label in enumerate(labels)})
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


def test_retokenizing_component_is_named(tmp_path, conllu_dir):
    """Pipelines must keep the gold tokens; the error names the component that changes them."""
    nlp = spacy.blank("la")
    nlp.add_pipe("entity_ruler").add_patterns([{"label": "PER", "pattern": [{"ORTH": "Arma"}, {"ORTH": "virum"}]}])
    nlp.add_pipe("merge_entities")
    nlp.to_disk(tmp_path / "model")
    data = {"language": "la", "task": "lemmatization", "output": {"dir": None},
            "dataset": {"source": "conllu", "path": str(conllu_dir)},
            "model": {"backend": "spacy", "name": str(tmp_path / "model")}}
    with pytest.raises(SeflagError, match=r"in its component 'merge_entities'\. Add it to 'exclude'"):
        run(load_config(data))
    run(load_config({**data, "model": {**data["model"], "exclude": ["merge_entities"]}}))


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
    default = launcher.python_command(launcher.LauncherSpec(requirements=[]))
    assert default[4:8] == ["--python", f"{sys.version_info.major}.{sys.version_info.minor}", "--with", "seflag==0.1.0"]
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    with pytest.raises(MissingDependencyError, match="uv"):
        launcher.python_command(spec)
    command = ["conda", "run", "-n", "x", "python"]
    assert launcher.python_command(launcher.LauncherSpec(command=command)) == command  # no uv needed


def test_seflag_requirement_from_local_tree(monkeypatch, tmp_path):
    import json

    from seflag import launcher

    tree = tmp_path / "my seflag"  # a space: percent-encoded in the file URL
    direct_url = json.dumps({"url": tree.as_uri(), "dir_info": {"editable": True}})

    class Distribution:
        def read_text(self, name):
            return direct_url

    monkeypatch.setattr(launcher, "distribution", lambda name: Distribution())
    assert launcher.seflag_requirement() == (str(tree), True)


@pytest.mark.parametrize("launcher", [{}, {"command": []}, {"command": ["python"], "requirements": ["x"]},
                                      {"unknown": 1}])
def test_invalid_launcher(launcher):
    from seflag.errors import ConfigError

    with pytest.raises(ConfigError):
        load_config({"language": "la", "task": "lemmatization", "dataset": {"source": "conllu", "path": "x"},
                     "model": {"backend": "spacy", "name": "x", "launcher": launcher}})
