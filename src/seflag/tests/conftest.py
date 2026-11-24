from __future__ import annotations

from pathlib import Path

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from seflag import registry
from seflag.backends.base import Backend


def pytest_configure(config):
    config.addinivalue_line("markers", "network: tests that need internet access (skipped without it)")


@pytest.fixture
def tmp_path(tmp_path: Path) -> Path:
    """pytest's tmp_path, resolved: on Windows it can be an 8.3 short path (C:\\Users\\RUNNER~1\\...), while SEFLAG
    resolves relative paths and output directories to their long form, so the two would not compare equal."""
    return tmp_path.resolve()


SAMPLE_CONLLU = """# sent_id = s1
# text = Arma virumque cano
1\tArma\tarma\tNOUN\t_\tCase=Acc|Number=Plur\t3\tobj\t_\t_
2-3\tvirumque\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No
2\tvirum\tvir\tNOUN\t_\tCase=Acc\t4\tobj\t_\t_
3\tque\tque\tCCONJ\t_\t_\t4\tcc\t_\t_
4\tcano\tcano\tVERB\t_\t_\t0\troot\t_\t_

# source = Vergil, Aeneid 1.1
# text = Troiae qui primus
# sent_id = s2
# newpar
1\tTroiae\tTroia\tPROPN\t_\t_\t3\tobl\t_\tGloss=of-Troy
2\tqui\tqui\tPRON\t_\t_\t3\tnsubj\t3:nsubj\t_
3\tprimus\t_\tADJ\t_\t_\t0\troot\t0:root\tSpaceAfter=No
3.1\tfuit\tsum\tAUX\t_\t_\t_\t_\t3:cop\t_

"""


class LowercaseBackend(Backend):
    """Test backend: lemma = lowercased form, every token tagged NOUN."""

    description = "test backend"

    def predict(self, sentences: list[list[str]]) -> list[Doc]:
        vocab = Vocab()
        return [Doc(vocab, words=w, lemmas=[x.lower() for x in w], pos=["NOUN"] * len(w)) for w in sentences]

    def metadata(self):
        return {"name": "lowercase"}


@pytest.fixture
def conllu_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    # newline="\n": CoNLL-U files use LF (as SEFLAG writes them), also on Windows, where text mode would write CRLF
    (d / "la_test-ud-test.conllu").write_text(SAMPLE_CONLLU, encoding="utf-8", newline="\n")
    (d / "la_other-ud-dev.conllu").write_text(SAMPLE_CONLLU.replace("s1", "o1").replace("s2", "o2"),
                                               encoding="utf-8", newline="\n")
    return d


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setitem(registry.BUILTINS["backends"], "lowercase", "seflag.tests.conftest:LowercaseBackend")
    return "lowercase"
