"""Regression tests against real models; skipped unless the models are installed. Run with: pytest -m models

The expected scores are those published in the package README and ENCODING.md. They hold for the model versions
that SEFLAG's install hints pin (backends/spacy.py), with spaCy 3.8; other versions are skipped, not failed.
"""

import os
from importlib.metadata import version
from pathlib import Path

import pytest
import spacy.util
from packaging.version import Version

from seflag import load_config, run
from seflag.backends import morpheus_install
from seflag.backends.spacy import GRECY_RELEASE, LATINCY_WHEELS

EXAMPLES = Path(__file__).parents[1] / "examples"
pytestmark = pytest.mark.models

MODEL_VERSIONS = {"la_core_web_lg": LATINCY_WHEELS["lg"][0], "grc_proiel_lg": GRECY_RELEASE}


def require_model(model: str) -> None:
    if model not in spacy.util.get_installed_models():
        pytest.skip(f"{model} not installed")
    if version(model) != MODEL_VERSIONS[model] or Version(spacy.__version__) < Version("3.8"):
        pytest.skip(f"the expected scores are for {model} {MODEL_VERSIONS[model]} with spaCy 3.8 "
                    f"(installed: {version(model)}, spaCy {spacy.__version__})")


@pytest.mark.parametrize("config_name, model, expected", [
    ("latincy_lemma_ud.yaml", "la_core_web_lg", 0.9196),
    ("grecy_lemma_ud.yaml", "grc_proiel_lg", 0.9047),
])
def test_readme_results(config_name, model, expected):
    require_model(model)
    config = load_config(EXAMPLES / config_name, output={"dir": None})
    if "path" in config.dataset.params and not config.resolve_path(config.dataset.params["path"]).exists():
        pytest.skip("repository data not available")
    assert run(config).metrics["accuracy"] == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("config_name, model, expected", [
    ("latincy_morph_ud.yaml", "la_core_web_lg", 0.7090),
    ("grecy_morph_ud.yaml", "grc_proiel_lg", 0.6649),
])
def test_morphology(config_name, model, expected):
    test_readme_results(config_name, model, expected)


@pytest.mark.parametrize("config_name, model, expected", [
    ("latincy_deps_ud.yaml", "la_core_web_lg", 0.7660),
    ("grecy_deps_ud.yaml", "grc_proiel_lg", 0.5663),
])
def test_dependency_parsing(config_name, model, expected):
    require_model(model)
    result = run(load_config(EXAMPLES / config_name, output={"dir": None}))
    assert result.metrics["las"] == pytest.approx(expected, abs=5e-4)
    # gold sentence boundaries: the parser must not split sentences into several trees
    assert result.metrics["predicted_multiple_roots"] == 0


def test_model_sentence_segmentation_creates_multiple_roots():
    if "grc_proiel_lg" not in spacy.util.get_installed_models():
        pytest.skip("grc_proiel_lg not installed")
    config = load_config(EXAMPLES / "grecy_deps_ud.yaml", output={"dir": None},
                         options={"sentence_segmentation": "model", "max_sentences": 300})
    assert run(config).metrics["predicted_multiple_roots"] > 0


@pytest.mark.parametrize("config_name, model, expected", [
    ("latincy_ner_herodotos.yaml", "la_core_web_lg", 0.8519),
    ("grecy_ner_yousef.yaml", "grc_proiel_lg", 0.6704),
])
def test_ner(config_name, model, expected):
    require_model(model)
    config = load_config(EXAMPLES / config_name, output={"dir": None})
    assert run(config).metrics["f1"] == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("config_name, expected", [
    ("ugarit_bert_ner_yousef.yaml", 0.9857),
    ("flair_grc_bert_ner_yousef.yaml", 0.8824),
])
def test_ner_backends(config_name, expected):
    pytest.importorskip("transformers" if "ugarit" in config_name else "flair")
    assert run(load_config(EXAMPLES / config_name, output={"dir": None})).metrics["f1"] == \
        pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("config_name, expected", [
    ("morpheus_lemma_ud_latin.yaml", 0.7253),
    ("morpheus_lemma_ud_greek.yaml", 0.8755),
])
def test_morpheus(config_name, expected):
    if not (morpheus_install.installed() or "MORPHEUS_PATH" in os.environ):
        pytest.skip("Morpheus not installed (built on first use, or set MORPHEUS_PATH)")
    assert run(load_config(EXAMPLES / config_name, output={"dir": None})).metrics["accuracy"] == \
        pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("config_name, overrides, metric, expected", [
    ("agree_similarity_grc_fasttext.yaml", {}, "spearman", 0.3805),
    # the setup of the original SEFLAG script (gensim's evaluate_word_pairs defaults)
    ("agree_similarity_grc_fasttext.yaml", {"model": {"max_vocabulary": 300000, "case_insensitive": True}},
     "spearman", 0.3836),
    ("agree_retrieval_grc_fasttext.yaml", {}, "hits_at_10", 0.0396),
    ("lasla_synonyms_latin.yaml", {}, "accuracy", 0.8057),
])
def test_embeddings(config_name, overrides, metric, expected):
    pytest.importorskip("gensim")
    config = load_config(EXAMPLES / config_name, output={"dir": None}, **overrides)
    assert run(config).metrics[metric] == pytest.approx(expected, abs=5e-4)
