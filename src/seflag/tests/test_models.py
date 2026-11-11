"""Regression tests against real models; skipped unless the models are installed. Run with: pytest -m models"""

from pathlib import Path

import pytest
import spacy.util

from seflag import load_config, run

EXAMPLES = Path(__file__).parents[1] / "examples"
pytestmark = pytest.mark.models


@pytest.mark.parametrize("config_name, model, expected", [
    ("latincy_lemma_ud.yaml", "la_core_web_lg", 0.8905),  # README: 0.8843 incl. multiword-token lines
    ("grecy_lemma_ud.yaml", "grc_proiel_lg", 0.9056),  # README: 0.9055 incl. multiword-token lines
])
def test_readme_results(config_name, model, expected):
    if model not in spacy.util.get_installed_models():
        pytest.skip(f"{model} not installed")
    config = load_config(EXAMPLES / config_name, output={"dir": None})
    if "path" in config.dataset.params and not config.resolve_path(config.dataset.params["path"]).exists():
        pytest.skip("repository data not available")
    assert run(config).metrics["accuracy"] == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("config_name, model, expected", [
    ("latincy_morph_ud.yaml", "la_core_web_lg", 0.5464),
    ("grecy_morph_ud.yaml", "grc_proiel_lg", 0.6776),
])
def test_morphology(config_name, model, expected):
    test_readme_results(config_name, model, expected)


@pytest.mark.parametrize("config_name, model, expected", [
    ("latincy_deps_ud.yaml", "la_core_web_lg", 0.7023),
    ("grecy_deps_ud.yaml", "grc_proiel_lg", 0.5704),
])
def test_dependency_parsing(config_name, model, expected):
    if model not in spacy.util.get_installed_models():
        pytest.skip(f"{model} not installed")
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
    ("latincy_ner_herodotos.yaml", "la_core_web_lg", 0.5015),
    ("grecy_ner_yousef.yaml", "grc_proiel_lg", 0.6652),
])
def test_ner(config_name, model, expected):
    if model not in spacy.util.get_installed_models():
        pytest.skip(f"{model} not installed")
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
    ("morpheus_lemma_ud_greek.yaml", 0.8613),
])
def test_morpheus(config_name, expected):
    import os
    from pathlib import Path

    if not (Path(os.environ.get("MORPHEUS_PATH", "/nonexistent")) / "bin" / "morpheus").is_file():
        pytest.skip("Morpheus not installed (set MORPHEUS_PATH)")
    assert run(load_config(EXAMPLES / config_name, output={"dir": None})).metrics["accuracy"] == \
        pytest.approx(expected, abs=5e-4)
