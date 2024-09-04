import os.path
import subprocess
import betacode.conv
import conllu
import spacy
from conllu import SentenceList
from dotenv import load_dotenv
from spacy.tokens import Doc
from tqdm import tqdm
import xml.etree.ElementTree as ET
from config import Config
from metrics import accuracy

from models import Models

# need this for greCy to work properly
Doc.set_extension("trf_data", default=None)
beta_to_uni: dict[str, str] = dict()
uni_to_beta: dict[str, str] = dict()


def convert_labels(lemmata_predicted: list[str], lemmata_true: list[str]) -> tuple[list[int], list[int]]:
    """ Converts lemmatization results from strings to integers, which is useful for calculating metrics. """
    all_lemmata: set[str] = set(lemmata_predicted + lemmata_true)
    lemma_to_idx: dict[str, int] = {lemma: idx for idx, lemma in enumerate(all_lemmata)}
    predictions_int: list[int] = [lemma_to_idx[x] for x in lemmata_predicted]
    references_int: list[int] = [lemma_to_idx[x] for x in lemmata_true]
    return predictions_int, references_int


def lemmatize_greek(tokens: list[str]) -> list[str]:
    """ Lemmatizes Ancient Greek tokens using spaCy. """
    if not Models.lemmatizer_greek:
        Models.lemmatizer_greek = spacy.load(
            "grc_proiel_lg",  # grc_proiel_trf grc_odycy_joint_trf
            exclude=["morphologizer", "parser", "tagger", "transformer"],  #
        )
    doc: Doc = Models.lemmatizer_greek(Doc(vocab=Models.lemmatizer_greek.vocab, words=tokens))
    return [x.lemma_ for x in doc]


def lemmatize_latin(tokens: list[str]) -> list[str]:
    """Lemmatizes Latin tokens using spaCy."""
    if not Models.lemmatizer_latin:
        Models.lemmatizer_latin = spacy.load(
            'la_core_web_lg', exclude=["morphologizer", "parser", "tagger", "ner"])
    doc: Doc = Models.lemmatizer_latin(Doc(vocab=Models.lemmatizer_latin.vocab, words=tokens))
    return [x.lemma_ for x in doc]


def morpheus(text: str) -> list[str]:
    """ Runs Morpheus and uses it to lemmatize a given word form. """
    if text not in uni_to_beta:
        # Morpheus only accepts beta code; need to replace special sigma notation with ordinary one
        uni_to_beta[text] = betacode.conv.uni_to_beta(text).replace("s1", "s")
    # make sure that MORPHEUS_PATH is present in your .env file and points to the correct folder
    # see https://github.com/perseids-tools/morpheus-perseids for installation instructions
    load_dotenv()
    env: dict = os.environ.copy()
    env["MORPHLIB"] = "stemlib"
    cp: subprocess.CompletedProcess = subprocess.run(
        ["bin/morpheus", uni_to_beta[text]], capture_output=True, env=env, cwd=env["MORPHEUS_PATH"])
    output: bytes = cp.stdout
    xml: str = output.decode("utf-8")
    root: ET.Element = ET.fromstring(xml)
    headwords: list[ET.Element] = root.findall(".//hdwd")
    lemmata: list[str] = [x.text for x in headwords]
    for lemma in lemmata:
        if lemma not in beta_to_uni:
            beta_to_uni[lemma] = betacode.conv.beta_to_uni(lemma)
    return [beta_to_uni[x] for x in lemmata]


def run_evaluation(lemmatization_fn: callable, data_dir: str):
    """ Performs evaluation of a lemmatization model for the given dataset. """
    sl: SentenceList = SentenceList()
    for file in [x for x in os.listdir(data_dir) if x.endswith(".conllu")]:
        file_path: str = os.path.join(data_dir, file)
        with open(file_path, 'r') as f:
            new_sl: SentenceList = conllu.parse(f.read())
            sl += new_sl
    lemmata_predicted: list[str] = []
    lemmata_true: list[str] = []
    for sent in tqdm(sl):
        words: list[str] = [tok["form"] for tok in sent]
        new_lemmata_true: list[str] = [tok["lemma"] for tok in sent]
        lemmata_true += new_lemmata_true
        lemmata_predicted += lemmatization_fn(words)
    print("Errors: ", [(lemmata_predicted[i], lemmata_true[i]) for i in range(len(lemmata_true)) if
                       lemmata_predicted[i] != lemmata_true[i]])
    predictions_int, references_int = convert_labels(lemmata_predicted, lemmata_true)
    accuracy(predictions_int, references_int)


# run_evaluation(lemmatize_greek, os.path.join(Config.lemmatization_dir, "greek"))
# run_evaluation(lemmatize_latin, os.path.join(Config.lemmatization_dir, "latin"))
