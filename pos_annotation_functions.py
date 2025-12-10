'''DEFINE POS-ANNOTATION FUNCTIONS'''


# Import Spacy Model for Annotation #
# import la_core_web_lg
# import grc_proiel_trf
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

# Time Managment
import tqdm

from models import Models


### Latin Annotation Function for Spacy ###

# Funktion noch so überarbeiten, dass eine Datei als Argument reicht, indem die Lesefunktionen aufgerufen werden #
# Tagger Confidence kann noch eingebaut werden.#
def annotate_latin_texts(tokens: list[str]) -> tuple[list[list[str]]]:
    if not Models.pos_tagger_latin:
        Models.pos_tagger_latin = spacy.load("la_core_web_lg", exclude=["morphologizer", "parser", "ner"])
    doc: Doc = Models.pos_tagger_latin(Doc(vocab=Models.pos_tagger_latin.vocab, words=tokens))
    pos_anno = []
    # lemmata = []
    for token in doc:  # type: ignore
        pos_anno.append([token.text, token.pos_])
        #lemmata.append(token.lemma_)
    # pos_anno_str = ' '.join([' '.join(x) for x in pos_anno])
    return pos_anno, #lemmata

### Greek Annotation Function for Spacy ###
def annotate_greek_texts(str: str):
    nlp = spacy.load('grc_proiel_trf')
    doc = nlp(str)
    pos_anno = []
    for token in tqdm.tqdm(doc):
        pos_anno.append([token.text, token.pos_])
    # pos_anno_str = ' '.join([' '.join(x) for x in pos_anno])
    return pos_anno

### Process XML-Files ###
def process_papygreek(file_path, anno_version="", get_raw_text=False, token_index=False, form=False, lemma=False, pos=False, dep=False):
    doc = etree.parse(file_path)  # type: ignore
    root = doc.getroot()
    data = []
    tokens = []

    # Choose attribute suffix
    suffix = "_orig" if anno_version == "orig" else "_reg"

    for word in root.xpath("//word"):
        entry = {}

        token = word.get(f"form{suffix}")
        if not token:
            continue  # Skip if no token
        tokens.append(token)
        entry["token"] = token

        if pos:
            entry["pos"] = word.get(f"postag{suffix}")
        if lemma:
            entry["lemma"] = word.get(f"lemma{suffix}")
        if form:
            entry["form"] = token  # Already retrieved
        if dep:
            entry["id"] = word.get("id")
            entry["deprel"] = word.get(f"relation{suffix}")
            entry["head"] = word.get(f"head{suffix}")
        if token_index:
            entry["id"] = word.get("id")

        data.append(entry)

    raw_str = " ".join(tokens) if get_raw_text else None
    return data, raw_str