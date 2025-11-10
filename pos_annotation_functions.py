'''DEFINE POS-ANNOTATION FUNCTIONS'''


# Import Spacy Model for Annotation #
# import la_core_web_lg
# import grc_proiel_trf
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

# Time Managment
import tqdm

### Latin Annotation Function for Spacy ###

# Funktion noch so überarbeiten, dass eine Datei als Argument reicht, indem die Lesefunktionen aufgerufen werden #
# Tagger Confidence kann noch eingebaut werden.#
def annotate_latin_texts(str: str):
    nlp = spacy.load('la_core_web_lg')
    doc = nlp(str)
    pos_anno = []
    # lemmata = []
    for token in tqdm.tqdm(doc): # type: ignore
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