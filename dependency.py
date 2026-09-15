'''IMPORT NEEDED MODULES'''

# Path handling #
from pathlib import Path

# Time Management #
import tqdm as tqdm

# Processing and reading conllu files #
from pyconll.conllu import conllu

# Annotation #
import spacy
la_nlp = spacy.load('la_core_web_lg')
# grc_nlp = spacy.load('MODEL')

# Evaluation #
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.tokens import Doc

'''SET UP PATHS'''

#cwd
cwd = Path.cwd()

# conllu data (UD Treebanks v2.15) is stored in:
conllu_data_dir = cwd / "data/conllu_data"
la_files = list(conllu_data_dir.glob("la*.conllu"))
grc_files = list(conllu_data_dir.glob("grc*.conllu"))

# goldstandard will be saved to:
la_gstd_path = cwd / "gold_standards/la_gstd.conllu"
grc_gstd_path = cwd / "gold_standards/grc_gstd.conllu"

'''CREATE GOLDSTANDARDS'''

""" for file in la_files:
    filename = os.path.join(conllu_data_path, file)
    sentences = conllu.load_from_file(filename)
    with open(la_gstd_path, 'a+', encoding='utf-8') as f:
        conllu.write_corpus(sentences, f) """

la_corpus = conllu.load_from_file(la_gstd_path)

la_gold_forms = []
la_gold_deps = []
la_gold_heads = []

for sent in la_corpus:
    for token in sent.tokens:
        if token.is_multiword() == True:
            continue
        else:
            la_gold_forms.append(token.form)
            la_gold_deps.append(token.deprel)
            la_gold_heads.append(token.head)

la_gold_heads = [int(head) for head in la_gold_heads]

""" for file in grc_files:
    filename = os.path.join(conllu_data_path, file)
    sentences = conllu.load_from_file(filename)
    with open(grc_gstd_path, 'a+', encoding='utf-8') as f:
        conllu.write_corpus(sentences, f)
     """
# Könnte man in einer Funktion zusammenfassen

# grc_corpus = conllu.load_from_file(grc_gstd_path)

'''NORMALIZE GOLD STANDARD'''

la_gold_forms = [form.replace("v", "u") for form in la_gold_forms]
la_gold_forms = [form.replace("V", "U") for form in la_gold_forms]

'''SET UP ANNOTATION FUNCTIONS'''

""" def annotate_latin_texts(text: str):
    pred_text = []
    pred_deps = []
    doc = la_nlp(text)
    for token in doc:
        pred_text.append(token.text)
        pred_deps.append(token.dep_)
    return pred_text, pred_deps """

def annotate_latin_tokens(tokens: list[str]):
    pred_text = []
    pred_deps = []
    pred_heads = []
    nlp = spacy.load("la_core_web_lg")
    doc = Doc(nlp.vocab, words=tokens)
    doc = nlp(doc)
    for token in doc:
        pred_text.append(token.text)
        pred_deps.append(token.dep_)
        pred_heads.append(token.head.i)
    return pred_text, pred_deps, pred_heads

'''SET UP TEXT'''

# sentences

# for sentence in la_corpus:
#    text = sentence.meta['text']
#    sentences.append(text)

# string_for_annotation = " ".join(sentences)

'''RUN ANNOTATION'''

pred_text, pred_deps, pred_heads = annotate_latin_tokens(la_gold_forms)

'''RUN EVALUATION'''

predicted = Doc(la_nlp.vocab, words=pred_text, heads=pred_heads, deps=pred_deps)
reference = Doc(la_nlp.vocab, words=la_gold_forms, heads=la_gold_heads, deps=la_gold_deps)

ex_obj = []

ex_obj.append(Example(predicted, reference))

scorer = Scorer(la_nlp)

scores = scorer.score(ex_obj)

print(scores["dep_uas"], scores["dep_las"])