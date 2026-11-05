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
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer
import pandas as pd
from tabulate import tabulate

'''SET UP PATHS'''

#cwd
cwd = Path.cwd()

# conllu data is stored in:
conllu_data_dir = cwd / "data/conllu_data"
la_files = list(conllu_data_dir.glob("la*.conllu"))
grc_files = list(conllu_data_dir.glob("grc*.conllu"))

# goldstandard will be stored in:
la_gstd_path = cwd / "gold_standards/la_gstd.conllu"
grc_gstd_path = cwd / "gold_standards/grc_gstd.conllu"

'''CREATE GOLDSTANDARDS'''

""" for file in la_files:
    filename = os.path.join(conllu_data_path, file)
    sentences = conllu.load_from_file(filename)
    with open(la_gstd_path, 'a+', encoding='utf-8') as f:
        conllu.write_corpus(sentences, f) """

""" for file in grc_files:
    filename = os.path.join(conllu_data_path, file)
    sentences = conllu.load_from_file(filename)
    with open(grc_gstd_path, 'a+', encoding='utf-8') as f:
        conllu.write_corpus(sentences, f)
     """
# Könnte man in einer Funktion zusammenfassen

# grc_corpus = conllu.load_from_file(grc_gstd_path)

def evaluate_spacy(corpus_path, nlp):
    corpus = conllu.load_from_file(corpus_path)
    examples = []

    for sent in corpus:
        tokens = [token for token in sent.tokens if not token.is_multiword()]
        forms = [token.form for token in tokens]
        forms = [f.replace("v", "u").replace("V", "U") for f in forms]
        deps = [token.deprel for token in tokens]
        conllu_heads = [int(token.head) for token in tokens]

        # Head conversion
        spacy_heads = []
        for i, ch in enumerate(conllu_heads):
            if ch == 0:
                spacy_heads.append(i)
            else:
                spacy_heads.append(ch - 1)
        
        doc = Doc(nlp.vocab, words=forms)
        doc = nlp(doc)

        ref = Doc(nlp.vocab, words=forms, heads=spacy_heads, deps=deps)

        examples.append(Example(doc, ref))

    return examples

examples = evaluate_spacy(la_gstd_path, la_nlp)

score = Scorer(la_nlp).score(examples)

df = pd.json_normalize(score).T # .T transposes dataframe
df.columns = ["score"]

table = tabulate(df, headers="keys", tablefmt="pipe")

with open("table.md", "w") as file:
    file.write(table)