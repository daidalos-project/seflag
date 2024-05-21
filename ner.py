import evaluate
import yaml
from datasets import load_dataset
from tqdm import tqdm
import la_core_web_lg
from flair.data import Sentence
from flair.models import SequenceTagger
from spacy import Language
from spacy.tokens import Doc


class Mappings:
    def __init__(self, source_file_path: str):
        with open(source_file_path) as f:
            mappings: dict[str, dict[str, int]] = yaml.safe_load(f)
            self.per_loc_misc: dict[str, int] = mappings["per_loc_misc"]
            self.per_loc_norp: dict[str, int] = mappings["per_loc_norp"]
            self.prs_geo_grp: dict[str, int] = mappings["prs_geo_grp"]


def annotate_greek_texts(words: list[str]) -> list[str]:
    """ Adds NER labels to Ancient Greek texts. """
    tagger: SequenceTagger = SequenceTagger.load("UGARIT/flair_grc_bert_ner")
    values: list[str] = []
    for word in tqdm(words):
        sentence: Sentence = Sentence(word)
        tagger.predict(sentence)
        values.append(sentence.labels[0].value if sentence.labels else "0")
    return values


def annotate_latin_texts(words: list[str]) -> list[str]:
    """Adds NER labels to Latin texts."""
    # to install this, make sure you run these lines once before:
    # !pip install spacy
    # !pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl
    nlp: Language = la_core_web_lg.load()
    values: list[str] = []
    for word in words:
        doc: Doc = nlp(word)
        values.append(doc[0].ent_type_)
    return values


def calculate_metrics(predictions, references):
    """ Calculates various metrics for the given predictions and references (i.e., ground truth). """
    precision = evaluate.load("precision")
    print("Weighted precision:")
    print(precision.compute(predictions=predictions, references=references, average="weighted"))
    print("Micro precision:")
    print(precision.compute(predictions=predictions, references=references, average="micro"))
    print("Macro precision:")
    print(precision.compute(predictions=predictions, references=references, average="macro"))

    recall = evaluate.load("recall")
    print("Weighted recall:")
    print(recall.compute(predictions=predictions, references=references, average="weighted"))
    print("Micro recall:")
    print(recall.compute(predictions=predictions, references=references, average="micro"))
    print("Macro recall:")
    print(recall.compute(predictions=predictions, references=references, average="macro"))

    f1 = evaluate.load("f1")
    print("Weighted f1:")
    print(f1.compute(predictions=predictions, references=references, average="weighted"))
    print("Micro f1:")
    print(f1.compute(predictions=predictions, references=references, average="micro"))
    print("Macro f1:")
    print(f1.compute(predictions=predictions, references=references, average="macro"))


def map_labels(original_labels, mapping):
    """ Applies a mapping to string labels, thereby converting them to integers. """
    labels_mapped = []
    for label in original_labels:
        for key in mapping:
            if (len(key) <= 1 and key == label) or (len(key) > 1 and key in label):
                labels_mapped.append(mapping[key])
    return labels_mapped


def run_evaluation(file_name: str, reference_column_name: str, word_column_name: str, annotation_fn: callable,
                   references_mapping: dict, predictions_mapping: dict):
    """ Performs evaluation of an NER model for the given dataset. """
    dataset_full = load_dataset("csv", split="train", data_files=file_name, sep="\t")
    print(dataset_full)
    # reduce dataset size for testing purposes
    dataset_selection = dataset_full.select(list(range(100)))
    print(dataset_selection)
    Reference = dataset_selection[reference_column_name]
    print(f"Lenght G_Reference: {len(Reference)}")
    references = map_labels(Reference, references_mapping)
    print(f"Length references: {len(references)}")
    words: list[str] = dataset_selection[word_column_name]
    ner_labels: list[str] = annotation_fn(words)
    print(ner_labels[0:20])
    print(len(ner_labels))
    print("Tagging done")
    predictions = map_labels(ner_labels, predictions_mapping)
    calculate_metrics(predictions, references)


mappings: Mappings = Mappings("mappings.yaml")
run_evaluation("dev.tsv", "entity", "word", annotate_greek_texts, mappings.per_loc_misc, mappings.per_loc_misc)
run_evaluation("GWtest.crf", "Label", "Word", annotate_latin_texts, mappings.prs_geo_grp, mappings.per_loc_norp)
