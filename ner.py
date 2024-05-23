import csv
import os.path

import evaluate
import yaml
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from evaluate import EvaluationModule
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
    for word in tqdm(words):
        doc: Doc = nlp(word)
        values.append(doc[0].ent_type_)
    return values


def calculate_metrics(predictions, references):
    """ Calculates various metrics for the given predictions and references (i.e., ground truth). """
    averages: list[str] = ["weighted", "micro", "macro"]
    metrics: list[str] = ["precision", "recall", "f1"]
    for metric in metrics:
        evaluation_module: EvaluationModule = evaluate.load(metric)
        for average in averages:
            print(average, evaluation_module.compute(predictions=predictions, references=references, average=average))


def map_labels(original_labels, mapping):
    """ Applies a mapping to string labels, thereby converting them to integers. """
    labels_mapped = []
    for label in original_labels:
        for key in mapping:
            if (len(key) <= 1 and key == label) or (len(key) > 1 and key in label):
                labels_mapped.append(mapping[key])
    return labels_mapped


def run_evaluation(folder_path: str, reference_column_name: str, word_column_name: str, annotation_fn: callable,
                   references_mapping: dict, predictions_mapping: dict):
    """ Performs evaluation of an NER model for the given dataset. """
    # disable quoting in Pandas because the dataset contains quotation marks as single tokens
    dataset_with_splits: DatasetDict = load_dataset("csv", data_dir=folder_path, sep="\t", quoting=csv.QUOTE_NONE)
    splits: list[str] = list(dataset_with_splits.shape.keys())
    dataset_combined: Dataset = concatenate_datasets([dataset_with_splits[x] for x in splits])
    # you can reduce dataset size for testing purposes
    dataset_selection = dataset_combined  # .select(list(range(100)))
    print(dataset_selection)
    references_raw = dataset_selection[reference_column_name]
    references_mapped = map_labels(references_raw, references_mapping)
    words: list[str] = dataset_selection[word_column_name]
    ner_labels: list[str] = annotation_fn(words)
    predictions = map_labels(ner_labels, predictions_mapping)
    calculate_metrics(predictions, references_mapped)


mappings: Mappings = Mappings("mappings.yaml")
data_dir: str = os.path.abspath("data")
greek_data_path: str = os.path.join(data_dir, "yousef_et_al_dataset")
latin_data_path: str = os.path.join(data_dir, "Herodotos_dataset")
run_evaluation(greek_data_path, "entity", "word", annotate_greek_texts, mappings.per_loc_misc, mappings.per_loc_misc)
run_evaluation(latin_data_path, "Label", "Word", annotate_latin_texts, mappings.prs_geo_grp, mappings.per_loc_norp)
