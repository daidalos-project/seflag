### Currently set up to extract Annotation for every Wordform ###
### For further evaluation a lemma based approach might be better ###


import pyconll
import os
import csv
from collections import Counter
from pos_annotation_functions import process_papygreek

## CORPUS TEXTORUM CONLL-U ##

ds_path = os.path.join(os.getcwd(), "data/Dependency_Parsing/grc_ud_test_dataset_for_use_of_SEFLAG/grc_perseus-ud-test.conllu")
ds = pyconll.load_from_file(ds_path) # type: ignore

## CORPUS TEXTORUM XML ##

xml_ds = os.path.join(os.getcwd(), "data/xml_data/papygreek-treebanks-v3.0/documentary/bgu")

## sēmita ad csv

morph_anno_csv_file_path = os.path.join(os.getcwd(), "cross_annotation_scheme_eval/cross_corpus_morph_anno_stat.csv")
pos_csv_file_path = os.path.join(os.getcwd(), "cross_annotation_scheme_eval/cross_corpus_pos_stat.csv")
lemma_csv_file_path = os.path.join(os.getcwd(), "cross_annotation_scheme_eval/cross_corpus_lemma_stat.csv")

## PARS PERTINENS AD CONLL-U ##

conllu_data = {}

for sentence in ds:
    for word in sentence:
        if not word.is_multiword():
            form = word.form
            lemma = word.lemma
            upos = word.upos
            feats = str(word.feats) if word.feats is not None else "_" # handle emppty feats column
            if form not in conllu_data:
                conllu_data[form] = {
                    'lemma': [lemma],
                    'upos': [upos],
                    'feats': [feats],
                    'combined_anno': [(lemma, upos, feats)] # has to be tuple for Counter to work
                }
            else:
                conllu_data[form]['lemma'].append(lemma)
                conllu_data[form]['upos'].append(upos)
                conllu_data[form]['feats'].append(feats)
                conllu_data[form]['combined_anno'].append((lemma, upos, feats))

## PARS PERTINENS PAPYGREEK ##

xml_data = {}

for file in os.listdir(xml_ds):
    if not file.endswith(".xml"): 
        continue
    file_path = os.path.join(xml_ds, file)
    data, raw_string = process_papygreek(file_path, anno_version="", get_raw_text=False, form=True, lemma=True, pos=True)
    
    for entry in data:
        form = entry["form"]
        lemma = entry["lemma"]
        pos = str(entry["pos"])

        if form not in xml_data:
            xml_data[form] = {}
            xml_data[form]['pos'] = []
            xml_data[form]['lemma'] = []

        xml_data[form]['pos'].append(pos)
        xml_data[form]['lemma'].append(lemma)

## PARS PERTINENS CSV ##

# Create csv for whole morphological annotation
with open(morph_anno_csv_file_path, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")

    writer.writerow(["form", "source", "annotation", "count"])

    for form, data in conllu_data.items():
        source = "conllu"
        combined_anno_counts = Counter(data['combined_anno'])
        for comb_anno in combined_anno_counts:
            annotation = comb_anno
            count = combined_anno_counts[comb_anno]

            writer.writerow([
                form,
                source,
                annotation,
                count
            ])
    
    for form, annotations in xml_data.items():
        source = "xml"
        combined_anno_counts = Counter(annotations['pos'])
        for comb_anno in combined_anno_counts:
            annotation = comb_anno
            count = combined_anno_counts[comb_anno]

            writer.writerow([
                form,
                source,
                annotation,
                count
            ])

# Create csv for pos annotation
with open(pos_csv_file_path, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")

    writer.writerow(["form", "source", "annotation", "count"])

    for form, data in conllu_data.items():
        source = "conllu"
        pos_counts = Counter(data['upos'])
        for pos_anno in pos_counts:
            annotation = pos_anno
            count = pos_counts[pos_anno]

            writer.writerow([
                form,
                source,
                annotation,
                count
            ])
    
    for form, annotations in xml_data.items():
        source = "xml"
        combined_anno_counts = Counter(annotations['pos'])
        for comb_anno in combined_anno_counts:
            annotation = comb_anno[:1]
            count = combined_anno_counts[comb_anno]

            writer.writerow([
                form,
                source,
                annotation,
                count
            ])

# Create csv for lemma annotation
with open(lemma_csv_file_path, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")

    writer.writerow(["form", "source", "annotation", "count"])

    for form, data in conllu_data.items():
        source = "conllu"
        lemma_counts = Counter(data['lemma'])
        for lemma in lemma_counts:
            annotation = lemma
            count = lemma_counts[lemma]

            writer.writerow([
                form,
                source,
                annotation,
                count
            ])
    
    for form, annotations in xml_data.items():
        source = "xml"
        lemma_counts = Counter(annotations['lemma'])
        for lemma in lemma_counts:
            annotation = lemma
            count = lemma_counts[lemma]

            writer.writerow([
                form,
                source,
                annotation,
                count
            ])