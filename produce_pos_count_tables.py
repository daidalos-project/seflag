'''This Code produces two separate .csv-tables with 
pos-statistics for the .conllu- corpus and the .xml-corpus.'''

import pyconll
import pandas as pd
import os
import csv
from collections import Counter

ds_path = os.path.join(os.getcwd(), "data/POS_Tagging/gr_dataset_complete.conllu")
ds = pyconll.load_from_file(ds_path) # type: ignore

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

with open("pos_count.txt", "w", encoding="utf-8") as f:
    for form, data in conllu_data.items():  # form + data has to be extracted to access dict content
        lemma_list = data['lemma']
        upos_list = data['upos']
        feats_list = data['feats']
        combined_list = data['combined_anno']

        lemma_counts = Counter(lemma_list)
        upos_counts = Counter(upos_list)
        feats_counts = Counter(feats_list)
        combined_anno_counts = Counter(combined_list)

        print(
            f"form: {form}\n"
            f"lemma count: {lemma_counts}\n"
            f"upos count: {upos_counts}\n"
            f"feats count: {feats_counts}\n"
            f"combined annotations count: {combined_anno_counts}\n",
            file=f
        )

with open("pos_count.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")

    writer.writerow(["form", "lemma_counts", "upos_counts", "feats_counts", "combined_annotation_counts"])

    for form, data in conllu_data.items():
        lemma_counts = Counter(data['lemma'])
        upos_counts = Counter(data['upos'])
        feats_counts = Counter(data['feats'])
        combined_anno_counts = Counter(data['combined_anno'])


        writer.writerow([
            form,
            dict(lemma_counts),
            dict(upos_counts),
            dict(feats_counts),
            dict(combined_anno_counts)
        ])

## PARS PERTINENS PAPYGREEK ##
from pos_annotation_functions import process_papygreek
csv_file_path = os.path.join(os.getcwd(), "cross_annatation_scheme_eval/agdtb_pos_statistik.csv")

xml_ds = os.path.join(os.getcwd(), "data/POS_Tagging")

posdict = {}
all_pos_tags = set()

for file in os.listdir(xml_ds):
    if not file.endswith(".xml"): 
        continue
    file_path = os.path.join(xml_ds, file)
    data, raw_string = process_papygreek(file_path, anno_version="", get_raw_text=True, form=True, lemma=True, pos=True)
    
    for entry in data:
        lemma = entry["lemma"]
        pos = str(entry["pos"])

        if lemma not in posdict:
            posdict[lemma] = []

        posdict[lemma].append(pos)

        all_pos_tags.add(pos)
    
csv_header = ["lemma"] + sorted(all_pos_tags)

# Make csv
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

    for lemma, pos_list in posdict.items():
        counts = Counter(pos_list)
        row = [lemma] + [counts.get(pos, 0) for pos in sorted(all_pos_tags)]
        writer.writerow(row)

# Make both .csv-tables directly comparable in long format #

df = pd.read_csv(csv_file_path, encoding="utf-8") # takes the agdtb_pos_statistik.csv

df_long = df.melt(
    id_vars=["lemma"],
    var_name="Feature",
    value_name="Count"
)
df_long = df_long[df_long["Count"] > 0]
df_long.to_csv("agdtb_pos_statistik_long.csv", index=False, encoding="utf-8")