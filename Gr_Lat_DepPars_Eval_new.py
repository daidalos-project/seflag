'''IMPORT NEEDED MODULES''' 

# Path handling #
import os
import os.path

# Time Management #
import tqdm as tqdm

# Processing and Reading files #
from Utility.dep_annotation_functions import annotate_greek_texts, annotate_latin_texts, convert_annot_to_conllu

from Utility.process_conllu_functions import save_conllu_file, read_first_five, save_conllu_dir_as_file, conllu_to_nltk_dependency_graph

from Utility.process_nltk_functions import extract_text_from_nltk_dependency_graphs, extract_indv_words_from_nltk_dep_graphs, extract_text_from_nltk_dep_graphs_as_single_str, extract_list_of_indv_words_from_list_of_sents

from Dependency_Parsing.conll18_ud_eval import load_conllu, evaluate, load_conllu_file

from Utility.visualisation_dep import tabulate_dep_eval

# Saving Results #
import json

'''SET UP GOLD STANDARD'''

current_directory = os.getcwd()
folder_path = os.path.join(current_directory, "data/conllu_data")
directory_path = os.path.join(current_directory, "gold_standards")
dataset = save_conllu_dir_as_file(folder_path, directory_path) # Creates file called "dataset_complete.conllu"
graphs= conllu_to_nltk_dependency_graph(dataset)
text = extract_text_from_nltk_dependency_graphs(graphs)
indv_words = extract_indv_words_from_nltk_dep_graphs(graphs)
indv_words_2 = extract_list_of_indv_words_from_list_of_sents(text)

'''ANNOTATION'''

annotations = [annotate_latin_texts(text_list) for text_list in tqdm.tqdm(indv_words_2, desc="Annotating data")] # type: ignore
# annotations = [annotate_greek_texts(text_list) for text_list in tqdm(indv_words_2, desc="Annotating data")]
# print(annotations)
parsed_to_conllu = convert_annot_to_conllu(annotations)

prediction_folder_path = os.path.join(current_directory, "predictions")
if not os.path.exists(prediction_folder_path):
        os.makedirs(prediction_folder_path)
prediction_file_path = os.path.join(prediction_folder_path, "latincy_predictions_.conllu")

print(f'Saving predictions to {prediction_file_path}')

save_conllu_file(prediction_file_path, parsed_to_conllu)

'''EVAL'''

pred = load_conllu_file(prediction_file_path)
gold = load_conllu_file(dataset)

eval = evaluate(gold, pred)
eval_path = os.path.join(current_directory, "visualisation_outputs/conll18_ud_eval_output.json")

import json

# Conversion is neccessary because the ud eval script uses costum python types
def serialize_scores(score_dict):
    return {k: v.__dict__ for k, v in score_dict.items()}

# Save
with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(serialize_scores(eval), f, ensure_ascii=False, indent=2)

print(f"Evaluation saved to {eval_path}")

tabulate_dep_eval(eval_path)