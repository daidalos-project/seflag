'''CODE TO EVALUATE POS-TAGGERS'''

'''IMPORT NEEDED MODULES'''

# Path handling #
import os
from config_v0_2 import Config # needed in github repository

# Encoding, Formatting and File Processing #
from Utility.pos_annotation_functions import annotate_latin_texts, annotate_greek_texts 
from Utility.process_conllu_functions import extract_content_of_conllu_fields
from Utility.process_nltk_functions import extract_text_from_nltk_dep_graphs_as_single_str
from Utility.visualisation_pos import visualisation

'''RUN SCRIPT'''

### Set up dataset ###

grc_folderpath = os.path.join(Config.lemmatization_dir, "grc")
lat_folderpath = os.path.join(Config.lemmatization_dir, "lat")
out_put_path = os.path.join(os.getcwd(), "evaluation_results")

### Prepare Goldstandard ###
pos_gold_data = []

for file in os.listdir(grc_folderpath):
    path = os.path.join(grc_folderpath, file)
    form_upos = extract_content_of_conllu_fields(path, 1, 3)
    for pair in form_upos:
        if pair not in pos_gold_data:
            pos_gold_data.append(pair)

### Prepare text for Annotation ###
text = []
for x in pos_gold_data:
    form = "".join(x[0])
    text.append(form)
final_text = " ".join(text)
print(final_text)

### Annotation ### 
annotations = annotate_latin_texts(final_text)

### Evaluation ###
print("Goldstandard POS:\n")
print(pos_gold_data)
print("\n")
print("Predicted POS:\n")
print(annotations)

gold_tags = [] # is used for the visualisation later
for token in pos_gold_data:
    gold_tags.append(token[1])

predicted_tags = [] # is used for the visualisation later
for token in annotations:
    predicted_tags.append(token[1])

print("Evaluation:\n\n")
def eval(goldstandard, predictions):
    index, correct = 0, 0
    for gold_token, predicted_token in zip(goldstandard, predictions):
        if gold_token == predicted_token:
            correct += 1
        if gold_token != predicted_token:
            continue
        index += 1
    print(f'Correctly predicted {correct}/{index}')
    accuracy = correct / index
    print(f"Accuracy: {accuracy}")

print("Evaluation Latincy:\n")    

eval(pos_gold_data, annotations)

visualisation(gold_tags, predicted_tags, final_text, model='la_core_web_lg', show_plot=False, show_confidence=False, export_table_csv="visualisation_outputs/latincy_pos_tabel.csv")

print("Evaluation Grecy:\n")
# grecy has to be installed and loaded

#eval(pos_gold_data, annotations)
#visualisation(gold_tags, predicted_tags, final_text, model='grc_proiel_trf', show_plot=False, show_confidence=False, export_table_csv="visualisation_outputs/grecy_pos_tabel.csv")

print("Evaluation lamonpy:\n")

path_to_predictions = os.path.join(os.getcwd(), "predictions/lamon_pos_predictions.json")
path_to_lamon_mapping = os.path.join(os.getcwd(), "POS/mappings/lamon_to_upos_mapping.json")

from Utility.process_json_functions import load_lamon_pred
from Utility.process_xml_functions import load_pos_mapping
from Utility.process_json_functions import convert_pos_tags

pred_dict = load_lamon_pred(path_to_predictions)
lamon_mappings = load_pos_mapping(path_to_lamon_mapping)

converted_pos_list = convert_pos_tags(pred_dict, lamon_mappings, out_put_path)
lamon_pred = []
for item in converted_pos_list:
    lamon_pred.append([item["lemma"], item["upos"]])

# Set up gold for lamon, because lamon uses only lemma not form #
pos_gold_data_lamon = extract_content_of_conllu_fields(first_five_path, 2, 3)
eval(pos_gold_data_lamon, lamon_pred)