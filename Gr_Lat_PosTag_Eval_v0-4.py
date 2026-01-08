'''Currently set up for Latin. To evaluate to Greek, uncomment the code dealing with the greek data'''

# IMPORT NEEDED MODULES #

import os # path handling
import pyconll # used to process .conllu-files
import matplotlib.pyplot as plt # used in visualisation step
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # used in visualisation step
from tqdm import tqdm # time management

# IMPORT EVALUATION METRIC #

from metrics import accuracy_for_strings

# IMPORT ANNOTION FUNCTIONS #

from pos_annotation_functions import annotate_latin_texts, annotate_greek_texts

# LOAD DATASET #

lat_gold_path = os.path.join(os.getcwd(), "data", "lat_dataset_complete.conllu")
lat_ds = pyconll.load.load_from_file(lat_gold_path)
""" grc_gold_path = os.path.join(os.getcwd(), "data", "lat_dataset_complete.conllu")
grc_ds = pyconll.load.load_from_file(grc_gold_path) """

# PREPARE GOLDSTANDARDS #
# Code removes multiword tokens for the purpose of evaluation. #
# Wordform and POS-Tag are appended to a list (pos_gold_data_lat) #

pos_gold_data_lat = []
multiword_token_list_lat = []
for sentence in lat_ds:
    sent_tokens = []
    for token in sentence:
        form = token.form
        upos = token.upos
        if token.is_multiword() == True:
            multiword_token_list_lat.append([form, upos])
        else:
            sent_tokens.append([form, upos])
    pos_gold_data_lat.append(sent_tokens)

print(f"Number of tokens in latin goldstandard after removal of multiword tokens: {len(pos_gold_data_lat)}")
print(f"Number of latin multiword tokens excluded from evaluation: {len(multiword_token_list_lat)}")

""" pos_gold_data_grc = []
multiword_token_list_grc = []
for sentence in grc_ds:
    sent_tokens = []
    for token in sentence:
        form = token.form
        upos = token.upos
        if token.is_multiword() == True:
            multiword_token_list_lat.append([form, upos])
        else:
            sent_tokens.append([form, upos])
    pos_gold_data_grc.append(sent_tokens)

print(f"Number of tokens in greek goldstandard after removal of multiword tokens: {len(pos_gold_data_grc)}")
print(f"Number of greek multiword tokens excluded from evaluation: {len(multiword_token_list_grc)}") """

# PREPARE AND ANNOTATE TEXT #
# Code iterates over goldstandard producing progress bar. # 
# The annotation function is called on the wordforms in the pos_gold_data list #
 
predictions_lat = []
for sent_token in tqdm(pos_gold_data_lat):
    tokens = [x[0] for x in sent_token]
    predictions_lat += annotate_latin_texts(tokens)

print(f"Number of tokens in predictions_lat: {len(predictions_lat)}")

""" predictions_grc = []
for sent_token in tqdm(pos_gold_data_grc):
    tokens = [x[0] for x in sent_token]
    predictions_grc += annotate_greek_texts(tokens)

print(f"Number of tokens in predictions_lat: {len(predictions_grc)}") """

# EVALUATE ANNOTATION #
# Code takes list of predictions_lat and pos_gold_data_lat list and produces accuracy metric. #

print("Evaluation Latincy:\n")

accuracy_for_strings([tok[1] for sent in pos_gold_data_lat for tok in sent], [x[1] for x in predictions_lat])

""" print("Evaluation Grecy:\n")

accuracy_for_strings([tok[1] for sent in pos_gold_data_grc for tok in sent], [x[1] for x in predictions_grc]) """

# VISUALISATION OF LATIN EVALUATION #

gold_tags = [tok[1] for sent in pos_gold_data_lat for tok in sent] # list of gold POS-tags
pred_tags = [x[1] for x in predictions_lat] # list of predicted POS-tags
cm = confusion_matrix(gold_tags, pred_tags, labels=sorted(set(gold_tags + pred_tags))) # CM using all unique tags as lables

fig, ax = plt.subplots(figsize=(14, 7)) # Creates figure with specified size
disp = ConfusionMatrixDisplay(confusion_matrix=cm, # Create CM with sklearn
                              display_labels=sorted(set(gold_tags + pred_tags)))
disp.plot(cmap="Greens", values_format="d", ax=ax, colorbar=False)
plt.title("POS Tag Confusion Matrix")
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)
plt.show()

# VISUALISATION OF GREEK EVALUATION #

""" gold_tags = [tok[1] for sent in pos_gold_data_grc for tok in sent] # list of gold POS-tags
pred_tags = [x[1] for x in predictions_grc] # list of predicted POS-tags
cm = confusion_matrix(gold_tags, pred_tags, labels=sorted(set(gold_tags + pred_tags))) # CM using all unique tags as lables

fig, ax = plt.subplots(figsize=(14, 7)) # Creates figure with specified size
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=sorted(set(gold_tags + pred_tags)))
disp.plot(cmap="Greens", values_format="d", ax=ax, colorbar=False)
plt.title("POS Tag Confusion Matrix")
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)
plt.show() """