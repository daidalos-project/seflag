import os
import pyconll
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pos_annotation_functions import annotate_latin_texts, annotate_greek_texts

gold_path = os.path.join(os.getcwd(), "data/POS_Tagging", "dataset_complete.conllu")

# LOAD DATASET #
ds = pyconll.load.load_from_file(gold_path)

# PREPARE GOLDSTANDARD #
# REMOVE MULTIWORD TOKENS FOR EVALUATION #

pos_gold_data = []
multiword_token_list = []
for sentence in ds:
    for token in sentence:
        if token.is_multiword() == True:
            form = token.form
            upos = token.upos
            multiword_token_list.append([form, upos])
        else:
            form = token.form
            upos = token.upos
            pos_gold_data.append([form, upos])

print(f"Number of tokens in goldstandard after removal of multiword tokens: {len(pos_gold_data)}")
print(f"Number of multiword tokens excluded from evaluation: {len(multiword_token_list)}")

# PREPARE TEXT FOR ANNOTATION #
text_for_annotation = ""
for sentence in ds:
    text = sentence.text
    text_for_annotation = text_for_annotation + " " + text # type: ignore

# ANNOTATE TEXT #
predictions_tuple = annotate_latin_texts(text_for_annotation)
predictions = predictions_tuple[0]

print(f"Number of tokens in predictions: {len(predictions)}")

# FIND ABBREVIATED FORMS #
""" abbr_forms = []
for sentence in ds:
        sent_id = sentence.id
        for token in sentence:
            if 'Abbr' in token.feats.keys():
                form = token.form
                abbr_forms.append(form)

unique_abbr = set(abbr_forms) """

# {'mal.', 'ioan.', 'matth.', 'hab.', 'ier.', 'iac.', 'cor.', 'zach.', 'ethic.', 'num.', 
# 'sap.', 'rom.', 'nom.', 'metaph.', 'cap.', 'ult.', 'metaphys.', 'deut.', 'reg.', 'ierem.', 
# 'luc.', 'hebr.', 'gen.', 'heb.', 'eccli.', 'mich.', 'prou.', 'diu.'}

# HANDLE ALIGNMENT ERRORS DUE TO TOKENIZATION DIFFERENCES #
def align_tagged_sequences(pos_gold_data, predictions):
    # gold: list of [str, str]
    # pred: list of [str, str]
    # Returns: alignments (list of tuples; (operation, gold_span, pred_span))
    #    - operations: {"equal", "replace", "insert", "delete"}
   
    matcher = SequenceMatcher(None, [g[0] for g in pos_gold_data], [p[0] for p in predictions])
    
    alignments = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        gold_span = pos_gold_data[i1:i2]   # subset of gold sequence
        pred_span = predictions[j1:j2]   # subset of predicted sequence
        alignments.append((tag, gold_span, pred_span))

    return alignments

alignments = align_tagged_sequences(pos_gold_data, predictions)
            
# EVALUATE ANNOTATION #
def eval(alignments):
    index, correct = 0, 0
    for operation, gold_span, pred_span in alignments:
        if operation == "equal":
            for (gold_token, gold_pos), (predicted_token, predicted_pos) in zip(gold_span, pred_span):
                index += 1
                if gold_pos == predicted_pos:
                    correct += 1
        
        elif operation == "replace":
            gold_normalised = [g[0].rstrip(".") for g in gold_span] # Tokenisation errors due to punctuation
            pred_normalised = [p[0].rstrip(".") for p in pred_span] # Tokenisation errors due to punctuation
            if gold_normalised == pred_normalised:
                index += len(gold_span)
                correct += len(gold_span)
            else:
                index += len(gold_span)
        elif operation in ("insert", "delete"):
            total += len(gold_span)
    correct_out_of_idx = f'Correctly predicted {correct}/{index}'
    accuracy = correct / index if index else 0.0
    return accuracy, correct_out_of_idx

print("Evaluation Latincy:\n")

accuracy, correct_out_of_idx = eval(alignments)

print(correct_out_of_idx)
print(f"Accuracy: {accuracy}")

# VISUALISATION OF EVALUATION #
def collect_tag_pairs(alignments):
    # Visualisation in form of a confusion matrix has to be based on the alignment of gold data and predicted data
    gold_tags, pred_tags = [], []

    for op, gold_span, pred_span in alignments:
        if op == "equal":
            for (g_tok, g_tag), (p_tok, p_tag) in zip(gold_span, pred_span):
                gold_tags.append(g_tag)
                pred_tags.append(p_tag)

        elif op == "replace":
            gold_norm = [g[0].rstrip(".") for g in gold_span]
            pred_norm = [p[0].rstrip(".") for p in pred_span]
            if gold_norm == pred_norm and len(pred_span) > 0:
                gold_tags.append(gold_span[0][1])
                pred_tags.append(pred_span[0][1])
            else:
                gold_tags.append(gold_span[0][1])
                pred_tags.append(pred_span[0][1] if pred_span else "MISSING")


    return gold_tags, pred_tags

gold_tags, pred_tags = collect_tag_pairs(alignments)

cm = confusion_matrix(gold_tags, pred_tags, labels=sorted(set(gold_tags + pred_tags)))

fig, ax = plt.subplots(figsize=(14, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=sorted(set(gold_tags + pred_tags)))
disp.plot(cmap="Greens", values_format="d", ax=ax, colorbar=False)
plt.title("POS Tag Confusion Matrix")
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)
plt.show()