import evaluate
from evaluate import EvaluationModule


def accuracy(y_pred: list[int], y_true: list[int]):
    """ Calculates the accuracy of the predicted results against the ground truth. """
    evaluation_module: EvaluationModule = evaluate.load("accuracy")
    print(evaluation_module.compute(references=y_true, predictions=y_pred))


def accuracy_for_strings(y_pred: list[str], y_true: list[str]):
    """ Calculates the accuracy of the predicted results against the ground truth. """
    evaluation_module: EvaluationModule = evaluate.load("accuracy")
    distinct_values: set[str] = set(y_pred + y_true)
    label_to_int: dict[str, int] = {value: idx for idx, value in enumerate(distinct_values)}
    print(accuracy([label_to_int[x] for x in y_pred], [label_to_int[x] for x in y_true]))


def precision_recall_f1(predictions: list[int], references: list[int]):
    """ Calculates various metrics for the given predicted and true labels. """
    averages: list[str] = ["weighted", "micro", "macro"]
    metrics: list[str] = ["precision", "recall", "f1"]
    for metric in metrics:
        evaluation_module: EvaluationModule = evaluate.load(metric)
        for average in averages:
            print(average,
                  evaluation_module.compute(predictions=predictions, references=references, average=average))
