"""SEFLAG: Systematic Evaluation Framework for NLP models and datasets in Latin and Ancient Greek.

Typical use::

    from seflag import load_config, run

    result = run(load_config("my_evaluation.yaml"))
    print(result.metrics)

    matrix = run_matrix(load_matrix("my_matrix.yaml"))  # several datasets × models
    print(matrix.table())
"""

__version__ = "0.1.0.dev0"

from seflag.config import EvaluationConfig, load_config
from seflag.runner import EvaluationResult, run
from seflag.matrix import MatrixResult, load_matrix, run_matrix

__all__ = ["EvaluationConfig", "EvaluationResult", "MatrixResult", "__version__", "load_config", "load_matrix", "run",
           "run_matrix"]
