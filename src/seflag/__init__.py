"""SEFLAG: Systematic Evaluation Framework for NLP models and datasets in Latin and Ancient Greek.

Typical use::

    from pathlib import Path

    from seflag import (EvaluationResult, load_config, load_dataset, load_matrix, run, run_matrix,
                        table_from_results)

    result = run(load_config("my_evaluation.yaml"))
    print(result.metrics)

    matrix = run_matrix(load_matrix("my_matrix.yaml"))  # several datasets × models
    print(matrix.table())

    dataset = load_dataset("my_evaluation.yaml")         # SEFLAG's data access on its own
    docs = dataset.docs()                                # gold sentences as spaCy Docs

    results = [EvaluationResult.load(p) for p in Path("seflag_results").glob("*.json")]
    print(table_from_results(results).table())           # Markdown table of earlier runs
"""

__version__ = "1.1.0"

from seflag.config import EvaluationConfig, load_config
from seflag.runner import EvaluationResult, run
from seflag.matrix import MatrixResult, load_matrix, run_matrix, table_from_results
from seflag.components import load_backend, load_dataset
from seflag.registry import register

__all__ = ["EvaluationConfig", "EvaluationResult", "MatrixResult", "__version__", "load_backend", "load_config",
           "load_dataset", "load_matrix", "register", "run", "run_matrix", "table_from_results"]
