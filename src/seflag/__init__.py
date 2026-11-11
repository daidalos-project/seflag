"""SEFLAG: Systematic Evaluation Framework for NLP models and datasets in Latin and Ancient Greek.

Typical use::

    from seflag import load_config, run

    result = run(load_config("my_evaluation.yaml"))
    print(result.metrics)
"""

__version__ = "0.1.0.dev0"

from seflag.config import EvaluationConfig, load_config
from seflag.runner import EvaluationResult, run

__all__ = ["EvaluationConfig", "EvaluationResult", "__version__", "load_config", "run"]
