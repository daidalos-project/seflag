"""The code examples of the package README must keep working."""

import re
from pathlib import Path

import pytest

from seflag import registry

README = Path(__file__).parent.parent / "README.md"


def python_blocks(heading: str | None = None) -> list[str]:
    text = README.read_text(encoding="utf-8")
    if heading is not None:
        text = text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"```python\n(.*?)```", text, re.S)


@pytest.mark.parametrize("code", python_blocks())
def test_examples_compile(code):
    compile(code, str(README), "exec")


def test_extension_example_runs(conllu_dir, monkeypatch, capsys):
    monkeypatch.setattr(registry, "_REGISTERED", {kind: {} for kind in registry.BUILTINS})
    code = python_blocks("Extending SEFLAG")[0].replace('"my_treebank/"', repr(str(conllu_dir)))
    code = code.replace('"language": "la", "task": "lemmatization",',
                        '"language": "la", "task": "lemmatization", "output": {"dir": None},')
    exec(compile(code, str(README), "exec"), {"__name__": "readme_example"})
    accuracy = float(capsys.readouterr().out.strip())
    assert 0 < accuracy < 1
    assert "lowercase" in registry.available("backends")
