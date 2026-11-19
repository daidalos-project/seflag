"""The code examples of the package README must keep working."""

import argparse
import re
import shlex
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

import pytest
import yaml
from packaging.utils import canonicalize_name, parse_wheel_filename

from seflag import load_config, load_matrix, registry
from seflag.backends.spacy import KNOWN_MODELS
from seflag.cli import build_parser
from seflag.matrix import is_matrix

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
    code = code.replace('"language": Language.LATIN,', '"language": Language.LATIN, "output": {"dir": None},')
    exec(compile(code, str(README), "exec"), {"__name__": "readme_example"})
    accuracy = float(capsys.readouterr().out.strip())
    assert 0 < accuracy < 1
    assert "lowercase" in registry.available("backends")


def test_typed_configuration_example_is_valid():
    code = python_blocks("Python API")[0].replace("result = seflag.run(config)", "")
    namespace: dict = {}
    exec(compile(code, str(README), "exec"), namespace)
    assert namespace["config"].model_dump(mode="json")["options"]["label_normalization"] == ["v_to_u", "j_to_i"]


# --- the other examples: shell commands, configurations, entry points ---------------------------------------------

def blocks(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def shell_commands() -> list[list[str]]:
    return [words for block in blocks("bash") for line in block.splitlines()
            if (words := shlex.split(line, comments=True))]


def wheel_urls(commands) -> list[str]:
    return [word for command in commands for word in command if word.startswith("https://")]


def test_model_install_commands_are_the_ones_seflag_prints():
    readme = [" ".join(words) for words in shell_commands() if "https://" in " ".join(words)]
    assert readme, "the README shows how to install a model"
    assert set(readme) <= set(KNOWN_MODELS.values())


@pytest.mark.parametrize("name, command", KNOWN_MODELS.items())
def test_model_wheels_are_installable_and_pinned(name, command):
    (url,) = wheel_urls([shlex.split(command)])
    # pip rejects wheel names without a concrete version, e.g. la_core_web_lg-any-py3-none-any.whl
    distribution, version, _, _ = parse_wheel_filename(url.rsplit("/", 1)[1])
    assert canonicalize_name(distribution) == canonicalize_name(name) and version.release
    # main branches and "latest" releases change; old wheels have been deleted from them before
    assert not re.search(r"/(main|master|latest)/", url)


@pytest.mark.network
@pytest.mark.parametrize("url", wheel_urls(shlex.split(c) for c in KNOWN_MODELS.values()))
def test_model_wheels_exist(url):
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})  # kept on redirects: no full download
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status in (200, 206)
    except urllib.error.HTTPError as e:
        if e.code in (404, 410):
            raise AssertionError(f"{url} no longer exists") from e
        pytest.skip(f"cannot check {url}: {e}")
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"no network: {e}")


def test_extras_exist():
    pyproject = README.parents[2] / "pyproject.toml"  # in a checkout, the installed metadata may be outdated
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomllib = None
    if pyproject.is_file() and tomllib:
        provided = set(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"])
    else:
        try:
            provided = set(metadata("seflag").get_all("Provides-Extra") or [])
        except PackageNotFoundError:
            pytest.skip("seflag is not installed")
    text = README.read_text(encoding="utf-8")
    mentioned = set(re.findall(r'"seflag\[(\w+)\]"', text))
    mentioned |= set(re.findall(r"`(\w+)`", text.split("\nExtras: ", 1)[1].split("\n\n", 1)[0]))
    assert mentioned and mentioned <= provided


def test_cli_examples_parse():
    parser = build_parser()
    commands = [words[1:] for words in shell_commands() if words[0] == "seflag"]
    assert commands
    for args in commands:
        parser.parse_args(args)  # exits on unknown commands or options


def subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return next(a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction))


def test_command_table_matches_the_cli():
    section = README.read_text(encoding="utf-8").split("\n## Commands\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `seflag")]
    assert rows
    for row in rows:
        command, *alternatives = re.findall(r"`([^`]+)`", row.split("|")[1])
        words = [w for w in command.split()[1:] if re.fullmatch(r"[a-z][a-z-]*", w)]
        parser = build_parser()
        for word in words:
            parser = subcommands(parser)[word]
        for alternative in alternatives:  # `seflag cache info` / `purge`
            assert alternative in subcommands(subcommands(build_parser())[words[0]]), row
        options = {o for a in parser._actions for o in a.option_strings}
        assert set(re.findall(r"--[a-z-]+", row)) <= options, row


def test_mentioned_example_files_exist():
    text = README.read_text(encoding="utf-8")
    written_in_readme = set(re.findall(r"^`([\w.-]+\.yaml)`:$", text, re.M))
    mentioned = set(re.findall(r"([\w-]+\.yaml)\b", text)) - written_in_readme
    assert mentioned
    for name in mentioned:
        assert (README.parent / "examples" / name).is_file(), name


@pytest.mark.parametrize("block", blocks("yaml"))
def test_configurations_are_valid(block, conllu_dir):
    data = yaml.safe_load(block)
    if is_matrix(data):
        load_matrix(data)
    elif "task" in data:
        load_config(data)
    else:  # a section of a configuration
        load_config({"language": "la", "task": "ner", "dataset": {"source": "conllu", "path": str(conllu_dir)},
                     **data})


def test_entry_point_example():
    tomllib = pytest.importorskip("tomllib")
    (block,) = blocks("toml")
    for group, entries in tomllib.loads(block)["project"]["entry-points"].items():
        assert group.removeprefix("seflag.") in registry.BUILTINS
        assert all(re.fullmatch(r"[\w.]+:[\w.]+", value) for value in entries.values())
