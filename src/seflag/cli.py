"""Command line interface: ``seflag run`` (single evaluations and matrices), ``list``, ``convert-conllu``,
``export``, ``inspect-encoding`` and ``cache``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from seflag import __version__, registry
from seflag.errors import SeflagError


def _cmd_run(args: argparse.Namespace) -> int:
    from seflag.config import load_config
    from seflag.matrix import is_matrix, load_matrix, run_matrix
    from seflag.runner import run

    overrides: dict = {}
    if args.max_sentences:
        overrides["options"] = {"max_sentences": args.max_sentences}
    if args.output:
        overrides["output"] = {"dir": str(Path(args.output).resolve())}
    if args.no_save:
        overrides["output"] = {"dir": None}
    status: int = 0
    for path in args.configs:
        if is_matrix(path):
            matrix = run_matrix(load_matrix(path, **overrides), fail_fast=args.fail_fast)
            print(f"{matrix.name}\n\n{matrix.table()}")
            if matrix.path:
                print(f"  -> {matrix.path}")
            status = status or (1 if matrix.failed else 0)
            continue
        result = run(load_config(path, **overrides))
        print(result.summary())
        if result.path:
            print(f"  -> {result.path}")
    return status


def _cmd_list(args: argparse.Namespace) -> int:
    from seflag.normalization import NORMALIZERS

    for kind, field in (("tasks", "task"), ("datasets", "dataset.source"), ("backends", "model.backend")):
        print(f"{kind} ({field}):")
        for name, target in sorted(registry.available(kind).items()):
            try:
                component = registry.get(kind, name)
                print(f"  {name:<16} {getattr(component, 'description', '')}")
                if args.verbose:
                    for fname, finfo in component.Config.model_fields.items():
                        if finfo.is_required():
                            default = "required"
                        else:
                            default = f"default: {finfo.get_default(call_default_factory=True)!r}"
                        print(f"      {fname} ({default}) {finfo.description or ''}")
            except ImportError as e:
                print(f"  {name:<16} (unavailable: {e})")
    print("normalizers (options.input_normalization / options.label_normalization):")
    for name, (_, description) in NORMALIZERS.items():
        print(f"  {name:<22} {description}")
    return 0


def _cmd_convert_conllu(args: argparse.Namespace) -> int:
    from seflag.data.convert import conllu_to_dataset_dict, verify_roundtrip, write_parquet

    paths: list[Path] = []
    for p in map(Path, args.inputs):
        paths += sorted(p.glob("*.conllu")) if p.is_dir() else [p]
    if not paths:
        print("error: no CoNLL-U files found", file=sys.stderr)
        return 1
    dataset_dict = conllu_to_dataset_dict(paths, default_split=args.default_split)
    print(dataset_dict)
    written: list[Path] = write_parquet(dataset_dict, Path(args.output))
    for path in written:
        print(f"wrote {path}")
    problems: list[str] = verify_roundtrip(paths, {p.stem: p for p in written}, default_split=args.default_split)
    if problems:
        print("ROUND-TRIP CHECK FAILED: the Parquet files do not reproduce the sources exactly:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"round-trip check passed: Parquet -> CoNLL-U reproduces all {len(paths)} source file(s) byte for byte")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from seflag.config import load_config, validate_component_params
    from seflag.data.convert import records_to_dataset, verify_records

    config = load_config(args.config)
    component = registry.get("datasets", config.dataset.source)
    dataset = component(validate_component_params(component, config.dataset.params, "dataset"), config)
    records = list(dataset.records())
    target: Path = Path(args.output) / "data" / f"{args.split}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    records_to_dataset(records).to_parquet(str(target))
    print(f"wrote {len(records)} sentences to {target}")
    problems: list[str] = verify_records(records, target)
    if problems:
        print("CHECK FAILED: the Parquet file does not reproduce the records:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("check passed: the Parquet file reproduces all records exactly")
    return 0


def _inspect_targets(path: str, with_model: bool) -> list[tuple[str, str, object]]:
    """(kind, label, component instance) for the dataset(s) and, if requested, model(s) of a configuration file."""
    from seflag.config import load_config, validate_component_params
    from seflag.matrix import is_matrix, load_matrix

    configs = [(f"{c.dataset} × {c.model}", c.config) for c in load_matrix(path).cells if c.config] \
        if is_matrix(path) else [("", load_config(path))]
    targets: list[tuple[str, str, object]] = []
    seen: set[str] = set()
    for _, config in configs:
        dataset_key: str = f"dataset {config.dataset.source} {config.dataset.params}"
        if dataset_key not in seen:
            seen.add(dataset_key)
            component = registry.get("datasets", config.dataset.source)
            label: str = str(config.dataset.params.get("repo") or config.dataset.params.get("path")
                             or config.dataset.params.get("url") or config.dataset.source)
            targets.append(("dataset", f"{label} ({config.language})",
                            component(validate_component_params(component, config.dataset.params, "dataset"), config)))
        model_key: str = f"model {config.model.backend} {config.model.params}"
        if with_model and model_key not in seen:
            seen.add(model_key)
            component = registry.get("backends", config.model.backend)
            label = str(config.model.params.get("name") or config.model.params.get("url") or config.model.backend)
            targets.append(("model", f"{label} ({config.model.backend})",
                            component(validate_component_params(component, config.model.params, "model"), config)))
    return targets


def _cmd_inspect_encoding(args: argparse.Namespace) -> int:
    import json

    from seflag.backends.embeddings import EmbeddingBackend
    from seflag.encoding import analyse, inspect_dataset

    report: list[dict] = []
    for path in args.configs:
        for kind, label, component in _inspect_targets(path, args.model):
            if kind == "dataset":
                stats = inspect_dataset(component, by_subset=args.by_subset, max_sentences=args.max_sentences)
            else:
                vocabulary = component.vocabulary() if isinstance(component, EmbeddingBackend) else None
                if vocabulary is None:
                    print(f"== {path}: model {label}: vocabulary inspection is only available for embedding "
                          "backends; see ENCODING.md for how other models were examined", file=sys.stderr)
                    continue
                stats = {"": {"vocabulary": analyse(vocabulary)}}
            report.append({"config": path, "kind": kind, "label": label,
                           "subsets": {s: {f: v.to_dict() for f, v in fields.items()} for s, fields in stats.items()}})
            if args.json:
                continue
            print(f"== {path}: {kind} {label}")
            for subset, fields in stats.items():
                for name, values in fields.items():
                    print(f"[{subset or 'all'}] {name}")
                    print("\n".join(values.format()))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []


def _cmd_cache(args: argparse.Namespace) -> int:
    import shutil

    from seflag.data.download import downloads_root, seflag_home

    root: Path = downloads_root()
    files: list[Path] = _files(root)
    tools_root: Path = seflag_home() / "tools"  # programs built by SEFLAG (Morpheus)
    tools: list[Path] = sorted(p for p in tools_root.iterdir() if p.is_dir()) if tools_root.is_dir() else []
    tool_files: list[Path] = _files(tools_root)
    size: int = sum(p.stat().st_size for p in files + tool_files)
    if args.cache_command == "info":
        print(f"downloads: {root}")
        print(f"  {len(files)} file(s), {_human_size(sum(p.stat().st_size for p in files))}")
        for path in sorted(files, key=lambda p: -p.stat().st_size)[:args.limit]:
            print(f"  {_human_size(path.stat().st_size):>9}  {path.name}")
        print(f"tools: {tools_root}")
        for tool in tools:
            print(f"  {_human_size(sum(p.stat().st_size for p in _files(tool))):>9}  {tool.name}")
        print("Models and datasets from the Hugging Face Hub and environments created by uv are cached by those "
              "tools (see `hf cache` and `uv cache`).")
        return 0
    # purge
    if not files and not tools:
        print(f"nothing to delete in {seflag_home()}")
        return 0
    what: str = f"{len(files)} downloaded file(s)" + "".join(
        f" and {t.name}" for t in tools if not t.name.startswith("."))  # hidden: unfinished builds
    if not args.yes:
        if not sys.stdin.isatty():
            print(f"error: refusing to delete {what} ({_human_size(size)}) without confirmation; pass --yes",
                  file=sys.stderr)
            return 1
        answer: str = input(f"Delete {what} ({_human_size(size)}) in {seflag_home()}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("nothing deleted")
            return 0
    for directory in (root, tools_root):
        shutil.rmtree(directory, ignore_errors=not directory.exists())
    print(f"deleted {what}, freed {_human_size(size)}; everything is downloaded or rebuilt again when needed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seflag", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seflag {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print results.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one or more evaluations from configuration files.")
    p_run.add_argument("configs", nargs="+", help="YAML configuration file(s), single evaluations or matrices.")
    p_run.add_argument("--max-sentences", type=int, help="Override options.max_sentences (quick test runs).")
    p_run.add_argument("--output", help="Override output.dir.")
    p_run.add_argument("--no-save", action="store_true", help="Do not write result files.")
    p_run.add_argument("--fail-fast", action="store_true",
                       help="Matrices: stop at the first failing cell instead of reporting it in the table.")
    p_run.set_defaults(func=_cmd_run)

    p_list = sub.add_parser("list", help="List available tasks, dataset sources, backends and normalizers.")
    p_list.add_argument("-v", "--verbose", action="store_true", help="Also show configuration fields.")
    p_list.set_defaults(func=_cmd_list)

    p_conv = sub.add_parser("convert-conllu", help="Convert CoNLL-U files into SEFLAG's Hugging Face layout.")
    p_conv.add_argument("inputs", nargs="+", help="CoNLL-U files or directories.")
    p_conv.add_argument("--output", required=True, help="Output directory (parquet files go to OUTPUT/data/).")
    p_conv.add_argument("--default-split", default="test", help="Split for files without -train/-dev/-test suffix.")
    p_conv.set_defaults(func=_cmd_convert_conllu)

    p_exp = sub.add_parser("export", help="Export the dataset of a configuration file to SEFLAG's Parquet layout.")
    p_exp.add_argument("config", help="YAML configuration file; only its 'dataset' section is used.")
    p_exp.add_argument("--output", required=True, help="Output directory (parquet file goes to OUTPUT/data/).")
    p_exp.add_argument("--split", default="test", help="Split name, i.e. the parquet file name (default: test).")
    p_exp.set_defaults(func=_cmd_export)

    p_enc = sub.add_parser("inspect-encoding",
                           help="Report how the strings of a configuration's dataset (and embedding vocabulary) are "
                                "encoded: normalization form, elision marks, sigma variants, u/v, case, ...")
    p_enc.add_argument("configs", nargs="+", help="YAML configuration file(s), single evaluations or matrices.")
    p_enc.add_argument("--by-subset", action="store_true", help="Also report each subset (e.g. treebank) separately.")
    p_enc.add_argument("--model", action="store_true", help="Also inspect the vocabulary of embedding models.")
    p_enc.add_argument("--max-sentences", type=int, help="Only inspect the first N sentences or items.")
    p_enc.add_argument("--json", action="store_true", help="Print the full statistics as JSON.")
    p_enc.set_defaults(func=_cmd_inspect_encoding)

    p_cache = sub.add_parser("cache", help="Show or delete files downloaded by URL (datasets, embedding models) "
                                           "and tools built by SEFLAG (Morpheus).")
    cache_sub = p_cache.add_subparsers(dest="cache_command", required=True)
    p_info = cache_sub.add_parser("info", help="Show location, size and the largest downloaded files.")
    p_info.add_argument("--limit", type=int, default=10, help="Number of files to list (default: 10).")
    p_purge = cache_sub.add_parser("purge", help="Delete all downloaded files and built tools to free disk space.")
    p_purge.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation.")
    p_cache.set_defaults(func=_cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # INFO messages only from SEFLAG itself; libraries such as gensim are very talkative at INFO level
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("seflag").setLevel(logging.WARNING if args.quiet else logging.INFO)
    try:
        return args.func(args)
    except SeflagError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
