"""Command line interface: ``seflag run``, ``seflag list``, ``seflag convert-conllu``, ``seflag export``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from seflag import __version__, registry
from seflag.errors import SeflagError


def _cmd_run(args: argparse.Namespace) -> int:
    from seflag.config import load_config
    from seflag.runner import run

    overrides: dict = {}
    if args.max_sentences:
        overrides["options"] = {"max_sentences": args.max_sentences}
    if args.output:
        overrides["output"] = {"dir": str(Path(args.output).resolve())}
    if args.no_save:
        overrides["output"] = {"dir": None}
    for path in args.configs:
        result = run(load_config(path, **overrides))
        print(result.summary())
        if result.path:
            print(f"  -> {result.path}")
    return 0


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seflag", description=__doc__)
    parser.add_argument("--version", action="version", version=f"seflag {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print results.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one or more evaluations from configuration files.")
    p_run.add_argument("configs", nargs="+", help="YAML configuration file(s).")
    p_run.add_argument("--max-sentences", type=int, help="Override options.max_sentences (quick test runs).")
    p_run.add_argument("--output", help="Override output.dir.")
    p_run.add_argument("--no-save", action="store_true", help="Do not write result files.")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except SeflagError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
