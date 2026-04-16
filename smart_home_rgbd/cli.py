from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_INDEX_CSV,
    DEFAULT_INSTANCE_LABELS_JSON,
    DEFAULT_INSTANCE_MANIFEST_JSON,
    DEFAULT_LABEL_UI_PORT,
    DEFAULT_MANUAL_LABELS_CSV,
    DEFAULT_STATS_JSON,
    DEFAULT_SUNRGBD_ROOT,
)
from .label_tool import run_label_tool
from .labeling_data import build_instance_manifest, write_instance_manifest
from .manual_labels import seed_manual_label_subset
from .sunrgbd_index import build_index, read_index_csv, summarize_records, write_index_csv


def cmd_index(args: argparse.Namespace) -> int:
    dataset_root = Path(args.root)
    output_csv = Path(args.output)
    stats_json = Path(args.stats_output)

    records = build_index(dataset_root)
    write_index_csv(records, output_csv)

    summary = summarize_records(records)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Indexed {summary['total_scenes']} scenes into {output_csv}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    rows = read_index_csv(Path(args.index))
    print(json.dumps(summarize_records(rows), indent=2))
    return 0


def cmd_seed_labels(args: argparse.Namespace) -> int:
    rows = read_index_csv(Path(args.index))
    selected = seed_manual_label_subset(
        rows,
        Path(args.output),
        max_per_combination=args.max_per_combination,
    )
    print(f"Wrote {len(selected)} label rows to {args.output}")
    return 0


def cmd_prepare_instances(args: argparse.Namespace) -> int:
    manifest = build_instance_manifest(Path(args.root), Path(args.seed_csv))
    write_instance_manifest(manifest, Path(args.output))
    print(f"Wrote instance manifest for {len(manifest)} scenes to {args.output}")
    return 0


def cmd_label_ui(args: argparse.Namespace) -> int:
    run_label_tool(
        dataset_root=Path(args.root),
        seed_csv=Path(args.seed_csv),
        manifest_path=Path(args.manifest),
        label_store_path=Path(args.labels),
        split_filter=args.split,
        port=args.port,
        open_browser=args.open_browser,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SUNRGBD smart-home pipeline utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index the local SUNRGBD dataset")
    index_parser.add_argument("--root", default=str(DEFAULT_SUNRGBD_ROOT))
    index_parser.add_argument("--output", default=str(DEFAULT_INDEX_CSV))
    index_parser.add_argument("--stats-output", default=str(DEFAULT_STATS_JSON))
    index_parser.set_defaults(func=cmd_index)

    stats_parser = subparsers.add_parser("stats", help="Summarize an existing index CSV")
    stats_parser.add_argument("--index", default=str(DEFAULT_INDEX_CSV))
    stats_parser.set_defaults(func=cmd_stats)

    seed_parser = subparsers.add_parser("seed-labels", help="Create a manual labeling seed CSV")
    seed_parser.add_argument("--index", default=str(DEFAULT_INDEX_CSV))
    seed_parser.add_argument("--output", default=str(DEFAULT_MANUAL_LABELS_CSV))
    seed_parser.add_argument("--max-per-combination", type=int, default=40)
    seed_parser.set_defaults(func=cmd_seed_labels)

    prep_parser = subparsers.add_parser("prepare-instances", help="Extract per-instance targets for labeling")
    prep_parser.add_argument("--root", default=str(DEFAULT_SUNRGBD_ROOT))
    prep_parser.add_argument("--seed-csv", default=str(DEFAULT_MANUAL_LABELS_CSV))
    prep_parser.add_argument("--output", default=str(DEFAULT_INSTANCE_MANIFEST_JSON))
    prep_parser.set_defaults(func=cmd_prepare_instances)

    ui_parser = subparsers.add_parser("label-ui", help="Launch the local Tkinter labeling tool")
    ui_parser.add_argument("--root", default=str(DEFAULT_SUNRGBD_ROOT))
    ui_parser.add_argument("--seed-csv", default=str(DEFAULT_MANUAL_LABELS_CSV))
    ui_parser.add_argument("--manifest", default=str(DEFAULT_INSTANCE_MANIFEST_JSON))
    ui_parser.add_argument("--labels", default=str(DEFAULT_INSTANCE_LABELS_JSON))
    ui_parser.add_argument("--split", default=None)
    ui_parser.add_argument("--port", type=int, default=DEFAULT_LABEL_UI_PORT)
    ui_parser.add_argument("--open-browser", action="store_true")
    ui_parser.set_defaults(func=cmd_label_ui)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
