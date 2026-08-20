from __future__ import annotations

import argparse
import sys

from artifact_utils import add_src_to_path


def main() -> None:
    add_src_to_path()
    parser = argparse.ArgumentParser(description="Prepare LLM4PPM datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("canonicalize", help="Map raw activity names to canonical semantic names.")
    subparsers.add_parser("label-outcomes", help="Add final-outcome labels.")
    subparsers.add_parser("remaining-time", help="Add remaining-time labels.")
    subparsers.add_parser("pretrain-data", help="Build semantic-story corpus for continuous pre-training.")
    subparsers.add_parser("sft-data", help="Build multi-task instruction-tuning data.")
    args, rest = parser.parse_known_args()

    sys.argv = [sys.argv[0], *rest]
    if args.command == "canonicalize":
        from tools.canonicalize_activities import main as run
    elif args.command == "label-outcomes":
        from tools.label_outcomes import main as run
    elif args.command == "remaining-time":
        from tools.add_remaining_time import main as run
    elif args.command == "pretrain-data":
        from llm4ppm.story import main as run
    else:
        from llm4ppm.instruction_data import main as run
    run()


if __name__ == "__main__":
    main()
