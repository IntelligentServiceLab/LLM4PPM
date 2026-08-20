from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM4PPM command collection")
    parser.add_argument("command", choices=["pretrain-data", "sft-data", "pretrain", "sft", "evaluate"])
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    if args.command == "pretrain-data":
        from .story import main as run
    elif args.command == "sft-data":
        from .instruction_data import main as run
    elif args.command == "pretrain":
        from .pretrain import main as run
    elif args.command == "sft":
        from .sft import main as run
    else:
        from .evaluate import main as run
    run()


if __name__ == "__main__":
    main()
