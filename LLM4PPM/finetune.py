from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from artifact_utils import add_src_to_path, config_to_argv, load_yaml_args


DEFAULTS: Dict[str, Any] = {
    "model_path": "outputs/pretrain_adapter/merged",
    "dataset_path": "data/finetune_story.csv",
    "output_path": "outputs/sft_adapter",
    "log_path": "logs/sft",
    "text_column": "text",
    "response_template": "### Response\n",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "trust_remote_code": False,
    "allow_remote_model": False,
    "train_on_responses_only": True,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "random_seed": 3407,
    "num_train_epochs": 1.0,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "save_steps": 500,
    "logging_steps": 20,
    "eval_steps": 500,
    "save_total_limit": 2,
    "validation_ratio": 0.0,
    "dataset_num_proc": 64,
    "length_stats_sample_size": 2000,
    "max_train_samples": -1,
    "max_eval_samples": -1,
    "drop_overlong": False,
    "save_merged": True,
}

NEGATABLE_KEYS = {"load_in_4bit", "train_on_responses_only", "save_merged"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-task instruction fine-tuning for LLM4PPM.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config overriding script defaults.")
    for key, value in DEFAULTS.items():
        flag = "--" + key
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, int):
            parser.add_argument(flag, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(flag, type=float, default=value)
        else:
            parser.add_argument(flag, type=str, default=value)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser


def without_config_arg(argv: List[str]) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            i += 2
            continue
        cleaned.append(argv[i])
        i += 1
    return cleaned


def main() -> None:
    add_src_to_path()
    parser = build_arg_parser()
    config_args: List[str] = []
    config_probe, _ = parser.parse_known_args()
    if config_probe.config is not None:
        config_args = load_yaml_args(["--config", str(config_probe.config)], negatable_keys=NEGATABLE_KEYS)

    merged_args = config_args + without_config_arg(sys.argv[1:])
    args = parser.parse_args(merged_args)
    config = {key: getattr(args, key) for key in DEFAULTS}
    if args.resume_from_checkpoint:
        config["resume_from_checkpoint"] = args.resume_from_checkpoint
    argv = config_to_argv(config, negatable_keys=NEGATABLE_KEYS)

    if "outputs/pretrain_adapter/merged" == config["model_path"]:
        print("Using the default fine-tuning model path: outputs/pretrain_adapter/merged")

    sys.argv = [sys.argv[0], *argv]
    from llm4ppm.sft import main as run

    run()


if __name__ == "__main__":
    main()
