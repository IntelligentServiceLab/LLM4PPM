from unsloth import FastLanguageModel, is_bfloat16_supported
import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction fine-tuning on semantic process stories.")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Local file (.json/.jsonl/.csv/.parquet), a directory containing train/validation files, or a HF dataset name.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the LoRA adapter and tokenizer.")
    parser.add_argument("--log_path", type=str, required=True, help="Directory for checkpoints and logs.")

    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--response_template", type=str, default="### Response\n")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--allow_remote_model", action="store_true", default=False)
    parser.add_argument("--train_on_responses_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.0,
        help="Create a validation split from train when the dataset has only one split.",
    )

    parser.add_argument("--dataset_num_proc", type=int, default=64)
    parser.add_argument("--length_stats_sample_size", type=int, default=2000)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument(
        "--drop_overlong",
        action="store_true",
        default=False,
        help="Filter out samples whose raw tokenized length exceeds max_seq_length.",
    )
    parser.add_argument(
        "--save_merged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Additionally save a merged model under <output_path>_merged.",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()



def set_all_seeds(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _load_single_file(path: Path) -> DatasetDict:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return load_dataset("json", data_files=str(path))
    if suffix == ".csv":
        return load_dataset("csv", data_files=str(path))
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path))
    raise ValueError(f"Unsupported file type: {path}")



def load_text_dataset(dataset_path: str) -> DatasetDict:
    path = Path(dataset_path)

    if path.is_file():
        ds = _load_single_file(path)
        if "train" not in ds:
            only_split = list(ds.keys())[0]
            ds = DatasetDict({"train": ds[only_split]})
        return ds

    if path.is_dir():
        train_file = None
        valid_file = None
        for name in ["train.csv", "train.jsonl", "train.json", "train.parquet"]:
            candidate = path / name
            if candidate.exists():
                train_file = candidate
                break
        for name in [
            "validation.csv",
            "validation.jsonl",
            "validation.json",
            "validation.parquet",
            "valid.csv",
            "valid.jsonl",
            "valid.json",
            "valid.parquet",
            "dev.csv",
            "dev.jsonl",
            "dev.json",
            "dev.parquet",
        ]:
            candidate = path / name
            if candidate.exists():
                valid_file = candidate
                break

        if train_file is None:
            raise ValueError(f"No train file found under directory: {path}")

        train_ds = _load_single_file(train_file)["train"]
        if valid_file is not None:
            valid_ds = _load_single_file(valid_file)["train"]
            return DatasetDict({"train": train_ds, "validation": valid_ds})
        return DatasetDict({"train": train_ds})

    ds = load_dataset(dataset_path)
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({"train": ds})



def ensure_text_column(ds: Dataset, text_column: str) -> Dataset:
    if text_column not in ds.column_names:
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {ds.column_names}")

    def _clean(batch):
        texts = []
        for x in batch[text_column]:
            if x is None:
                texts.append("")
            elif isinstance(x, str):
                texts.append(x.strip())
            else:
                texts.append(str(x).strip())
        return {text_column: texts}

    ds = ds.map(_clean, batched=True, num_proc=1)
    ds = ds.filter(lambda x: isinstance(x[text_column], str) and len(x[text_column]) > 0)
    return ds



def maybe_make_validation_split(ds_dict: DatasetDict, validation_ratio: float, seed: int) -> DatasetDict:
    if "validation" in ds_dict or validation_ratio <= 0:
        return ds_dict
    split = ds_dict["train"].train_test_split(test_size=validation_ratio, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})



def maybe_limit_samples(ds: Dataset, max_samples: int) -> Dataset:
    if max_samples is None or max_samples < 0:
        return ds
    return ds.select(range(min(max_samples, len(ds))))



def maybe_filter_overlong(ds: Dataset, tokenizer, text_column: str, max_seq_length: int, num_proc: int) -> Dataset:
    def _token_len(batch):
        enc = tokenizer(batch[text_column], add_special_tokens=False, truncation=False)
        return {"_n_tokens": [len(x) for x in enc["input_ids"]]}

    ds = ds.map(_token_len, batched=True, num_proc=max(1, num_proc), desc="Measuring token lengths")
    before = len(ds)
    ds = ds.filter(lambda x: x["_n_tokens"] < max_seq_length, num_proc=max(1, num_proc), desc="Filtering overlong samples")
    after = len(ds)
    dropped = before - after
    print(f"Filtered {dropped} overlong samples ({dropped / max(before, 1):.2%}) with max_seq_length={max_seq_length}.")
    keep_cols = [c for c in ds.column_names if c != "_n_tokens"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
    return ds



def print_length_stats(ds: Dataset, tokenizer, text_column: str, sample_size: int, seed: int) -> None:
    if len(ds) == 0:
        print("Dataset is empty after preprocessing.")
        return
    n = min(sample_size, len(ds))
    sample = ds.shuffle(seed=seed).select(range(n))
    enc = tokenizer(sample[text_column], add_special_tokens=False, truncation=False)
    lengths = np.array([len(x) for x in enc["input_ids"]], dtype=np.int32)
    stats = {
        "count": int(lengths.shape[0]),
        "min": int(lengths.min()),
        "p50": int(np.percentile(lengths, 50)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
    }
    print("Token length statistics on a random sample:")
    for k, v in stats.items():
        print(f"  {k}: {v}")



def build_tokenize_fn(tokenizer, text_column: str, response_template: str, max_seq_length: int, eos_token: Optional[str], train_on_responses_only: bool):
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        input_ids_batch: List[List[int]] = []
        attention_masks_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        loss_token_counts: List[int] = []

        for raw_text in batch[text_column]:
            text = raw_text.strip()
            if eos_token and not text.endswith(eos_token):
                text = text + eos_token

            marker_pos = text.rfind(response_template)
            if marker_pos < 0:
                raise ValueError(
                    f"Response template not found in sample. Expected marker: {response_template!r}\nSample head: {text[:300]!r}"
                )
            response_start_char = marker_pos + len(response_template)
            prompt_text = text[:response_start_char]

            full_enc = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )
            prompt_enc = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )

            input_ids = list(full_enc["input_ids"])
            attention_mask = list(full_enc["attention_mask"])
            labels = list(input_ids)

            if train_on_responses_only:
                prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100

            loss_token_count = sum(1 for x in labels if x != -100)
            input_ids_batch.append(input_ids)
            attention_masks_batch.append(attention_mask)
            labels_batch.append(labels)
            loss_token_counts.append(loss_token_count)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks_batch,
            "labels": labels_batch,
            "_loss_token_count": loss_token_counts,
        }

    return _tokenize


@dataclass
class SupervisedDataCollator:
    tokenizer: any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]

        labels = []
        for feature in features:
            label = list(feature["labels"])
            pad_length = max_length - len(label)
            if self.tokenizer.padding_side == "right":
                label = label + [self.label_pad_token_id] * pad_length
            else:
                label = [self.label_pad_token_id] * pad_length + label
            labels.append(label)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch



def main() -> None:
    args = parse_args()
    set_all_seeds(args.random_seed)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    eos_token = tokenizer.eos_token

    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        local_files_only=not args.allow_remote_model,
    )
    model.config.use_cache = False

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
    )

    print(f"\nLoading dataset from {args.dataset_path}")
    ds_dict = load_text_dataset(args.dataset_path)
    ds_dict = DatasetDict({k: ensure_text_column(v, args.text_column) for k, v in ds_dict.items()})
    ds_dict = maybe_make_validation_split(ds_dict, args.validation_ratio, args.random_seed)

    ds_dict["train"] = maybe_limit_samples(ds_dict["train"], args.max_train_samples)
    if "validation" in ds_dict:
        ds_dict["validation"] = maybe_limit_samples(ds_dict["validation"], args.max_eval_samples)

    print(f"Train size before length filtering: {len(ds_dict['train'])}")
    if "validation" in ds_dict:
        print(f"Validation size before length filtering: {len(ds_dict['validation'])}")

    print_length_stats(ds_dict["train"], tokenizer, args.text_column, args.length_stats_sample_size, args.random_seed)

    if args.drop_overlong:
        ds_dict["train"] = maybe_filter_overlong(
            ds_dict["train"], tokenizer, args.text_column, args.max_seq_length, args.dataset_num_proc
        )
        if "validation" in ds_dict:
            ds_dict["validation"] = maybe_filter_overlong(
                ds_dict["validation"], tokenizer, args.text_column, args.max_seq_length, args.dataset_num_proc
            )

    print(f"Train size after optional filtering: {len(ds_dict['train'])}")
    if len(ds_dict["train"]) == 0:
        raise ValueError("Training dataset is empty.")
    print("\nTraining example:\n")
    print(ds_dict["train"][0][args.text_column][:2000])
    print()

    tokenize_fn = build_tokenize_fn(
        tokenizer=tokenizer,
        text_column=args.text_column,
        response_template=args.response_template,
        max_seq_length=args.max_seq_length,
        eos_token=eos_token,
        train_on_responses_only=args.train_on_responses_only,
    )

    tokenized = DatasetDict()
    for split_name, split_ds in ds_dict.items():
        tok_ds = split_ds.map(
            tokenize_fn,
            batched=True,
            num_proc=max(1, args.dataset_num_proc),
            remove_columns=split_ds.column_names,
            desc=f"Tokenizing {split_name}",
        )
        tok_ds = tok_ds.filter(lambda x: x["_loss_token_count"] > 0, num_proc=1)
        tok_ds = tok_ds.remove_columns(["_loss_token_count"])
        tokenized[split_name] = tok_ds

    print(f"Tokenized train size: {len(tokenized['train'])}")
    if "validation" in tokenized:
        print(f"Tokenized validation size: {len(tokenized['validation'])}")

    has_eval = "validation" in tokenized and len(tokenized["validation"]) > 0
    eval_strategy = "steps" if has_eval else "no"
    save_strategy = "steps"

    data_collator = SupervisedDataCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if has_eval else None,
        max_steps=args.max_steps,
        save_total_limit=args.save_total_limit,
        logging_first_step=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.random_seed,
        output_dir=args.log_path,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        group_by_length=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if has_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.\n")
    else:
        start_gpu_memory = 0.0
        max_memory = 0.0

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max(max_memory, 1e-8) * 100, 3)
        train_percentage = round(used_memory_for_training / max(max_memory, 1e-8) * 100, 3)
        print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {train_percentage} %.\n")

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    if args.save_merged:
        merged_output = args.output_path.rstrip("/\\") + "_merged"
        model.save_pretrained_merged(merged_output, tokenizer)
        print(f"Saved merged model to {merged_output}")

    print(f"Saved LoRA adapter and tokenizer to {args.output_path}")


if __name__ == "__main__":
    main()
