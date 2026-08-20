from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


VALID_TASKS = [
    "next_activity_prediction",
    "remaining_time_prediction",
    "final_outcome_prediction",
]
OUTCOME_CANDIDATES = ["success", "failure"]
INVALID_LABEL = "__INVALID__"
STAGE_NAMES = ["early", "middle", "late"]
TRIE_END = "__END__"
DEFAULT_RESPONSE_TEMPLATE = "### Response\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a semantic-story PPM model with the same prompt style used in fine-tuning."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing per-dataset CSV logs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory used to save splits, predictions, and metrics.")
    parser.add_argument(
        "--story_module",
        type=str,
        default="llm4ppm.story",
        help="Python module that defines TEST_DATASETS and dataset-loading helpers.",
    )
    parser.add_argument(
        "--finetune_module",
        type=str,
        default="llm4ppm.instruction_data",
        help="Python module used to build fine-tuning prompts.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=None,
        help="Target datasets. If omitted, TEST_DATASETS from --story_module is used.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Merged model path, or base model path when --adapter_path is set.",
    )
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load the evaluation model in 4-bit.")
    parser.add_argument("--trust_remote_code", action="store_true", default=False)

    parser.add_argument("--tasks", nargs="*", default=VALID_TASKS, choices=VALID_TASKS)
    parser.add_argument(
        "--include_system_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the same system-prompt setting as fine-tuning.",
    )
    parser.add_argument(
        "--include_dataset_intro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the dataset/process intro sentence in each prefix story.",
    )
    parser.add_argument(
        "--quote_activities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap activity names in quotation marks inside the story.",
    )
    parser.add_argument(
        "--shuffle_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle next-activity and outcome candidate order per prefix to match fine-tuning more closely.",
    )
    parser.add_argument(
        "--max_prefix_story_events",
        type=int,
        default=40,
        help="Maximum number of events retained in the story shown to the model. Matches fine-tuning by default.",
    )
    parser.add_argument(
        "--min_prefix_len",
        type=int,
        default=2,
        help="Evaluate prefixes with length >= this value. Default matches your current protocol.",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Case-level test ratio inside each target dataset. The remaining cases are saved for baseline training.",
    )
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens_next", type=int, default=32)
    parser.add_argument("--max_new_tokens_time", type=int, default=12)
    parser.add_argument("--max_new_tokens_outcome", type=int, default=12)

    parser.add_argument(
        "--response_template",
        type=str,
        default=DEFAULT_RESPONSE_TEMPLATE,
        help="Response marker used during fine-tuning. Needed for constrained decoding tokenization.",
    )
    parser.add_argument(
        "--use_constrained_decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constrain categorical generation to the candidate set by token-prefix filtering.",
    )
    parser.add_argument(
        "--force_candidate_block_in_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If compose_text does not support candidate_block, inject a candidate block into the prompt string.",
    )

    parser.add_argument("--save_case_splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prediction_file_name", type=str, default="predictions.csv")
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": "auto",
    }
    compute_dtype = get_compute_dtype()
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    if args.adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required when --adapter_path is provided.")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_pipeline_modules(story_module_name: str, finetune_module_name: str):
    story_mod = importlib.import_module(story_module_name)
    ft_mod = importlib.import_module(finetune_module_name)
    return story_mod, ft_mod


def canonicalize_case_label(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Encountered NaN in label column.")
    text = str(value).strip().lower()
    if text in {"succeed", "success", "successful", "1", "true", "yes"}:
        return "success"
    if text in {"failure", "fail", "failed", "0", "false", "no"}:
        return "failure"
    raise ValueError(f"Unsupported label value: {value!r}")


def stratified_case_split(
    case_ids: Sequence[Any],
    case_label_map: Dict[Any, str],
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[Any], List[Any]]:
    if test_ratio <= 0:
        return list(case_ids), []

    buckets: Dict[str, List[Any]] = defaultdict(list)
    for case_id in case_ids:
        buckets[case_label_map[case_id]].append(case_id)

    train_ids: List[Any] = []
    test_ids: List[Any] = []
    for _, bucket in buckets.items():
        bucket = list(bucket)
        rng.shuffle(bucket)
        if len(bucket) == 1:
            train_ids.extend(bucket)
            continue
        n_test = int(round(len(bucket) * test_ratio))
        n_test = max(1, n_test)
        n_test = min(n_test, len(bucket) - 1)
        test_ids.extend(bucket[:n_test])
        train_ids.extend(bucket[n_test:])

    if not test_ids and len(train_ids) > 1:
        rng.shuffle(train_ids)
        test_ids.append(train_ids.pop())

    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    return train_ids, test_ids


def save_case_split_files(
    output_dir: Path,
    dataset_name: str,
    full_df: pd.DataFrame,
    case_id_col: str,
    baseline_train_case_ids: Sequence[Any],
    zero_shot_test_case_ids: Sequence[Any],
) -> None:
    split_dir = output_dir / "case_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    baseline_train_rows = full_df[full_df[case_id_col].isin(set(baseline_train_case_ids))].copy()
    zero_shot_test_rows = full_df[full_df[case_id_col].isin(set(zero_shot_test_case_ids))].copy()
    baseline_train_rows.to_csv(split_dir / f"{dataset_name}_baseline_train.csv", index=False, encoding="utf-8")
    zero_shot_test_rows.to_csv(split_dir / f"{dataset_name}_zero_shot_test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"case_id": list(baseline_train_case_ids)}).to_csv(
        split_dir / f"{dataset_name}_baseline_train_case_ids.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame({"case_id": list(zero_shot_test_case_ids)}).to_csv(
        split_dir / f"{dataset_name}_zero_shot_test_case_ids.csv", index=False, encoding="utf-8"
    )


def deterministic_shuffle(items: Sequence[str], seed_text: str, base_seed: int) -> List[str]:
    copied = list(items)
    stable_offset = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(base_seed + stable_offset)
    rng.shuffle(copied)
    return copied


def fallback_slice_prefix_for_story(case_df: pd.DataFrame, prefix_len: int, max_prefix_story_events: int) -> Tuple[pd.DataFrame, int]:
    prefix_df = case_df.iloc[:prefix_len].copy()
    segment_start = 0
    if max_prefix_story_events > 0 and len(prefix_df) > max_prefix_story_events:
        segment_start = len(prefix_df) - max_prefix_story_events
        prefix_df = prefix_df.iloc[segment_start:].copy()
    return prefix_df.reset_index(drop=True), int(segment_start)


def call_build_case_story(
    build_case_story_fn,
    prefix_story_df: pd.DataFrame,
    observed_prefix_df: pd.DataFrame,
    dataset_name: str,
    activity_col: str,
    timestamp_col: Optional[str],
    include_dataset_intro: bool,
    quote_activities: bool,
    segment_start: int,
) -> str:
    sig = inspect.signature(build_case_story_fn)
    kwargs: Dict[str, Any] = {
        "case_df": prefix_story_df,
        "dataset_name": dataset_name,
        "activity_col": activity_col,
        "timestamp_col": timestamp_col,
        "include_dataset_intro": include_dataset_intro,
        "quote_activities": quote_activities,
    }
    if "segment_start" in sig.parameters:
        kwargs["segment_start"] = segment_start
    if "trace_context_df" in sig.parameters:
        kwargs["trace_context_df"] = observed_prefix_df
    return build_case_story_fn(**kwargs)


def count_test_prefixes(cases: Dict[Any, Dict[str, Any]], case_ids: Sequence[Any], min_prefix_len: int) -> int:
    total = 0
    for case_id in case_ids:
        trace_len = int(cases[case_id]["trace_len"])
        total += max(0, trace_len - min_prefix_len)
    return total


def iter_prefix_examples(
    *,
    dataset_name: str,
    cases: Dict[Any, Dict[str, Any]],
    case_ids: Sequence[Any],
    activity_col: str,
    timestamp_col: Optional[str],
    remaining_time_col: Optional[str],
    include_dataset_intro: bool,
    quote_activities: bool,
    max_prefix_story_events: int,
    min_prefix_len: int,
    shuffle_candidates: bool,
    random_seed: int,
    dataset_activity_candidates: Sequence[str],
    slice_prefix_fn,
    build_case_story_fn,
    get_remaining_time_label_fn,
) -> Iterator[Dict[str, Any]]:
    for case_id in case_ids:
        record = cases[case_id]
        case_df = record["case_df"]
        trace_len = int(record["trace_len"])
        case_label = str(record["label"])

        for prefix_len in range(min_prefix_len, trace_len):
            observed_prefix_df = case_df.iloc[:prefix_len].copy().reset_index(drop=True)
            prefix_story_df, segment_start = slice_prefix_fn(
                case_df=case_df,
                prefix_len=prefix_len,
                max_prefix_story_events=max_prefix_story_events,
            )
            prefix_story = call_build_case_story(
                build_case_story_fn=build_case_story_fn,
                prefix_story_df=prefix_story_df,
                observed_prefix_df=observed_prefix_df,
                dataset_name=dataset_name,
                activity_col=activity_col,
                timestamp_col=timestamp_col,
                include_dataset_intro=include_dataset_intro,
                quote_activities=quote_activities,
                segment_start=segment_start,
            )
            next_activity = str(case_df.iloc[prefix_len][activity_col]).strip()
            remaining_time_days = get_remaining_time_label_fn(
                case_df=case_df,
                prefix_len=prefix_len,
                timestamp_col=timestamp_col,
                remaining_time_col=remaining_time_col,
            )
            if shuffle_candidates:
                next_candidates = deterministic_shuffle(
                    dataset_activity_candidates,
                    f"{dataset_name}|next|{case_id}|{prefix_len}",
                    random_seed,
                )
                outcome_candidates = deterministic_shuffle(
                    OUTCOME_CANDIDATES,
                    f"{dataset_name}|outcome|{case_id}|{prefix_len}",
                    random_seed,
                )
            else:
                next_candidates = list(dataset_activity_candidates)
                outcome_candidates = list(OUTCOME_CANDIDATES)
            yield {
                "dataset": dataset_name,
                "case_id": str(case_id),
                "prefix_len": int(prefix_len),
                "trace_len": int(trace_len),
                "prefix_ratio": round(float(prefix_len) / max(float(trace_len), 1.0), 4),
                "story_segment_start": int(segment_start),
                "story_n_events": int(len(prefix_story_df)),
                "input_story": prefix_story,
                "next_activity": next_activity,
                "remaining_time_days": int(remaining_time_days),
                "final_outcome": case_label,
                "next_candidates": next_candidates,
                "outcome_candidates": outcome_candidates,
            }


def clean_generated_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("<s>", " ").replace("</s>", " ").replace("<unk>", " ")
    text = text.replace("```", " ").replace("**", " ")
    text = text.replace("\u0000", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def normalize_text_for_match(text: str) -> str:
    text = str(text).strip().strip("\"'`")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\-:;,.]+|[\s\-:;,.]+$", "", text)
    return text.lower()


_PREFIX_RE = re.compile(
    r"^(?:the\s+next\s+activity\s+is|next\s+activity\s+is|the\s+final\s+outcome\s+is|final\s+outcome\s+is|"
    r"the\s+prediction\s+is|prediction\s+is|prediction|answer|response|label|output)\s*[:：-]*\s*",
    flags=re.IGNORECASE,
)


def strip_answer_prefixes(text: str) -> str:
    current = text.strip()
    previous = None
    while previous != current and current:
        previous = current
        current = _PREFIX_RE.sub("", current).strip()
        current = re.sub(r"^[>•*\-\d\.)\s]+", "", current).strip()
        current = current.strip("\"'`").strip()
    return current


def tokenize_match_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text_for_match(text))


def ordered_token_prefix(short_text: str, long_text: str) -> bool:
    short_tokens = tokenize_match_text(short_text)
    long_tokens = tokenize_match_text(long_text)
    if len(short_tokens) < 2 or len(short_tokens) > len(long_tokens):
        return False
    return long_tokens[: len(short_tokens)] == short_tokens


def extract_candidate_like_spans(cleaned: str) -> List[str]:
    spans: List[str] = []
    seen = set()

    def add_span(value: str) -> None:
        value = strip_answer_prefixes(value)
        value = value.strip()
        if not value:
            return
        if value not in seen:
            seen.add(value)
            spans.append(value)

    add_span(cleaned)
    for line in cleaned.splitlines():
        add_span(line)
        if ":" in line:
            add_span(line.split(":", 1)[1])
    for quoted in re.findall(r'"([^"\n]+)"', cleaned):
        add_span(quoted)
    for quoted in re.findall(r"'([^'\n]+)'", cleaned):
        add_span(quoted)
    for part in re.split(r"[\n\r;|]+", cleaned):
        add_span(part)
        for sub_part in re.split(r"[.]\s+|,\s+(?=[A-Za-z])", part):
            add_span(sub_part)
    return spans


def parse_categorical_prediction(raw_text: str, candidates: Sequence[str]) -> Tuple[str, bool]:
    cleaned = clean_generated_text(raw_text)
    if not cleaned:
        return INVALID_LABEL, False

    normalized_to_original = {normalize_text_for_match(c): c for c in candidates}
    spans = extract_candidate_like_spans(cleaned)

    for span in spans:
        span_norm = normalize_text_for_match(span)
        if span_norm in normalized_to_original:
            return normalized_to_original[span_norm], True

    for span in spans:
        span_norm = normalize_text_for_match(span)
        containing = [orig for norm, orig in normalized_to_original.items() if norm and norm in span_norm]
        if len(containing) == 1:
            return containing[0], True

    for span in spans:
        span_norm = normalize_text_for_match(span)
        prefix_matches = [orig for norm, orig in normalized_to_original.items() if ordered_token_prefix(span_norm, norm)]
        if len(prefix_matches) == 1:
            return prefix_matches[0], True
        inverse_prefix_matches = [orig for norm, orig in normalized_to_original.items() if ordered_token_prefix(norm, span_norm)]
        if len(inverse_prefix_matches) == 1:
            return inverse_prefix_matches[0], True

    best_ratio = -1.0
    second_ratio = -1.0
    best_candidate = None
    for span in spans:
        span_norm = normalize_text_for_match(span)
        if not span_norm:
            continue
        for cand_norm, cand_original in normalized_to_original.items():
            ratio = SequenceMatcher(None, span_norm, cand_norm).ratio()
            if ratio > best_ratio:
                second_ratio = best_ratio
                best_ratio = ratio
                best_candidate = cand_original
            elif ratio > second_ratio:
                second_ratio = ratio
    if best_candidate is not None and best_ratio >= 0.92 and (best_ratio - max(second_ratio, 0.0)) >= 0.05:
        return best_candidate, True

    return INVALID_LABEL, False


def parse_remaining_time_prediction(raw_text: str) -> Tuple[int, bool]:
    cleaned = clean_generated_text(raw_text)
    if not cleaned:
        return 0, False
    match_full_int = re.fullmatch(r"\d+", cleaned)
    if match_full_int:
        return int(match_full_int.group(0)), True
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not numbers:
        return 0, False
    value = int(math.floor(float(numbers[-1])))
    return max(0, value), True


def default_build_candidate_block(candidates: Optional[Sequence[str]]) -> Optional[str]:
    if not candidates:
        return None
    lines = ["Copy exactly one candidate string from the list below:"]
    lines.extend(f"- {str(c).strip()}" for c in candidates if str(c).strip())
    return "\n".join(lines)


def maybe_inject_candidate_block(prompt: str, candidate_block: Optional[str]) -> str:
    if not candidate_block:
        return prompt
    marker = "\n\n### Input\n"
    if marker in prompt:
        return prompt.replace(marker, f"\n\n### Candidates\n{candidate_block}{marker}", 1)
    return prompt + f"\n\n### Candidates\n{candidate_block}"


def compose_prompt_with_optional_candidate_block(
    compose_text_fn,
    *,
    task_name: str,
    instruction: str,
    input_story: str,
    include_system_prompt: bool,
    candidate_block: Optional[str],
    force_candidate_block_in_prompt: bool,
) -> str:
    params = inspect.signature(compose_text_fn).parameters
    kwargs = {
        "task_name": task_name,
        "instruction": instruction,
        "input_story": input_story,
        "response": "",
        "include_system_prompt": include_system_prompt,
    }
    if "candidate_block" in params:
        kwargs["candidate_block"] = candidate_block
        return compose_text_fn(**kwargs)

    prompt = compose_text_fn(**kwargs)
    if force_candidate_block_in_prompt:
        prompt = maybe_inject_candidate_block(prompt, candidate_block)
    return prompt


def build_prompt_for_task(
    *,
    task_name: str,
    dataset_name: str,
    input_story: str,
    next_candidates: Sequence[str],
    outcome_candidates: Sequence[str],
    include_system_prompt: bool,
    build_instruction_next_activity_fn,
    build_instruction_remaining_time_fn,
    build_instruction_final_outcome_fn,
    compose_text_fn,
    build_candidate_block_fn,
    force_candidate_block_in_prompt: bool,
) -> Tuple[str, Sequence[str]]:
    if task_name == "next_activity_prediction":
        instruction = build_instruction_next_activity_fn(next_candidates)
        candidates = next_candidates
    elif task_name == "remaining_time_prediction":
        instruction = build_instruction_remaining_time_fn()
        candidates = []
    elif task_name == "final_outcome_prediction":
        instruction = build_instruction_final_outcome_fn(dataset_name, outcome_candidates)
        candidates = outcome_candidates
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    candidate_block = build_candidate_block_fn(candidates) if candidates else None
    prompt = compose_prompt_with_optional_candidate_block(
        compose_text_fn,
        task_name=task_name,
        instruction=instruction,
        input_story=input_story,
        include_system_prompt=include_system_prompt,
        candidate_block=candidate_block,
        force_candidate_block_in_prompt=force_candidate_block_in_prompt,
    )
    return prompt, candidates


def unique_nonempty_strings(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_candidate_token_trie(tokenizer, candidates: Sequence[str], response_template: str) -> Dict[Any, Any]:
    prefix_ids = tokenizer(response_template, add_special_tokens=False)["input_ids"]
    trie: Dict[Any, Any] = {}

    for candidate in unique_nonempty_strings(candidates):
        full_ids = tokenizer(response_template + candidate, add_special_tokens=False)["input_ids"]
        if prefix_ids and full_ids[: len(prefix_ids)] == prefix_ids:
            cand_ids = full_ids[len(prefix_ids):]
        else:
            cand_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if not cand_ids:
            continue
        node = trie
        for tid in cand_ids:
            tid = int(tid)
            node = node.setdefault(tid, {})
        node[TRIE_END] = candidate

    if not trie:
        raise ValueError("Failed to build a non-empty candidate trie.")
    return trie


def allowed_tokens_from_trie(
    trie: Dict[Any, Any],
    generated_token_ids: Sequence[int],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> List[int]:
    node = trie
    for tid in generated_token_ids:
        tid = int(tid)
        if tid not in node:
            fallback_id = eos_token_id if eos_token_id is not None else pad_token_id
            return [int(fallback_id)] if fallback_id is not None else []
        node = node[tid]

    allowed = [int(tid) for tid in node.keys() if tid != TRIE_END]
    if TRIE_END in node:
        if eos_token_id is not None:
            allowed.append(int(eos_token_id))
        elif pad_token_id is not None:
            allowed.append(int(pad_token_id))

    if not allowed:
        fallback_id = eos_token_id if eos_token_id is not None else pad_token_id
        return [int(fallback_id)] if fallback_id is not None else []

    return sorted(set(allowed))


def resolve_shared_candidate_universe(batch_meta: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    if not batch_meta:
        return None
    first = set(unique_nonempty_strings(batch_meta[0].get("candidates", [])))
    if not first:
        return None
    for meta in batch_meta[1:]:
        cur = set(unique_nonempty_strings(meta.get("candidates", [])))
        if cur != first:
            return None
    return sorted(first)


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_input_length: int,
    max_new_tokens: int,
    constrained_candidates: Optional[Sequence[str]] = None,
    response_template: str = DEFAULT_RESPONSE_TEMPLATE,
    use_constrained_decoding: bool = False,
) -> List[str]:
    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    device = get_model_input_device(model)
    enc = {k: v.to(device) for k, v in enc.items()}

    prompt_len = enc["input_ids"].shape[1]
    prefix_allowed_tokens_fn = None

    if use_constrained_decoding and constrained_candidates:
        trie = build_candidate_token_trie(
            tokenizer=tokenizer,
            candidates=constrained_candidates,
            response_template=response_template,
        )
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
            generated_ids = input_ids[prompt_len:].tolist()
            return allowed_tokens_from_trie(
                trie=trie,
                generated_token_ids=generated_ids,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

    generated = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        renormalize_logits=True,
    )
    continuations = generated[:, prompt_len:]
    return tokenizer.batch_decode(continuations, skip_special_tokens=True)


def ensure_prediction_writer(prediction_path: Path) -> bool:
    if prediction_path.exists() and prediction_path.stat().st_size > 0:
        return False
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    return True


def append_prediction_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    header = ensure_prediction_writer(path)
    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        index=False,
        header=header,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels_for_macro: Sequence[str],
    valid_flags: Sequence[bool],
) -> Dict[str, float]:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    invalid_rate = 1.0 - float(np.mean(np.array(valid_flags, dtype=np.float32))) if valid_flags else 0.0
    return {
        "n_samples": int(len(y_true_list)),
        "accuracy": float(accuracy_score(y_true_list, y_pred_list)) if y_true_list else float("nan"),
        "macro_precision": float(
            precision_score(
                y_true_list,
                y_pred_list,
                labels=list(labels_for_macro),
                average="macro",
                zero_division=0,
            )
        ) if y_true_list else float("nan"),
        "macro_f1": float(
            f1_score(
                y_true_list,
                y_pred_list,
                labels=list(labels_for_macro),
                average="macro",
                zero_division=0,
            )
        ) if y_true_list else float("nan"),
        "invalid_rate": invalid_rate,
    }


def compute_regression_metrics(y_true: Sequence[int], y_pred: Sequence[int], valid_flags: Sequence[bool]) -> Dict[str, float]:
    y_true_arr = np.array(list(y_true), dtype=np.float64)
    y_pred_arr = np.array(list(y_pred), dtype=np.float64)
    invalid_rate = 1.0 - float(np.mean(np.array(valid_flags, dtype=np.float32))) if valid_flags else 0.0
    if y_true_arr.size == 0:
        return {"n_samples": 0, "mae": float("nan"), "rmse": float("nan"), "invalid_rate": invalid_rate}
    return {
        "n_samples": int(y_true_arr.size),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "invalid_rate": invalid_rate,
    }


def prefix_stage(prefix_ratio: float) -> str:
    if prefix_ratio < 0.34:
        return "early"
    if prefix_ratio < 0.67:
        return "middle"
    return "late"


def compute_case_macro_metrics(task_name: str, prediction_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not prediction_rows:
        return {"n_cases": 0}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[str(row["case_id"])].append(row)
    values = []
    invalids = []
    for rows in grouped.values():
        invalids.append(float(np.mean([0.0 if r["is_valid_parse"] else 1.0 for r in rows])))
        if task_name == "remaining_time_prediction":
            vals = [abs(float(r["prediction"]) - float(r["gold"])) for r in rows]
            values.append(float(np.mean(vals)))
        else:
            vals = [1.0 if r["prediction"] == r["gold"] else 0.0 for r in rows]
            values.append(float(np.mean(vals)))
    out = {"n_cases": int(len(grouped)), "invalid_rate": float(np.mean(invalids))}
    if task_name == "remaining_time_prediction":
        out["case_macro_mae"] = float(np.mean(values))
    else:
        out["case_macro_accuracy"] = float(np.mean(values))
    return out


def compute_stage_rows(task_name: str, dataset_name: str, prediction_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for stage in STAGE_NAMES:
        stage_rows = [r for r in prediction_rows if prefix_stage(float(r["prefix_ratio"])) == stage]
        if not stage_rows:
            continue
        if task_name == "remaining_time_prediction":
            metrics = compute_regression_metrics(
                [int(r["gold"]) for r in stage_rows],
                [int(r["prediction"]) for r in stage_rows],
                [bool(r["is_valid_parse"]) for r in stage_rows],
            )
            rows_out.append(
                {
                    "dataset": dataset_name,
                    "task": task_name,
                    "stage": stage,
                    "n_samples": int(metrics["n_samples"]),
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "invalid_rate": float(metrics["invalid_rate"]),
                }
            )
        else:
            labels = sorted({str(r["gold"]) for r in stage_rows}) if task_name == "next_activity_prediction" else OUTCOME_CANDIDATES
            metrics = compute_classification_metrics(
                [str(r["gold"]) for r in stage_rows],
                [str(r["prediction"]) for r in stage_rows],
                labels,
                [bool(r["is_valid_parse"]) for r in stage_rows],
            )
            rows_out.append(
                {
                    "dataset": dataset_name,
                    "task": task_name,
                    "stage": stage,
                    "n_samples": int(metrics["n_samples"]),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_precision": float(metrics["macro_precision"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "invalid_rate": float(metrics["invalid_rate"]),
                }
            )
    return rows_out


def evaluate_single_task(
    *,
    model,
    tokenizer,
    dataset_name: str,
    task_name: str,
    prefix_iter: Iterable[Dict[str, Any]],
    include_system_prompt: bool,
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int,
    prediction_path: Optional[Path],
    progress_total: int,
    build_instruction_next_activity_fn,
    build_instruction_remaining_time_fn,
    build_instruction_final_outcome_fn,
    compose_text_fn,
    build_candidate_block_fn,
    force_candidate_block_in_prompt: bool,
    use_constrained_decoding: bool,
    response_template: str,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    y_true: List[Any] = []
    y_pred: List[Any] = []
    valid_flags: List[bool] = []
    prediction_rows: List[Dict[str, Any]] = []

    batch_prompts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    prediction_rows_buffer: List[Dict[str, Any]] = []

    pbar = tqdm(total=progress_total, desc=f"{dataset_name} | {task_name}", leave=False)

    def flush_batch() -> None:
        nonlocal batch_prompts, batch_meta, prediction_rows_buffer
        if not batch_prompts:
            return

        constrained_candidates: Optional[List[str]] = None
        constraint_applied = False
        if use_constrained_decoding and task_name != "remaining_time_prediction":
            constrained_candidates = resolve_shared_candidate_universe(batch_meta)
            constraint_applied = constrained_candidates is not None and len(constrained_candidates) > 0

        raw_outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            constrained_candidates=constrained_candidates,
            response_template=response_template,
            use_constrained_decoding=constraint_applied,
        )
        for meta, raw_output in zip(batch_meta, raw_outputs):
            gold = meta["gold"]
            if task_name == "remaining_time_prediction":
                parsed, is_valid = parse_remaining_time_prediction(raw_output)
            else:
                parsed, is_valid = parse_categorical_prediction(raw_output, meta["candidates"])
            row = {
                "dataset": dataset_name,
                "task": task_name,
                "case_id": meta["case_id"],
                "prefix_len": meta["prefix_len"],
                "trace_len": meta["trace_len"],
                "prefix_ratio": meta["prefix_ratio"],
                "story_segment_start": meta["story_segment_start"],
                "story_n_events": meta["story_n_events"],
                "gold": gold,
                "prediction": parsed,
                "raw_output": raw_output.strip(),
                "is_valid_parse": bool(is_valid),
                "is_correct": bool(parsed == gold),
                "candidate_count": int(len(meta["candidates"])),
                "used_constrained_decoding": bool(constraint_applied),
            }
            y_true.append(gold)
            y_pred.append(parsed)
            valid_flags.append(is_valid)
            prediction_rows.append(row)
            prediction_rows_buffer.append(row)
            pbar.update(1)
        if prediction_path is not None and prediction_rows_buffer:
            append_prediction_rows(prediction_path, prediction_rows_buffer)
            prediction_rows_buffer = []
        batch_prompts = []
        batch_meta = []

    for example in prefix_iter:
        if task_name == "next_activity_prediction":
            gold = example["next_activity"]
        elif task_name == "remaining_time_prediction":
            gold = int(example["remaining_time_days"])
        else:
            gold = example["final_outcome"]

        prompt, candidates = build_prompt_for_task(
            task_name=task_name,
            dataset_name=dataset_name,
            input_story=example["input_story"],
            next_candidates=example["next_candidates"],
            outcome_candidates=example["outcome_candidates"],
            include_system_prompt=include_system_prompt,
            build_instruction_next_activity_fn=build_instruction_next_activity_fn,
            build_instruction_remaining_time_fn=build_instruction_remaining_time_fn,
            build_instruction_final_outcome_fn=build_instruction_final_outcome_fn,
            compose_text_fn=compose_text_fn,
            build_candidate_block_fn=build_candidate_block_fn,
            force_candidate_block_in_prompt=force_candidate_block_in_prompt,
        )
        batch_prompts.append(prompt)
        batch_meta.append(
            {
                "case_id": example["case_id"],
                "prefix_len": example["prefix_len"],
                "trace_len": example["trace_len"],
                "prefix_ratio": example["prefix_ratio"],
                "story_segment_start": example["story_segment_start"],
                "story_n_events": example["story_n_events"],
                "gold": gold,
                "candidates": list(candidates),
            }
        )
        if len(batch_prompts) >= batch_size:
            flush_batch()

    flush_batch()
    pbar.close()

    if task_name == "remaining_time_prediction":
        sample_metrics = compute_regression_metrics(y_true, y_pred, valid_flags)
    elif task_name == "final_outcome_prediction":
        sample_metrics = compute_classification_metrics(y_true, y_pred, OUTCOME_CANDIDATES, valid_flags)
    else:
        labels_for_macro = sorted(set(y_true))
        sample_metrics = compute_classification_metrics(y_true, y_pred, labels_for_macro, valid_flags)

    case_macro_metrics = compute_case_macro_metrics(task_name, prediction_rows)
    stage_rows = compute_stage_rows(task_name, dataset_name, prediction_rows)
    return sample_metrics, case_macro_metrics, stage_rows, prediction_rows


def build_dataset_summary_row(
    dataset_name: str,
    task_name: str,
    metrics: Dict[str, float],
    n_cases_train: int,
    n_cases_test: int,
    n_prefixes_test: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset_name,
        "task": task_name,
        "baseline_train_cases": int(n_cases_train),
        "zero_shot_test_cases": int(n_cases_test),
        "zero_shot_test_prefixes": int(n_prefixes_test),
        "n_samples": int(metrics.get("n_samples", 0)),
        "invalid_rate": float(metrics.get("invalid_rate", float("nan"))),
    }
    if task_name == "remaining_time_prediction":
        row["mae"] = float(metrics.get("mae", float("nan")))
        row["rmse"] = float(metrics.get("rmse", float("nan")))
    else:
        row["accuracy"] = float(metrics.get("accuracy", float("nan")))
        row["macro_precision"] = float(metrics.get("macro_precision", float("nan")))
        row["macro_f1"] = float(metrics.get("macro_f1", float("nan")))
    return row


def build_case_macro_row(dataset_name: str, task_name: str, case_metrics: Dict[str, float]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset_name,
        "task": task_name,
        "n_cases": int(case_metrics.get("n_cases", 0)),
        "invalid_rate": float(case_metrics.get("invalid_rate", float("nan"))),
    }
    if task_name == "remaining_time_prediction":
        row["case_macro_mae"] = float(case_metrics.get("case_macro_mae", float("nan")))
    else:
        row["case_macro_accuracy"] = float(case_metrics.get("case_macro_accuracy", float("nan")))
    return row


def build_overall_rows(metric_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overall_rows: List[Dict[str, Any]] = []
    for task_name in VALID_TASKS:
        task_rows = [row for row in metric_rows if row["task"] == task_name]
        if not task_rows:
            continue
        overall: Dict[str, Any] = {
            "dataset": "__dataset_macro__",
            "task": task_name,
            "baseline_train_cases": int(sum(row["baseline_train_cases"] for row in task_rows)),
            "zero_shot_test_cases": int(sum(row["zero_shot_test_cases"] for row in task_rows)),
            "zero_shot_test_prefixes": int(sum(row["zero_shot_test_prefixes"] for row in task_rows)),
            "n_samples": int(sum(row["n_samples"] for row in task_rows)),
            "invalid_rate": float(np.mean([row["invalid_rate"] for row in task_rows])),
        }
        if task_name == "remaining_time_prediction":
            overall["mae"] = float(np.mean([row["mae"] for row in task_rows]))
            overall["rmse"] = float(np.mean([row["rmse"] for row in task_rows]))
        else:
            overall["accuracy"] = float(np.mean([row["accuracy"] for row in task_rows]))
            overall["macro_precision"] = float(np.mean([row["macro_precision"] for row in task_rows]))
            overall["macro_f1"] = float(np.mean([row["macro_f1"] for row in task_rows]))
        overall_rows.append(overall)
    return overall_rows


def print_metric_rows(metric_rows: Sequence[Dict[str, Any]]) -> None:
    print("\nPer-dataset results:")
    for row in metric_rows:
        if row["task"] == "remaining_time_prediction":
            print(
                f"  [{row['dataset']}] {row['task']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, "
                f"invalid_rate={row['invalid_rate']:.4f}, samples={row['n_samples']}"
            )
        else:
            print(
                f"  [{row['dataset']}] {row['task']}: Acc={row['accuracy']:.4f}, "
                f"Macro-P={row['macro_precision']:.4f}, Macro-F1={row['macro_f1']:.4f}, "
                f"invalid_rate={row['invalid_rate']:.4f}, samples={row['n_samples']}"
            )


def main() -> None:
    args = parse_args()
    set_all_seeds(args.random_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    story_mod, ft_mod = import_pipeline_modules(args.story_module, args.finetune_module)
    TEST_DATASETS = getattr(story_mod, "TEST_DATASETS")
    DEFAULT_CASE_ID_CANDIDATES = getattr(ft_mod, "DEFAULT_CASE_ID_CANDIDATES", getattr(story_mod, "DEFAULT_CASE_ID_CANDIDATES"))
    DEFAULT_ACTIVITY_CANDIDATES = getattr(ft_mod, "DEFAULT_ACTIVITY_CANDIDATES", getattr(story_mod, "DEFAULT_ACTIVITY_CANDIDATES"))
    DEFAULT_TIMESTAMP_CANDIDATES = getattr(ft_mod, "DEFAULT_TIMESTAMP_CANDIDATES", getattr(story_mod, "DEFAULT_TIMESTAMP_CANDIDATES"))
    normalize_dataset_name = getattr(ft_mod, "normalize_dataset_name", getattr(story_mod, "normalize_dataset_name"))
    find_dataset_file = getattr(ft_mod, "find_dataset_file", getattr(story_mod, "find_dataset_file"))
    load_dataset_csv = getattr(ft_mod, "load_dataset_csv", getattr(story_mod, "load_dataset_csv"))
    resolve_column = getattr(ft_mod, "resolve_column", getattr(story_mod, "resolve_column"))
    build_case_story_fn = getattr(ft_mod, "build_case_story", getattr(story_mod, "build_case_story"))
    slice_prefix_fn = getattr(ft_mod, "slice_prefix_for_story", fallback_slice_prefix_for_story)
    LABEL_COL_CANDIDATES = getattr(ft_mod, "LABEL_COL_CANDIDATES")
    REMAINING_TIME_CANDIDATES = getattr(ft_mod, "REMAINING_TIME_CANDIDATES")
    BASE_RULE_DESCRIPTIONS = getattr(ft_mod, "BASE_RULE_DESCRIPTIONS")
    build_case_records = getattr(ft_mod, "build_case_records")
    build_instruction_next_activity_fn = getattr(ft_mod, "build_instruction_next_activity")
    build_instruction_remaining_time_fn = getattr(ft_mod, "build_instruction_remaining_time")
    build_instruction_final_outcome_fn = getattr(ft_mod, "build_instruction_final_outcome")
    compose_text_fn = getattr(ft_mod, "compose_text")
    build_candidate_block_fn = getattr(ft_mod, "build_candidate_block", default_build_candidate_block)
    get_remaining_time_label_fn = getattr(ft_mod, "get_remaining_time_label")

    model, tokenizer = load_model_and_tokenizer(args)
    prediction_path = output_dir / args.prediction_file_name if args.save_predictions else None
    if prediction_path is not None and prediction_path.exists():
        prediction_path.unlink()

    metric_rows: List[Dict[str, Any]] = []
    case_macro_rows: List[Dict[str, Any]] = []
    stage_metric_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    dataset_names = args.dataset_names or TEST_DATASETS
    dataset_names = [normalize_dataset_name(name) for name in dataset_names]
    dataset_names = list(dict.fromkeys(dataset_names))

    for dataset_name in dataset_names:
        csv_path = find_dataset_file(args.input_dir, dataset_name)
        full_df, case_id_col, activity_col, timestamp_col = load_dataset_csv(
            csv_path=csv_path,
            case_id_candidates=DEFAULT_CASE_ID_CANDIDATES,
            activity_candidates=DEFAULT_ACTIVITY_CANDIDATES,
            timestamp_candidates=DEFAULT_TIMESTAMP_CANDIDATES,
        )
        label_col = resolve_column(full_df, LABEL_COL_CANDIDATES, required=True)
        remaining_time_col = resolve_column(full_df, REMAINING_TIME_CANDIDATES, required=False)

        cases = build_case_records(
            df=full_df,
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            label_col=label_col,
        )
        case_ids = list(cases.keys())
        if not case_ids:
            print(f"[{dataset_name}] skipped because no eligible case with length >= {args.min_prefix_len + 1} was found.")
            continue

        case_label_map = {case_id: canonicalize_case_label(cases[case_id]["label"]) for case_id in case_ids}
        stable_offset = int(hashlib.md5(dataset_name.encode("utf-8")).hexdigest()[:8], 16)
        dataset_rng = random.Random(args.random_seed + stable_offset)
        baseline_train_case_ids, zero_shot_test_case_ids = stratified_case_split(
            case_ids=case_ids,
            case_label_map=case_label_map,
            test_ratio=args.test_ratio,
            rng=dataset_rng,
        )
        if not zero_shot_test_case_ids:
            print(f"[{dataset_name}] skipped because the test split is empty.")
            continue

        if args.save_case_splits:
            save_case_split_files(
                output_dir=output_dir,
                dataset_name=dataset_name,
                full_df=full_df,
                case_id_col=case_id_col,
                baseline_train_case_ids=baseline_train_case_ids,
                zero_shot_test_case_ids=zero_shot_test_case_ids,
            )

        dataset_activity_candidates = sorted({str(x).strip() for x in full_df[activity_col].dropna().tolist() if str(x).strip()})
        n_prefixes_test = count_test_prefixes(cases, zero_shot_test_case_ids, args.min_prefix_len)
        split_rows.append(
            {
                "dataset": dataset_name,
                "baseline_train_cases": len(baseline_train_case_ids),
                "zero_shot_test_cases": len(zero_shot_test_case_ids),
                "zero_shot_test_prefixes": n_prefixes_test,
                "n_activities": len(dataset_activity_candidates),
                "success_rule": BASE_RULE_DESCRIPTIONS.get(normalize_dataset_name(dataset_name)),
                "include_system_prompt": args.include_system_prompt,
                "include_dataset_intro": args.include_dataset_intro,
                "quote_activities": args.quote_activities,
                "shuffle_candidates": args.shuffle_candidates,
                "max_prefix_story_events": args.max_prefix_story_events,
                "min_prefix_len": args.min_prefix_len,
                "use_constrained_decoding": args.use_constrained_decoding,
            }
        )
        print(
            f"[{dataset_name}] baseline_train_cases={len(baseline_train_case_ids)} "
            f"zero_shot_test_cases={len(zero_shot_test_case_ids)} "
            f"zero_shot_test_prefixes={n_prefixes_test} activities={len(dataset_activity_candidates)}"
        )

        for task_name in args.tasks:
            max_new_tokens = {
                "next_activity_prediction": args.max_new_tokens_next,
                "remaining_time_prediction": args.max_new_tokens_time,
                "final_outcome_prediction": args.max_new_tokens_outcome,
            }[task_name]
            prefix_iter = iter_prefix_examples(
                dataset_name=dataset_name,
                cases=cases,
                case_ids=zero_shot_test_case_ids,
                activity_col=activity_col,
                timestamp_col=timestamp_col,
                remaining_time_col=remaining_time_col,
                include_dataset_intro=args.include_dataset_intro,
                quote_activities=args.quote_activities,
                max_prefix_story_events=args.max_prefix_story_events,
                min_prefix_len=args.min_prefix_len,
                shuffle_candidates=args.shuffle_candidates,
                random_seed=args.random_seed,
                dataset_activity_candidates=dataset_activity_candidates,
                slice_prefix_fn=slice_prefix_fn,
                build_case_story_fn=build_case_story_fn,
                get_remaining_time_label_fn=get_remaining_time_label_fn,
            )
            sample_metrics, case_metrics, stage_rows, _ = evaluate_single_task(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                task_name=task_name,
                prefix_iter=prefix_iter,
                include_system_prompt=args.include_system_prompt,
                max_input_length=args.max_input_length,
                max_new_tokens=max_new_tokens,
                batch_size=args.batch_size,
                prediction_path=prediction_path,
                progress_total=n_prefixes_test,
                build_instruction_next_activity_fn=build_instruction_next_activity_fn,
                build_instruction_remaining_time_fn=build_instruction_remaining_time_fn,
                build_instruction_final_outcome_fn=build_instruction_final_outcome_fn,
                compose_text_fn=compose_text_fn,
                build_candidate_block_fn=build_candidate_block_fn,
                force_candidate_block_in_prompt=args.force_candidate_block_in_prompt,
                use_constrained_decoding=args.use_constrained_decoding,
                response_template=args.response_template,
            )
            metric_rows.append(
                build_dataset_summary_row(
                    dataset_name,
                    task_name,
                    sample_metrics,
                    len(baseline_train_case_ids),
                    len(zero_shot_test_case_ids),
                    n_prefixes_test,
                )
            )
            case_macro_rows.append(build_case_macro_row(dataset_name, task_name, case_metrics))
            stage_metric_rows.extend(stage_rows)

    if not metric_rows:
        raise RuntimeError("No metric row was generated. Please check the input datasets and split settings.")

    overall_rows = build_overall_rows(metric_rows)
    pd.DataFrame(split_rows).to_csv(output_dir / "case_split_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics_by_dataset_task.csv", index=False, encoding="utf-8")
    pd.DataFrame(case_macro_rows).to_csv(output_dir / "metrics_case_macro_by_dataset_task.csv", index=False, encoding="utf-8")
    pd.DataFrame(stage_metric_rows).to_csv(output_dir / "metrics_by_dataset_task_stage.csv", index=False, encoding="utf-8")
    pd.DataFrame(overall_rows).to_csv(output_dir / "metrics_dataset_macro.csv", index=False, encoding="utf-8")
    with (output_dir / "metrics_dataset_macro.json").open("w", encoding="utf-8") as f:
        json.dump(overall_rows, f, ensure_ascii=False, indent=2)
    with (output_dir / "eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print_metric_rows(metric_rows)
    print("\nDataset-macro summary:")
    for row in overall_rows:
        if row["task"] == "remaining_time_prediction":
            print(
                f"  {row['task']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, "
                f"invalid_rate={row['invalid_rate']:.4f}, samples={row['n_samples']}"
            )
        else:
            print(
                f"  {row['task']}: Acc={row['accuracy']:.4f}, Macro-P={row['macro_precision']:.4f}, "
                f"Macro-F1={row['macro_f1']:.4f}, invalid_rate={row['invalid_rate']:.4f}, samples={row['n_samples']}"
            )
    print(f"\nSaved split summary to {output_dir / 'case_split_summary.csv'}")
    print(f"Saved per-dataset metrics to {output_dir / 'metrics_by_dataset_task.csv'}")
    print(f"Saved case-macro metrics to {output_dir / 'metrics_case_macro_by_dataset_task.csv'}")
    print(f"Saved stage metrics to {output_dir / 'metrics_by_dataset_task_stage.csv'}")
    print(f"Saved dataset-macro metrics to {output_dir / 'metrics_dataset_macro.csv'}")
    if prediction_path is not None:
        print(f"Saved row-level predictions to {prediction_path}")


if __name__ == "__main__":
    main()
