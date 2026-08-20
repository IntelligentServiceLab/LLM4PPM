from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from .story import (
        TRAIN_DATASETS,
        DEFAULT_CASE_ID_CANDIDATES,
        DEFAULT_ACTIVITY_CANDIDATES,
        DEFAULT_TIMESTAMP_CANDIDATES,
        normalize_dataset_name,
        find_dataset_file,
        load_dataset_csv,
        ensure_time_features,
        build_case_story,
        resolve_column,
    )
except ImportError:  # pragma: no cover
    from story import (
        TRAIN_DATASETS,
        DEFAULT_CASE_ID_CANDIDATES,
        DEFAULT_ACTIVITY_CANDIDATES,
        DEFAULT_TIMESTAMP_CANDIDATES,
        normalize_dataset_name,
        find_dataset_file,
        load_dataset_csv,
        ensure_time_features,
        build_case_story,
        resolve_column,
    )


RAW_RULE_DESCRIPTIONS: Dict[str, str] = {
    "BPIC2013I": "the case finishes with complete incident",
    "BPIC2013O": "the case finishes with complete problem",
    "BPIC2013C": "the case finishes with complete problem",
    "BPIC2017O": "the case contains accept offer",
    "BPIC2019": "the case contains clear invoice",
    "BPIC2020D": "the case contains handle payment",
    "BPIC2020I": "the case contains handle payment",
    "BPIC2020Pe": "the case contains handle payment",
    "BPIC2020Pr": "the case contains handle payment",
    "BPIC2020R": "the case contains handle payment",
    "BPIC2012A": "the case contains accept application",
    "BPIC2012O": "the case contains accept offer",
    "BPIC2012W": "the work-item case belongs to a loan application that is eventually accepted",
    "Hospital-billing": "the case is finalized or issued and never reopened",
    "Receipt": "the case never enters hold, stop, or unlicensed branches",
    "Sepsis": "the patient does not return to emergency within 28 days after release",
    "Helpdesk": "the ticket is closed and never marked duplicate or invalid",
    "Production": "the rejected quantity is zero for the whole case",
    "Road_Traffic_Fine": "the fine is paid and never sent for debt collection",
    "Service_process": "REPAIR_IN_TIME_5D is true",
    "Service-process": "REPAIR_IN_TIME_5D is true",
}
BASE_RULE_DESCRIPTIONS: Dict[str, str] = {
    normalize_dataset_name(k): v for k, v in RAW_RULE_DESCRIPTIONS.items()
}

SYS_PROMPT_NEXT = (
    "You are a next activity predictor specialized in predictive process monitoring. "
    "Your task is to analyze the execution case and predict the next activity."
)
SYS_PROMPT_TIME = (
    "You are a process time estimator specialized in predictive process monitoring. "
    "Your task is to analyze the execution case and predict the remaining time."
)
SYS_PROMPT_OUTCOME = (
    "You are a process outcome assessor specialized in predictive process monitoring. "
    "Your task is to analyze the execution case and predict the final outcome."
)

LABEL_COL_CANDIDATES = [
    "label",
    "Label",
    "case_label",
    "case:label",
    "final_label",
    "outcome",
    "target",
]

REMAINING_TIME_CANDIDATES = [
    "remaining_time",
    "remainingtime",
    "remain_time",
    "remaining_days",
    "remaining_day",
    "rest_time",
]

CaseRecord = Dict[str, Any]
PrefixPlan = Dict[Any, List[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unified multi-task fine-tuning data for PPM.")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing per-dataset CSV logs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train/validation files.")
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=TRAIN_DATASETS,
        help="Datasets used to build the fine-tuning set. Defaults to TRAIN_DATASETS in llm4ppm.story.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        choices=["csv", "jsonl", "both"],
        help="Output format for train/validation data and metadata.",
    )

    parser.add_argument(
        "--include-system-prompt",
        dest="include_system_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether each sample starts with a short task-specific system prompt.",
    )
    parser.add_argument(
        "--include-dataset-intro",
        dest="include_dataset_intro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the prefix story starts with the same process-level intro sentence used in pretraining.",
    )
    parser.add_argument(
        "--quote-activities",
        dest="quote_activities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether activity names in the story are wrapped in quotation marks.",
    )

    parser.add_argument(
        "--max_prefix_story_events",
        type=int,
        default=40,
        help=(
            "Maximum number of events retained in the story shown under ### Input. "
            "If a sampled prefix is longer, only the latest max_prefix_story_events events are kept, "
            "while elapsed_time stays absolute to the case start. Use <=0 to keep the whole prefix story."
        ),
    )

    parser.add_argument(
        "--max_cases_per_dataset",
        type=int,
        default=1500,
        help="Variant-aware cap for the number of cases kept from each dataset. Use <=0 to keep all cases.",
    )
    parser.add_argument(
        "--max_prefixes_per_case",
        type=int,
        default=6,
        help="Maximum number of prefix lengths sampled per case. Use <=0 to keep all eligible prefixes.",
    )
    parser.add_argument(
        "--max_prefix_instances_per_dataset",
        type=int,
        default=8000,
        help=(
            "Cap the number of sampled prefixes per dataset before expanding to 3 tasks. "
            "Use <=0 to disable this cap."
        ),
    )
    parser.add_argument(
        "--prefix_anchor_ratios",
        type=str,
        default="0.1,0.2,0.35,0.5,0.65,0.8,0.9",
        help="Comma-separated anchor ratios used when a long case is prefix-sampled.",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.02,
        help="Case-level validation ratio within each dataset. 0.0 means only generate train files.",
    )
    parser.add_argument("--random_seed", type=int, default=3407)

    return parser.parse_args()



def parse_ratio_list(text: str) -> List[float]:
    ratios: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            continue
        ratios.append(value)
    if not ratios:
        ratios = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
    return ratios



def canonicalize_label(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Encountered NaN in label column.")

    text = str(value).strip().lower()
    if text in {"succeed", "success", "successful", "1", "true", "yes"}:
        return "success"
    if text in {"failure", "fail", "failed", "0", "false", "no"}:
        return "failure"

    raise ValueError(f"Unsupported label value: {value!r}. Expected succeed/failure or equivalent.")



def get_case_label(case_df: pd.DataFrame, label_col: str) -> str:
    if label_col not in case_df.columns:
        raise KeyError(f"Label column '{label_col}' not found.")

    values = [canonicalize_label(v) for v in case_df[label_col].dropna().tolist()]
    if not values:
        raise ValueError("No non-null label found for a case.")

    unique_values = sorted(set(values))
    if len(unique_values) > 1:
        raise ValueError(f"A case contains multiple labels: {unique_values}")
    return unique_values[0]



def get_remaining_time_label(
    case_df: pd.DataFrame,
    prefix_len: int,
    timestamp_col: Optional[str],
    remaining_time_col: Optional[str],
) -> int:
    row_idx = prefix_len - 1

    if remaining_time_col is not None and remaining_time_col in case_df.columns:
        raw_value = case_df.iloc[row_idx][remaining_time_col]
        if not pd.isna(raw_value):
            return max(0, int(math.floor(float(raw_value))))

    enriched = ensure_time_features(case_df, timestamp_col)
    last_elapsed = float(enriched["elapsed_time"].iloc[-1])
    cur_elapsed = float(enriched["elapsed_time"].iloc[row_idx])
    remaining = max(0.0, last_elapsed - cur_elapsed)
    return int(math.floor(remaining))



def compute_variant_key(case_df: pd.DataFrame, activity_col: str) -> Tuple[str, ...]:
    return tuple(str(x).strip() for x in case_df[activity_col].tolist())



def maybe_cap_cases_variant_balanced(
    cases: Dict[Any, CaseRecord],
    max_cases_per_dataset: int,
    rng: random.Random,
) -> List[Any]:
    case_ids = list(cases.keys())
    if max_cases_per_dataset <= 0 or len(case_ids) <= max_cases_per_dataset:
        rng.shuffle(case_ids)
        return case_ids

    buckets: Dict[Tuple[str, ...], List[Any]] = defaultdict(list)
    for case_id, record in cases.items():
        buckets[record["variant"]].append(case_id)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    active_variants = list(buckets.keys())
    selected: List[Any] = []

    while active_variants and len(selected) < max_cases_per_dataset:
        rng.shuffle(active_variants)
        next_active: List[Tuple[str, ...]] = []
        for variant in active_variants:
            bucket = buckets[variant]
            if bucket:
                selected.append(bucket.pop())
                if len(selected) >= max_cases_per_dataset:
                    break
            if bucket:
                next_active.append(variant)
        active_variants = next_active

    rng.shuffle(selected)
    return selected



def split_case_ids_variant_stratified(
    case_ids: List[Any],
    cases: Dict[Any, CaseRecord],
    validation_ratio: float,
    rng: random.Random,
) -> Tuple[List[Any], List[Any]]:
    if validation_ratio <= 0 or len(case_ids) < 2:
        return case_ids, []

    buckets: Dict[Tuple[str, ...], List[Any]] = defaultdict(list)
    for case_id in case_ids:
        buckets[cases[case_id]["variant"]].append(case_id)

    train_ids: List[Any] = []
    val_ids: List[Any] = []

    for variant in sorted(buckets.keys(), key=lambda x: (len(x), x)):
        bucket = list(buckets[variant])
        rng.shuffle(bucket)

        if len(bucket) < 5:
            train_ids.extend(bucket)
            continue

        n_val = int(round(len(bucket) * validation_ratio))
        n_val = max(1, n_val)
        n_val = min(n_val, len(bucket) - 1)

        val_ids.extend(bucket[:n_val])
        train_ids.extend(bucket[n_val:])

    if not train_ids and val_ids:
        train_ids.append(val_ids.pop())
    elif not val_ids and len(train_ids) > 1:
        val_ids.append(train_ids.pop())

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids



def evenly_subsample_sorted(values: List[int], n_keep: int) -> List[int]:
    if len(values) <= n_keep:
        return values
    if n_keep <= 1:
        return [values[-1]]

    picked: List[int] = []
    last_idx = len(values) - 1
    for i in range(n_keep):
        idx = round(i * last_idx / (n_keep - 1))
        picked.append(values[idx])
    return sorted(set(picked))



def select_prefix_lengths(
    trace_len: int,
    max_prefixes_per_case: int,
    anchor_ratios: Sequence[float],
    rng: random.Random,
) -> List[int]:
    eligible = list(range(2, trace_len))
    if not eligible:
        return []

    if max_prefixes_per_case <= 0 or len(eligible) <= max_prefixes_per_case:
        return eligible

    anchors = {2, trace_len - 1}
    for ratio in anchor_ratios:
        p = int(round(trace_len * ratio))
        p = max(2, min(trace_len - 1, p))
        anchors.add(p)

    anchor_list = sorted(a for a in anchors if 2 <= a <= trace_len - 1)
    if len(anchor_list) > max_prefixes_per_case:
        anchor_list = evenly_subsample_sorted(anchor_list, max_prefixes_per_case)

    selected = set(anchor_list)
    if len(selected) < max_prefixes_per_case:
        remaining = [p for p in eligible if p not in selected]
        rng.shuffle(remaining)
        for p in remaining:
            selected.add(p)
            if len(selected) >= max_prefixes_per_case:
                break

    return sorted(selected)



def shuffled_copy(items: Sequence[str], rng: random.Random) -> List[str]:
    copied = list(items)
    rng.shuffle(copied)
    return copied



def format_candidate_list(candidates: Sequence[str]) -> str:
    return json.dumps(list(candidates), ensure_ascii=False)



def build_instruction_next_activity(candidates: Sequence[str]) -> str:
    return f"Please carefully analyze the process story and predict the next activity of the case. You must choose only one activity name from the following candidates: {format_candidate_list(candidates)}."



def build_instruction_remaining_time() -> str:
    return "Please carefully analyze the process story and predict the remaining time until the case is completed. Return only an integer to represent the remaining time of the case in days."



def build_instruction_final_outcome(dataset_name: str, candidates: Sequence[str]) -> str:
    key = normalize_dataset_name(dataset_name)
    if key not in BASE_RULE_DESCRIPTIONS:
        raise KeyError(
            f"No success-rule description found for dataset '{dataset_name}' (normalized as '{key}'). "
            "Please add it to BASE_RULE_DESCRIPTIONS."
        )
    rule_text = BASE_RULE_DESCRIPTIONS[key]
    return f"Please carefully analyze the process story and predict the final outcome of the case. In this dataset, a case is labeled \"succeed\" when {rule_text}. Otherwise, it is labeled \"failure\". You must choose only one from the following outcome candidates: {format_candidate_list(candidates)}."



def get_system_prompt(task_name: str) -> str:
    if task_name == "next_activity_prediction":
        return SYS_PROMPT_NEXT
    if task_name == "remaining_time_prediction":
        return SYS_PROMPT_TIME
    if task_name == "final_outcome_prediction":
        return SYS_PROMPT_OUTCOME
    return "You are a predictive process monitoring assistant."



def compose_text(task_name: str, instruction: str, input_story: str, response: str, include_system_prompt: bool) -> str:
    parts = []
    if include_system_prompt:
        parts.append(get_system_prompt(task_name))
    parts.append("### Instruction\n" + instruction)
    parts.append("### Input\n" + input_story)
    parts.append("### Response\n" + response)
    return "\n\n".join(parts)



def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], mode: str = "w") -> int:
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count



def write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8", quoting=csv.QUOTE_ALL)
    return len(df)



def serialize_meta_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        candidates = copied.get("candidates")
        if isinstance(candidates, list):
            copied["candidates"] = json.dumps(candidates, ensure_ascii=False)
        serialized.append(copied)
    return serialized



def write_rows(path: Path, rows: List[Dict[str, Any]], output_format: str, is_meta: bool = False) -> int:
    total = 0
    rows_to_write = serialize_meta_rows(rows) if is_meta else rows

    if output_format in {"csv", "both"}:
        csv_path = path.with_suffix(".csv")
        total = write_csv_rows(csv_path, rows_to_write)
    if output_format in {"jsonl", "both"}:
        jsonl_path = path.with_suffix(".jsonl")
        total = write_jsonl(jsonl_path, rows_to_write, mode="a")
    return total



def build_case_records(
    df: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    timestamp_col: Optional[str],
    label_col: str,
) -> Dict[Any, CaseRecord]:
    cases: Dict[Any, CaseRecord] = {}
    for case_id, case_df in df.groupby(case_id_col, sort=False):
        sorted_case = ensure_time_features(case_df, timestamp_col).reset_index(drop=True)
        if len(sorted_case) < 3:
            continue

        label = get_case_label(sorted_case, label_col)
        variant = compute_variant_key(sorted_case, activity_col)
        cases[case_id] = {
            "case_df": sorted_case,
            "trace_len": len(sorted_case),
            "label": label,
            "variant": variant,
        }
    return cases



def build_prefix_plan(
    case_ids: Sequence[Any],
    cases: Dict[Any, CaseRecord],
    max_prefixes_per_case: int,
    anchor_ratios: Sequence[float],
    rng: random.Random,
) -> PrefixPlan:
    prefix_plan: PrefixPlan = {}
    for case_id in case_ids:
        trace_len = int(cases[case_id]["trace_len"])
        selected = select_prefix_lengths(
            trace_len=trace_len,
            max_prefixes_per_case=max_prefixes_per_case,
            anchor_ratios=anchor_ratios,
            rng=rng,
        )
        if selected:
            prefix_plan[case_id] = selected
    return prefix_plan



def maybe_cap_prefix_plan(
    prefix_plan: PrefixPlan,
    max_prefix_instances_per_dataset: int,
    rng: random.Random,
) -> PrefixPlan:
    total_prefixes = sum(len(v) for v in prefix_plan.values())
    if max_prefix_instances_per_dataset <= 0 or total_prefixes <= max_prefix_instances_per_dataset:
        return prefix_plan

    pools: Dict[Any, List[int]] = {}
    for case_id, prefixes in prefix_plan.items():
        copied = list(prefixes)
        rng.shuffle(copied)
        pools[case_id] = copied

    selected: PrefixPlan = {case_id: [] for case_id in prefix_plan}
    active_case_ids = [case_id for case_id, prefixes in pools.items() if prefixes]
    n_selected = 0

    while active_case_ids and n_selected < max_prefix_instances_per_dataset:
        rng.shuffle(active_case_ids)
        next_active: List[Any] = []
        for case_id in active_case_ids:
            pool = pools[case_id]
            if pool:
                selected[case_id].append(pool.pop())
                n_selected += 1
                if n_selected >= max_prefix_instances_per_dataset:
                    break
            if pool:
                next_active.append(case_id)
        active_case_ids = next_active

    compacted: PrefixPlan = {}
    for case_id, prefixes in selected.items():
        if prefixes:
            compacted[case_id] = sorted(set(prefixes))
    return compacted



def count_prefix_instances(prefix_plan: PrefixPlan) -> int:
    return sum(len(v) for v in prefix_plan.values())



def slice_prefix_for_story(
    case_df: pd.DataFrame,
    prefix_len: int,
    max_prefix_story_events: int,
) -> Tuple[pd.DataFrame, int]:
    prefix_df = case_df.iloc[:prefix_len].copy()
    segment_start = 0

    if max_prefix_story_events > 0 and len(prefix_df) > max_prefix_story_events:
        segment_start = len(prefix_df) - max_prefix_story_events
        prefix_df = prefix_df.iloc[segment_start:].copy()

    return prefix_df.reset_index(drop=True), int(segment_start)



def sample_rows_for_split(
    split_name: str,
    dataset_name: str,
    prefix_plan: PrefixPlan,
    cases: Dict[Any, CaseRecord],
    activity_col: str,
    timestamp_col: Optional[str],
    remaining_time_col: Optional[str],
    dataset_activity_candidates: Sequence[str],
    include_dataset_intro: bool,
    quote_activities: bool,
    include_system_prompt: bool,
    max_prefix_story_events: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    data_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    stats = Counter()

    outcome_candidates = ["succeed", "failure"]

    for case_id in prefix_plan:
        record = cases[case_id]
        case_df = record["case_df"]
        trace_len = record["trace_len"]
        case_label = record["label"]
        variant_key = " || ".join(record["variant"])

        for prefix_len in prefix_plan[case_id]:
            observed_prefix_df = case_df.iloc[:prefix_len].copy().reset_index(drop=True)
            prefix_story_df, segment_start = slice_prefix_for_story(
                case_df=case_df,
                prefix_len=prefix_len,
                max_prefix_story_events=max_prefix_story_events,
            )
            try:
                prefix_story = build_case_story(
                    case_df=prefix_story_df,
                    dataset_name=dataset_name,
                    activity_col=activity_col,
                    timestamp_col=timestamp_col,
                    include_dataset_intro=include_dataset_intro,
                    quote_activities=quote_activities,
                    segment_start=segment_start,
                    trace_context_df=observed_prefix_df,
                )
            except TypeError:
                prefix_story = build_case_story(
                    case_df=prefix_story_df,
                    dataset_name=dataset_name,
                    activity_col=activity_col,
                    timestamp_col=timestamp_col,
                    include_dataset_intro=include_dataset_intro,
                    quote_activities=quote_activities,
                    segment_start=segment_start,
                )

            next_activity = str(case_df.iloc[prefix_len][activity_col]).strip()
            remaining_days = get_remaining_time_label(
                case_df=case_df,
                prefix_len=prefix_len,
                timestamp_col=timestamp_col,
                remaining_time_col=remaining_time_col,
            )

            next_candidates = shuffled_copy(dataset_activity_candidates, rng)
            outcome_candidates_cur = shuffled_copy(outcome_candidates, rng)

            task_payloads = [
                (
                    "next_activity_prediction",
                    build_instruction_next_activity(next_candidates),
                    next_activity,
                    next_candidates,
                ),
                (
                    "remaining_time_prediction",
                    build_instruction_remaining_time(),
                    str(int(remaining_days)),
                    None,
                ),
                (
                    "final_outcome_prediction",
                    build_instruction_final_outcome(dataset_name, outcome_candidates_cur),
                    case_label,
                    outcome_candidates_cur,
                ),
            ]

            for task_name, instruction, response, candidates in task_payloads:
                text = compose_text(
                    task_name=task_name,
                    instruction=instruction,
                    input_story=prefix_story,
                    response=response,
                    include_system_prompt=include_system_prompt,
                )
                data_rows.append({"text": text})
                meta_rows.append(
                    {
                        "split": split_name,
                        "dataset": dataset_name,
                        "case_id": str(case_id),
                        "task": task_name,
                        "prefix_len": int(prefix_len),
                        "trace_len": int(trace_len),
                        "prefix_ratio": round(float(prefix_len) / max(float(trace_len), 1.0), 4),
                        "story_segment_start": int(segment_start),
                        "story_n_events": int(len(prefix_story_df)),
                        "response": response,
                        "next_activity": next_activity,
                        "remaining_time_days": int(remaining_days),
                        "final_outcome": case_label,
                        "candidates": candidates,
                        "rule_description": BASE_RULE_DESCRIPTIONS.get(normalize_dataset_name(dataset_name)),
                        "variant": variant_key,
                    }
                )
                stats[task_name] += 1
                stats[dataset_name] += 1
                stats[f"{dataset_name}::{task_name}"] += 1

    return data_rows, meta_rows, stats



def initialize_output_files(output_dir: Path, output_format: str, validation_ratio: float) -> None:
    stems = ["train", "train_meta"]
    if validation_ratio > 0:
        stems.extend(["validation", "validation_meta"])

    suffixes: List[str] = []
    if output_format in {"csv", "both"}:
        suffixes.append(".csv")
    if output_format in {"jsonl", "both"}:
        suffixes.append(".jsonl")

    for stem in stems:
        for suffix in suffixes:
            (output_dir / f"{stem}{suffix}").write_text("", encoding="utf-8")



def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    anchor_ratios = parse_ratio_list(args.prefix_anchor_ratios)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_output_files(output_dir, args.output_format, args.validation_ratio)

    global_stats = Counter()
    n_train_total = 0
    n_train_meta_total = 0
    n_val_total = 0
    n_val_meta_total = 0

    dataset_names = [normalize_dataset_name(name) for name in args.dataset_names]
    dataset_names = list(dict.fromkeys(dataset_names))
    rng.shuffle(dataset_names)

    for dataset_name in dataset_names:
        csv_path = find_dataset_file(args.input_dir, dataset_name)
        df, case_id_col, activity_col, timestamp_col = load_dataset_csv(
            csv_path=csv_path,
            case_id_candidates=DEFAULT_CASE_ID_CANDIDATES,
            activity_candidates=DEFAULT_ACTIVITY_CANDIDATES,
            timestamp_candidates=DEFAULT_TIMESTAMP_CANDIDATES,
        )

        label_col = resolve_column(df, LABEL_COL_CANDIDATES, required=True)
        remaining_time_col = resolve_column(df, REMAINING_TIME_CANDIDATES, required=False)

        dataset_activity_candidates = sorted(
            {str(x).strip() for x in df[activity_col].dropna().tolist() if str(x).strip()}
        )
        if not dataset_activity_candidates:
            raise ValueError(f"No activity candidates found in dataset: {dataset_name}")

        cases = build_case_records(
            df=df,
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            label_col=label_col,
        )

        selected_case_ids = maybe_cap_cases_variant_balanced(
            cases=cases,
            max_cases_per_dataset=args.max_cases_per_dataset,
            rng=rng,
        )
        train_case_ids, val_case_ids = split_case_ids_variant_stratified(
            case_ids=selected_case_ids,
            cases=cases,
            validation_ratio=args.validation_ratio,
            rng=rng,
        )

        train_prefix_plan = build_prefix_plan(
            case_ids=train_case_ids,
            cases=cases,
            max_prefixes_per_case=args.max_prefixes_per_case,
            anchor_ratios=anchor_ratios,
            rng=rng,
        )
        train_prefix_plan = maybe_cap_prefix_plan(
            prefix_plan=train_prefix_plan,
            max_prefix_instances_per_dataset=args.max_prefix_instances_per_dataset,
            rng=rng,
        )

        train_rows, train_meta, train_stats = sample_rows_for_split(
            split_name="train",
            dataset_name=dataset_name,
            prefix_plan=train_prefix_plan,
            cases=cases,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            remaining_time_col=remaining_time_col,
            dataset_activity_candidates=dataset_activity_candidates,
            include_dataset_intro=args.include_dataset_intro,
            quote_activities=args.quote_activities,
            include_system_prompt=args.include_system_prompt,
            max_prefix_story_events=args.max_prefix_story_events,
            rng=rng,
        )
        n_train_total += write_rows(output_dir / "train", train_rows, args.output_format, is_meta=False)
        n_train_meta_total += write_rows(output_dir / "train_meta", train_meta, args.output_format, is_meta=True)
        global_stats.update(train_stats)

        val_rows_count = 0
        val_meta_count = 0
        val_prefix_count = 0
        if val_case_ids:
            val_prefix_plan = build_prefix_plan(
                case_ids=val_case_ids,
                cases=cases,
                max_prefixes_per_case=args.max_prefixes_per_case,
                anchor_ratios=anchor_ratios,
                rng=rng,
            )
            val_prefix_plan = maybe_cap_prefix_plan(
                prefix_plan=val_prefix_plan,
                max_prefix_instances_per_dataset=max(1, int(round(args.max_prefix_instances_per_dataset * args.validation_ratio)))
                if args.max_prefix_instances_per_dataset > 0
                else -1,
                rng=rng,
            )

            val_rows, val_meta, val_stats = sample_rows_for_split(
                split_name="validation",
                dataset_name=dataset_name,
                prefix_plan=val_prefix_plan,
                cases=cases,
                activity_col=activity_col,
                timestamp_col=timestamp_col,
                remaining_time_col=remaining_time_col,
                dataset_activity_candidates=dataset_activity_candidates,
                include_dataset_intro=args.include_dataset_intro,
                quote_activities=args.quote_activities,
                include_system_prompt=args.include_system_prompt,
                max_prefix_story_events=args.max_prefix_story_events,
                rng=rng,
            )
            val_rows_count = write_rows(output_dir / "validation", val_rows, args.output_format, is_meta=False)
            val_meta_count = write_rows(output_dir / "validation_meta", val_meta, args.output_format, is_meta=True)
            n_val_total += val_rows_count
            n_val_meta_total += val_meta_count
            global_stats.update(val_stats)
            val_prefix_count = count_prefix_instances(val_prefix_plan)

        train_prefix_count = count_prefix_instances(train_prefix_plan)
        print(
            f"[{dataset_name}] cases_kept={len(selected_case_ids)} train_cases={len(train_case_ids)} "
            f"val_cases={len(val_case_ids)} train_prefixes={train_prefix_count} "
            f"train_samples={len(train_rows)} val_prefixes={val_prefix_count} val_samples={val_rows_count}"
        )

    if args.output_format in {"csv", "both"}:
        print(f"Saved train data to {output_dir / 'train.csv'} ({n_train_total} rows)")
        print(f"Saved train metadata to {output_dir / 'train_meta.csv'} ({n_train_meta_total} rows)")
        if args.validation_ratio > 0:
            print(f"Saved validation data to {output_dir / 'validation.csv'} ({n_val_total} rows)")
            print(f"Saved validation metadata to {output_dir / 'validation_meta.csv'} ({n_val_meta_total} rows)")
    if args.output_format in {"jsonl", "both"}:
        print(f"Saved train data to {output_dir / 'train.jsonl'} ({n_train_total} rows)")
        print(f"Saved train metadata to {output_dir / 'train_meta.jsonl'} ({n_train_meta_total} rows)")
        if args.validation_ratio > 0:
            print(f"Saved validation data to {output_dir / 'validation.jsonl'} ({n_val_total} rows)")
            print(f"Saved validation metadata to {output_dir / 'validation_meta.jsonl'} ({n_val_meta_total} rows)")

    print("\nGeneration summary:")
    for key in sorted(global_stats.keys()):
        if "::" not in key and key not in {
            "next_activity_prediction",
            "remaining_time_prediction",
            "final_outcome_prediction",
        }:
            print(f"  {key}: {global_stats[key]}")
    print("Task counts:")
    for key in [
        "next_activity_prediction",
        "remaining_time_prediction",
        "final_outcome_prediction",
    ]:
        print(f"  {key}: {global_stats[key]}")


if __name__ == "__main__":
    main()
