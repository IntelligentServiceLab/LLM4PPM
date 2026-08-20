from __future__ import annotations

import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


TRAIN_DATASETS = [
    "BPIC2013I",
    "BPIC2013O",
    "BPIC2019",
    "BPIC2020D",
    "BPIC2020I",
    "BPIC2020Pe",
    "BPIC2020Pr",
    "BPIC2012A",
    "BPIC2012O",
    "Receipt",
    "Sepsis",
    "Road_Traffic_Fine",
    "Service_process",
]

TEST_DATASETS = [
    "BPIC2013C",
    "BPIC2017O",
    "BPIC2020R",
    "BPIC2012W",
    "Hospital-billing",
    "Helpdesk",
]


DEFAULT_CASE_ID_CANDIDATES = [
    "case_id",
    "caseid",
    "case",
    "case:concept:name",
    "case_concept_name",
    "trace_id",
    "traceid",
    "id",
    "Case ID",
]

DEFAULT_ACTIVITY_CANDIDATES = [
    "activity",
    "concept:name",
    "concept_name",
    "event",
    "event_name",
    "task",
]

DEFAULT_TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time:timestamp",
    "time_timestamp",
    "complete_timestamp",
    "event_time",
    "datetime",
    "time",
]


FieldSpec = Dict[str, Any]
DatasetConfig = Dict[str, Any]
ColumnLike = Union[str, Sequence[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build balanced semantic-story pretraining data with interval + elapsed time in each event sentence."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing per-dataset CSV files.")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV path. Only a single 'text' column will be saved.")
    parser.add_argument(
        "--dataset_names",
        type=str,
        default=",".join(TRAIN_DATASETS),
        help="Comma-separated dataset names. Default: all train datasets except Production.",
    )
    parser.add_argument(
        "--keep_meta",
        action="store_true",
        default=False,
        help="Deprecated and ignored. Output is always a text-only CSV for pretraining.",
    )
    parser.set_defaults(include_dataset_intro=True)
    parser.add_argument("--include_dataset_intro", dest="include_dataset_intro", action="store_true")
    parser.add_argument("--no_include_dataset_intro", dest="include_dataset_intro", action="store_false")
    parser.set_defaults(quote_activities=True)
    parser.add_argument("--quote_activities", dest="quote_activities", action="store_true")
    parser.add_argument("--no_quote_activities", dest="quote_activities", action="store_false")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--full_case_max_events",
        type=int,
        default=40,
        help="Keep the full case story only when trace length <= this value.",
    )
    parser.add_argument(
        "--slice_trigger_events",
        type=int,
        default=40,
        help="Start generating sliding windows when trace length > this value under the current segmentation logic.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="16,24,32",
        help="Comma-separated sliding window sizes in number of events.",
    )
    parser.add_argument(
        "--window_stride_ratio",
        type=float,
        default=0.5,
        help="Stride = max(1, floor(window_size * ratio)).",
    )
    parser.add_argument(
        "--min_window_events",
        type=int,
        default=8,
        help="Do not create windows shorter than this many events.",
    )
    parser.add_argument(
        "--max_windows_per_case",
        type=int,
        default=20,
        help="Maximum number of sliding-window stories kept per case. -1 means no limit.",
    )
    parser.add_argument(
        "--target_samples_per_dataset",
        type=int,
        default=8000,
        help="If > 0, downsample each dataset to at most this many stories after slicing. Small datasets keep all stories.",
    )
    parser.add_argument(
        "--balance_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "median", "min", "none"],
        help="How to determine per-dataset cap when target_samples_per_dataset <= 0.",
    )
    parser.add_argument(
        "--max_samples_per_variant",
        type=int,
        default=64,
        help="Maximum number of stories kept for the same activity-sequence variant inside one dataset. -1 means no limit.",
    )
    parser.set_defaults(deduplicate_exact_text=True)
    parser.add_argument(
        "--deduplicate_exact_text",
        dest="deduplicate_exact_text",
        action="store_true",
        help="Remove exactly identical story texts before variant-aware sampling.",
    )
    parser.add_argument(
        "--no_deduplicate_exact_text",
        dest="deduplicate_exact_text",
        action="store_false",
        help="Keep exact duplicate story texts.",
    )
    return parser.parse_args()


def normalize_dataset_name(name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]", "", name).upper()
    aliases = {
        "BPI_CHALLENGE_2013_I": "BPIC2013I",
        "BPICHALLENGE2013I": "BPIC2013I",
        "BPIC2013O": "BPIC2013O",
        "BPICHALLENGE2013O": "BPIC2013O",
        "BPIC2013C": "BPIC2013C",
        "BPICHALLENGE2013C": "BPIC2013C",
        "BPICHALLENGE2019": "BPIC2019",
        "BPICHALLENGE2020D": "BPIC2020D",
        "BPICHALLENGE2020I": "BPIC2020I",
        "BPICHALLENGE2020PE": "BPIC2020Pe",
        "BPICHALLENGE2020PR": "BPIC2020Pr",
        "BPICHALLENGE2020R": "BPIC2020R",
        "BPIC2012A": "BPIC2012A",
        "BPIC2012O": "BPIC2012O",
        "BPIC20120": "BPIC2012O",
        "BPIC2012W": "BPIC2012W",
        "RECEIPT": "Receipt",
        "SEPSIS": "Sepsis",
        "ROADTRAFFICFINE": "Road_Traffic_Fine",
        "SERVICEPROCESS": "Service_process",
        "HOSPITALBILLING": "Hospital-billing",
        "HELPDESK": "Helpdesk",
        "BPICHALLENGE2017O": "BPIC2017O",
        "PRODUCTION": "Production",
    }
    return aliases.get(key, name)


DEFAULT_DATASET_CONFIG: DatasetConfig = {
    "intro": "",
    "trace_fields": [],
    "event_fields": [],
}


DATASET_CONFIG: Dict[str, DatasetConfig] = {
    "BPIC2013I": {
        "intro": "This is an incident management case.",
        "trace_fields": [
            {"columns": ["impact"], "template": "the impact is {value}", "lower": True},
            {
                "columns": ["resource country", "organization country"],
                "template": "the handling country is {value}",
            },
        ],
        "event_fields": [
            {
                "columns": ["lifecycle:transition"],
                "template": "transition {value}",
                "lower": True,
            }
        ],
    },
    "BPIC2013O": {
        "intro": "This is a problem management case.",
        "trace_fields": [
            {"columns": ["impact"], "template": "the impact is {value}", "lower": True},
            {
                "columns": ["resource country", "organization country"],
                "template": "the handling country is {value}",
            },
        ],
        "event_fields": [
            {
                "columns": ["lifecycle:transition"],
                "template": "transition {value}",
                "lower": True,
            }
        ],
    },
    "BPIC2013C": {
        "intro": "This is a problem management case.",
        "trace_fields": [
            {"columns": ["impact"], "template": "the impact is {value}", "lower": True},
            {
                "columns": ["resource country", "organization country"],
                "template": "the handling country is {value}",
            },
        ],
        "event_fields": [
            {
                "columns": ["lifecycle:transition"],
                "template": "transition {value}",
                "lower": True,
            }
        ],
    },
    "BPIC2019": {
        "intro": "This is a purchase order handling case.",
        "trace_fields": [
            {"columns": ["case:Spend area text"], "template": "the spend area is {value}"},
            {"columns": ["case:Sub spend area text"], "template": "the sub spend area is {value}"},
            {
                "columns": ["Cumulative net worth (EUR)"],
                "template": "the net worth is {value} EUR",
            },
            {"columns": ["case:Item Type"], "template": "the item type is {value}", "lower": True},
            {
                "columns": ["case:Item Category"],
                "template": "the item category is {value}",
                "lower": True,
            },
            {
                "columns": ["case:GR-Based Inv. Verif."],
                "true": "GR-based invoice verification is enabled",
                "false": "GR-based invoice verification is disabled",
            },
        ],
        "event_fields": [],
    },
    "BPIC2020D": {
        "intro": "This is a domestic travel declaration case.",
        "trace_fields": [],
        "event_fields": [
            {"columns": ["org:role"], "template": "role {value}", "lower": True},
            {"columns": ["org:resource"], "template": "resource {value}", "lower": True},
        ],
    },
    "BPIC2020I": {
        "intro": "This is an international travel declaration case.",
        "trace_fields": [],
        "event_fields": [
            {"columns": ["org:role"], "template": "role {value}", "lower": True},
            {"columns": ["org:resource"], "template": "resource {value}", "lower": True},
        ],
    },
    "BPIC2020Pe": {
        "intro": "This is a travel permit case.",
        "trace_fields": [
            {
                "columns": ["case:Overspent"],
                "true": "the case is overspent",
                "false": "the case is not overspent",
            }
        ],
        "event_fields": [
            {"columns": ["org:role"], "template": "role {value}", "lower": True},
            {"columns": ["org:resource"], "template": "resource {value}", "lower": True},
        ],
    },
    "BPIC2020Pr": {
        "intro": "This is a payment request case.",
        "trace_fields": [],
        "event_fields": [
            {"columns": ["org:role"], "template": "role {value}", "lower": True},
            {"columns": ["org:resource"], "template": "resource {value}", "lower": True},
        ],
    },
    "BPIC2020R": {
        "intro": "This is a payment request case.",
        "trace_fields": [],
        "event_fields": [
            {"columns": ["org:role"], "template": "role {value}", "lower": True},
            {"columns": ["org:resource"], "template": "resource {value}", "lower": True},
        ],
    },
    "BPIC2012A": {
        "intro": "This is a loan application case.",
        "trace_fields": [],
        "event_fields": [
            {
                "columns": ["lifecycle:transition"],
                "template": "transition {value}",
                "lower": True,
            }
        ],
    },
    "BPIC2012O": {
        "intro": "This is an offer handling case.",
        "trace_fields": [],
        "event_fields": [
            {
                "columns": ["lifecycle:transition"],
                "template": "transition {value}",
                "lower": True,
            }
        ],
    },
    "BPIC2012W": {
        "intro": "This is a loan application case.",
        "trace_fields": [],
        "event_fields": [
            {
                "columns": ["lifecycle:transition"],
                "template": "transition {value}",
                "lower": True,
            }
        ],
    },
    "BPIC2017O": {
        "intro": "This is an offer handling case.",
        "trace_fields": [
            {"columns": ["MonthlyCost"], "template": "the monthly cost is {value}"},
            {"columns": ["NumberOfTerms"], "template": "the number of terms is {value}"},
            {"columns": ["CreditScore"], "template": "the credit score is {value}"},
        ],
        "event_fields": [
            {"columns": ["action"], "template": "action {value}", "lower": True}
        ],
    },
    "Receipt": {
        "intro": "This is a building permit receipt case.",
        "trace_fields": [
            {
                "columns": ["case:channel", "channel"],
                "template": "the submission channel is {value}",
                "lower": True,
            }
        ],
        "event_fields": [],
    },
    "Sepsis": {
        "intro": "This is a hospital sepsis case.",
        "trace_fields": [
            {"columns": ["Diagnose"], "template": "the diagnosis is {value}"},
            {"columns": ["Age"], "template": "the age is {value}"},
            {
                "columns": ["nfectionSuspected", "InfectionSuspected"],
                "true": "infection is suspected",
                "false": "infection is not suspected",
            },
            {
                "columns": ["Hypotensie"],
                "true": "hypotension is present",
                "false": "hypotension is not present",
            },
        ],
        "event_fields": [],
    },
    "Road_Traffic_Fine": {
        "intro": "This is a road traffic fine case.",
        "trace_fields": [],
        "event_fields": [],
    },
    "Service_process": {
        "intro": "This is a device repair service case.",
        "trace_fields": [],
        "event_fields": [],
    },
    "Hospital-billing": {
        "intro": "This is a hospital billing case.",
        "trace_fields": [],
        "event_fields": [],
    },
    "Helpdesk": {
        "intro": "This is a helpdesk ticket case.",
        "trace_fields": [
            {"columns": ["responsiblesection"], "template": "the responsible section is {value}", "lower": True},
            {"columns": ["supportsection"], "template": "the support section is {value}", "lower": True},
        ],
        "event_fields": [],
    },
}


def compact_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text


def normalize_value(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        low = s.lower()
        if low in {"true", "yes"}:
            return "yes"
        if low in {"false", "no"}:
            return "no"
        return s
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def join_phrases(phrases: Sequence[str]) -> str:
    items = [compact_text(p) for p in phrases if p and compact_text(p)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def resolve_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    df_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        real = df_cols.get(c.lower())
        if real is not None:
            return real
    if required:
        raise KeyError(f"Cannot find a column from candidates: {candidates}")
    return None


def as_column_list(columns: ColumnLike) -> List[str]:
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def get_first_non_null_from_case(case_df: pd.DataFrame, columns: ColumnLike) -> Optional[str]:
    for col in as_column_list(columns):
        if col not in case_df.columns:
            continue
        series = case_df[col]
        non_null = series[~series.isna()]
        if not non_null.empty:
            value = normalize_value(non_null.iloc[0])
            if value is not None:
                return value
    return None


def get_value_from_row(row: pd.Series, columns: ColumnLike) -> Optional[str]:
    for col in as_column_list(columns):
        if col not in row.index:
            continue
        value = normalize_value(row[col])
        if value is not None:
            return value
    return None


def interpret_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    low = str(value).strip().lower()
    if low in {"yes", "true", "1"}:
        return True
    if low in {"no", "false", "0"}:
        return False
    return None


def format_from_spec(value: Optional[str], spec: FieldSpec) -> Optional[str]:
    if value is None:
        return None

    bool_value = interpret_bool(value)
    if "true" in spec or "false" in spec:
        if bool_value is True:
            return spec.get("true")
        if bool_value is False:
            return spec.get("false")

    text = str(value)
    if spec.get("lower"):
        text = text.lower()
    template = spec.get("template")
    if template:
        return template.format(value=text)
    return text


def maybe_add_sentence(text: str) -> str:
    text = compact_text(text)
    if not text:
        return ""
    if text.endswith((".", "!", "?")):
        return text
    return text + "."


def floor_days(value: float) -> int:
    if pd.isna(value):
        return 0
    return int(math.floor(max(float(value), 0.0)))


def humanize_interval(interval_days: float, is_first: bool, segment_starts_at_case_start: bool) -> str:
    floored = floor_days(interval_days)

    if is_first and segment_starts_at_case_start:
        return ""
    if is_first and not segment_starts_at_case_start:
        if floored <= 0:
            return "Later the same day in the case"
        if floored == 1:
            return "1 day after the previous event"
        return f"{floored} days after the previous event"

    if floored <= 0:
        return "Later the same day"
    if floored == 1:
        return "1 day later"
    return f"{floored} days later"


def humanize_elapsed(elapsed_days: float) -> str:
    floored = floor_days(elapsed_days)
    return f"on day {floored}"


def build_event_core(activity: str, is_first: bool, segment_starts_at_case_start: bool) -> str:
    if is_first and segment_starts_at_case_start:
        return f"the case started with {activity}"
    if is_first:
        return f"the case resumed with {activity}"
    return f"it went through {activity}"


def format_activity(activity: Any, quote_activities: bool = True) -> str:
    activity_text = compact_text(str(activity))
    if quote_activities:
        return f'"{activity_text}"'
    return activity_text


def ensure_time_features(case_df: pd.DataFrame, timestamp_col: Optional[str]) -> pd.DataFrame:
    df = case_df.copy()
    if "elapsed_time" in df.columns and "interval_time" in df.columns:
        df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce").fillna(0).clip(lower=0)
        df["interval_time"] = pd.to_numeric(df["interval_time"], errors="coerce").fillna(0).clip(lower=0)
        return df.reset_index(drop=True)

    if timestamp_col is None:
        raise ValueError("Need a timestamp column or precomputed elapsed_time/interval_time columns.")
    df["__orig_order"] = range(len(df))
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col],
            format='mixed',
            errors="coerce",
            utc=True
        )
        df = df.sort_values(
            by=[timestamp_col, "__orig_order"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            by="__orig_order",
            kind="mergesort",
        ).reset_index(drop=True)
    start_ts = df[timestamp_col].iloc[0]
    elapsed_days = ((df[timestamp_col] - start_ts).dt.total_seconds() / 86400.0).fillna(0)
    interval_days = df[timestamp_col].diff().dt.total_seconds().div(86400.0).fillna(0)

    df["elapsed_time"] = elapsed_days.clip(lower=0)
    df["interval_time"] = interval_days.clip(lower=0)
    df = df.drop(columns=["__orig_order"])

    return df


def render_trace_sentence(case_df: pd.DataFrame, cfg: DatasetConfig, include_dataset_intro: bool = True) -> str:
    intro = maybe_add_sentence(cfg.get("intro", "")) if include_dataset_intro else ""
    trace_specs = cfg.get("trace_fields", []) or []

    phrases: List[str] = []
    for spec in trace_specs:
        value = get_first_non_null_from_case(case_df, spec["columns"])
        phrase = format_from_spec(value, spec)
        if phrase:
            phrases.append(phrase)

    if phrases:
        trace_sentence = maybe_add_sentence(f"At the case level, {join_phrases(phrases)}")
        return compact_text(" ".join(p for p in [intro, trace_sentence] if p))

    return intro


def render_event_sentence(
    row: pd.Series,
    event_specs: Sequence[FieldSpec],
    activity_col: str,
    is_first: bool,
    segment_starts_at_case_start: bool,
    quote_activities: bool = True,
) -> str:
    lead = humanize_interval(
        interval_days=float(row.get("interval_time", 0) or 0),
        is_first=is_first,
        segment_starts_at_case_start=segment_starts_at_case_start,
    )
    activity = format_activity(row[activity_col], quote_activities=quote_activities)
    event_core = build_event_core(
        activity=activity,
        is_first=is_first,
        segment_starts_at_case_start=segment_starts_at_case_start,
    )
    elapsed = humanize_elapsed(float(row.get("elapsed_time", 0) or 0))

    extras: List[str] = []
    for spec in event_specs:
        value = get_value_from_row(row, spec["columns"])
        phrase = format_from_spec(value, spec)
        if phrase:
            extras.append(phrase)

    if lead:
        sentence = f"{lead}, {event_core} {elapsed}"
    else:
        sentence = f"{event_core} {elapsed}"
    if extras:
        sentence += f", with {join_phrases(extras)}"

    sentence = compact_text(sentence)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return maybe_add_sentence(sentence)


def build_case_story(
    case_df: pd.DataFrame,
    dataset_name: str,
    activity_col: str,
    timestamp_col: Optional[str],
    include_dataset_intro: bool = False,
    quote_activities: bool = True,
    segment_start: int = 0,
    trace_context_df: Optional[pd.DataFrame] = None,
) -> str:
    cfg = DATASET_CONFIG.get(dataset_name, DEFAULT_DATASET_CONFIG)
    case_df = ensure_time_features(case_df, timestamp_col)
    trace_df = case_df if trace_context_df is None else trace_context_df.copy().reset_index(drop=True)

    parts: List[str] = []
    trace_sentence = render_trace_sentence(
        case_df=trace_df,
        cfg=cfg,
        include_dataset_intro=include_dataset_intro,
    )
    if trace_sentence:
        parts.append(trace_sentence)

    event_specs = cfg.get("event_fields", []) or []
    for idx, (_, row) in enumerate(case_df.iterrows()):
        parts.append(
            render_event_sentence(
                row=row,
                event_specs=event_specs,
                activity_col=activity_col,
                is_first=(idx == 0),
                segment_starts_at_case_start=(segment_start == 0 and idx == 0),
                quote_activities=quote_activities,
            )
        )

    return compact_text(" ".join(p for p in parts if p))


def find_dataset_file(input_dir: str | Path, dataset_name: str) -> Path:
    input_dir = Path(input_dir)
    candidates = list(input_dir.glob("*.csv"))
    for path in candidates:
        if normalize_dataset_name(path.stem) == dataset_name:
            return path
    raise FileNotFoundError(f"Cannot find CSV file for dataset: {dataset_name} in {input_dir}")


def load_dataset_csv(
    csv_path: str | Path,
    case_id_candidates: Sequence[str] = DEFAULT_CASE_ID_CANDIDATES,
    activity_candidates: Sequence[str] = DEFAULT_ACTIVITY_CANDIDATES,
    timestamp_candidates: Sequence[str] = DEFAULT_TIMESTAMP_CANDIDATES,
) -> tuple[pd.DataFrame, str, str, Optional[str]]:
    df = pd.read_csv(csv_path)
    case_id_col = resolve_column(df, case_id_candidates, required=True)
    activity_col = resolve_column(df, activity_candidates, required=True)
    timestamp_col = resolve_column(df, timestamp_candidates, required=False)
    return df, case_id_col, activity_col, timestamp_col


def make_variant_key(case_df: pd.DataFrame, activity_col: str) -> str:
    acts = [compact_text(str(x)).lower() for x in case_df[activity_col].tolist()]
    return " || ".join(acts)


def deduplicate_segments(segments: Iterable[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    seen = set()
    out: List[Tuple[int, int, str]] = []
    for start, end, seg_type in segments:
        key = (int(start), int(end), seg_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def maybe_limit_windows(
    segments: List[Tuple[int, int, str]],
    max_windows_per_case: int,
) -> List[Tuple[int, int, str]]:
    if max_windows_per_case < 0:
        return segments

    full_segments = [seg for seg in segments if seg[2] == "full"]
    window_segments = [seg for seg in segments if seg[2] != "full"]
    if len(window_segments) <= max_windows_per_case:
        return segments

    keep_idx = np.linspace(0, len(window_segments) - 1, num=max_windows_per_case, dtype=int)
    kept_windows = [window_segments[int(i)] for i in keep_idx]
    return full_segments + kept_windows


def build_segments_for_case(
    case_df: pd.DataFrame,
    full_case_max_events: int,
    slice_trigger_events: int,
    window_sizes: Sequence[int],
    window_stride_ratio: float,
    min_window_events: int,
    max_windows_per_case: int,
) -> List[Tuple[int, int, str]]:
    n_events = len(case_df)
    segments: List[Tuple[int, int, str]] = []
    if n_events <= full_case_max_events:
        segments.append((0, n_events, "full"))
        return segments

    if n_events <= max(full_case_max_events, slice_trigger_events):
        segments.append((0, n_events, "full"))
        return segments

    valid_window_sizes = sorted({w for w in window_sizes if min_window_events <= w < n_events})

    if n_events <= 64:
        segments.append((0, n_events, "full"))

    if n_events >= slice_trigger_events and valid_window_sizes:
        for window_size in sorted(valid_window_sizes, reverse=True):
            step = max(1, int(math.floor(window_size * window_stride_ratio)))
            starts = list(range(0, n_events - window_size + 1, step))
            if starts and starts[-1] != n_events - window_size:
                starts.append(n_events - window_size)
            for start in starts:
                end = start + window_size
                segments.append((start, end, f"window_{window_size}"))

    if not segments:
        segments.append((0, n_events, "full"))

    segments = deduplicate_segments(segments)
    segments = maybe_limit_windows(segments, max_windows_per_case=max_windows_per_case)
    return segments


def build_candidate_rows_for_dataset(
    input_dir: str | Path,
    dataset_name: str,
    include_dataset_intro: bool,
    quote_activities: bool,
    full_case_max_events: int,
    slice_trigger_events: int,
    window_sizes: Sequence[int],
    window_stride_ratio: float,
    min_window_events: int,
    max_windows_per_case: int,
) -> List[Dict[str, Any]]:
    csv_path = find_dataset_file(input_dir, dataset_name)
    df, case_id_col, activity_col, timestamp_col = load_dataset_csv(
        csv_path,
        case_id_candidates=DEFAULT_CASE_ID_CANDIDATES,
        activity_candidates=DEFAULT_ACTIVITY_CANDIDATES,
        timestamp_candidates=DEFAULT_TIMESTAMP_CANDIDATES,
    )

    rows: List[Dict[str, Any]] = []
    for case_id, case_df in df.groupby(case_id_col, sort=False):
        case_df = case_df.copy().reset_index(drop=True)
        case_df = ensure_time_features(case_df, timestamp_col)
        segments = build_segments_for_case(
            case_df=case_df,
            full_case_max_events=full_case_max_events,
            slice_trigger_events=slice_trigger_events,
            window_sizes=window_sizes,
            window_stride_ratio=window_stride_ratio,
            min_window_events=min_window_events,
            max_windows_per_case=max_windows_per_case,
        )

        for slice_id, (start, end, slice_type) in enumerate(segments):
            seg_df = case_df.iloc[start:end].reset_index(drop=True)
            if seg_df.empty:
                continue
            story = build_case_story(
                case_df=seg_df,
                dataset_name=dataset_name,
                activity_col=activity_col,
                timestamp_col=timestamp_col,
                include_dataset_intro=include_dataset_intro,
                quote_activities=quote_activities,
                segment_start=int(start),
                trace_context_df=case_df,
            )
            if not story:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "slice_id": slice_id,
                    "slice_type": slice_type,
                    "slice_start": int(start),
                    "slice_end": int(end),
                    "n_events": int(len(seg_df)),
                    "variant_key": make_variant_key(seg_df, activity_col),
                    "text": story,
                }
            )
    return rows


def deduplicate_exact_text(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for rec in records:
        text = rec["text"]
        if text in seen:
            continue
        seen.add(text)
        out.append(rec)
    return out


def get_target_cap(
    dataset_to_count: Dict[str, int],
    target_samples_per_dataset: int,
    balance_strategy: str,
) -> Optional[int]:
    if not dataset_to_count:
        return None
    if target_samples_per_dataset and target_samples_per_dataset > 0:
        return int(target_samples_per_dataset)
    values = list(dataset_to_count.values())
    if balance_strategy == "none":
        return None
    if balance_strategy == "min":
        return int(min(values))
    if balance_strategy == "median":
        return int(np.median(values))
    return None


def sort_variant_group(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(x: Dict[str, Any]) -> Tuple[int, int, int, int]:
        is_full = 1 if x.get("slice_type") == "full" else 0
        n_events = int(x.get("n_events", 0))
        slice_start = int(x.get("slice_start", 0))
        slice_end = int(x.get("slice_end", 0))
        return (is_full, n_events, -slice_start, slice_end)

    return sorted(items, key=key_fn, reverse=True)


def sample_variant_balanced(
    records: List[Dict[str, Any]],
    target_n: Optional[int],
    max_samples_per_variant: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if target_n is None or target_n <= 0 or len(records) <= target_n:
        if max_samples_per_variant > 0:
            grouped = defaultdict(list)
            for rec in records:
                grouped[rec["variant_key"]].append(rec)
            out = []
            for key in sorted(grouped.keys()):
                items = sort_variant_group(grouped[key])[:max_samples_per_variant]
                out.extend(items)
            return out
        return records

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["variant_key"]].append(rec)

    for key, items in grouped.items():
        rng.shuffle(items)
        grouped[key] = sort_variant_group(items)

    variant_keys = list(grouped.keys())
    rng.shuffle(variant_keys)

    picked_per_variant = defaultdict(int)
    sampled: List[Dict[str, Any]] = []

    while len(sampled) < target_n:
        added = False
        rng.shuffle(variant_keys)
        for key in variant_keys:
            idx = picked_per_variant[key]
            if max_samples_per_variant > 0 and idx >= max_samples_per_variant:
                continue
            if idx >= len(grouped[key]):
                continue
            sampled.append(grouped[key][idx])
            picked_per_variant[key] += 1
            added = True
            if len(sampled) >= target_n:
                break
        if not added:
            break
    return sampled


def save_records(records: List[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != ".csv":
        raise ValueError("For pretraining, please use a .csv output path. This script now saves a text-only CSV.")

    pd.DataFrame({"text": [rec["text"] for rec in records]}).to_csv(output_path, index=False)



def build_pretrain_dataset(
    input_dir: str | Path,
    output_path: str | Path,
    dataset_names: Optional[Sequence[str]] = None,
    include_dataset_intro: bool = True,
    quote_activities: bool = True,
    random_seed: int = 3407,
    full_case_max_events: int = 40,
    slice_trigger_events: int = 40,
    window_sizes: Sequence[int] = (16, 24, 32),
    window_stride_ratio: float = 0.5,
    min_window_events: int = 8,
    max_windows_per_case: int = 20,
    target_samples_per_dataset: int = 8000,
    balance_strategy: str = "fixed",
    max_samples_per_variant: int = 64,
    dedup_text: bool = True,
) -> pd.DataFrame:
    dataset_names = list(dataset_names or TRAIN_DATASETS)

    all_rows: List[Dict[str, Any]] = []
    per_dataset_candidates: Dict[str, List[Dict[str, Any]]] = {}

    for i, dataset_name in enumerate(dataset_names):
        rows = build_candidate_rows_for_dataset(
            input_dir=input_dir,
            dataset_name=dataset_name,
            include_dataset_intro=include_dataset_intro,
            quote_activities=quote_activities,
            full_case_max_events=full_case_max_events,
            slice_trigger_events=slice_trigger_events,
            window_sizes=window_sizes,
            window_stride_ratio=window_stride_ratio,
            min_window_events=min_window_events,
            max_windows_per_case=max_windows_per_case,
        )
        if dedup_text:
            rows = deduplicate_exact_text(rows)
        per_dataset_candidates[dataset_name] = rows
        print(f"[{dataset_name}] candidates after slicing{' + dedup' if dedup_text else ''}: {len(rows)}")

    dataset_to_count = {k: len(v) for k, v in per_dataset_candidates.items()}
    target_cap = get_target_cap(
        dataset_to_count=dataset_to_count,
        target_samples_per_dataset=target_samples_per_dataset,
        balance_strategy=balance_strategy,
    )
    print(f"Per-dataset target cap: {target_cap if target_cap is not None else 'None'}")

    for i, dataset_name in enumerate(dataset_names):
        rows = per_dataset_candidates[dataset_name]
        sampled = sample_variant_balanced(
            rows,
            target_n=min(len(rows), target_cap) if target_cap is not None else None,
            max_samples_per_variant=max_samples_per_variant,
            seed=random_seed + i,
        )
        print(f"[{dataset_name}] kept after variant-balanced sampling: {len(sampled)}")
        all_rows.extend(sampled)

    final_df = pd.DataFrame({"text": [rec["text"] for rec in all_rows]})
    if final_df.empty:
        raise ValueError("No stories were generated. Please check dataset paths and column names.")

    save_records(all_rows, output_path=output_path)
    return final_df



def parse_window_sizes(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("window_sizes must contain at least one integer.")
    return values



def parse_dataset_names(text: str) -> List[str]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(normalize_dataset_name(part))
    return out



def main() -> None:
    args = parse_args()
    if args.keep_meta:
        print("[WARN] --keep_meta is deprecated and ignored. Output will still be a text-only CSV.")

    dataset_names = parse_dataset_names(args.dataset_names)
    window_sizes = parse_window_sizes(args.window_sizes)

    build_pretrain_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        dataset_names=dataset_names,
        include_dataset_intro=args.include_dataset_intro,
        quote_activities=args.quote_activities,
        random_seed=args.random_seed,
        full_case_max_events=args.full_case_max_events,
        slice_trigger_events=args.slice_trigger_events,
        window_sizes=window_sizes,
        window_stride_ratio=args.window_stride_ratio,
        min_window_events=args.min_window_events,
        max_windows_per_case=args.max_windows_per_case,
        target_samples_per_dataset=args.target_samples_per_dataset,
        balance_strategy=args.balance_strategy,
        max_samples_per_variant=args.max_samples_per_variant,
        dedup_text=args.deduplicate_exact_text,
    )


if __name__ == "__main__":
    main()
