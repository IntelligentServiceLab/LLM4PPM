#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xes", ".gz"}
DEFAULT_POSITIVE_LABEL = "success"
DEFAULT_NEGATIVE_LABEL = "failure"


BASE_RULE_DESCRIPTIONS: Dict[str, str] = {
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
    "BPIC2012W": "the same case is accepted in BPIC2012A",
    "Hospital-billing": "the case is finalized or issued and never reopened",
    "Receipt": "the case never enters hold, stop, or unlicensed branches",
    "Sepsis": "the patient does not return to emergency within 28 days after release",
    "Helpdesk": "the ticket is closed and never marked duplicate or invalid",
    "Road_Traffic_Fine": "the fine is paid and never sent for debt collection",
    "Service-process": "REPAIR_IN_TIME_5D is true",
}


DATASET_ALIASES: Dict[str, Sequence[str]] = {
    "BPIC2013I": ["bpic2013i", "bpi_challenge_2013_i", "bpi2013i", "incident"],
    "BPIC2013O": ["bpic2013o", "bpi_challenge_2013_o", "bpi2013o", "open problems", "open_problem"],
    "BPIC2013C": ["bpic2013c", "bpi_challenge_2013_c", "bpi2013c", "closed problems", "closed_problem"],
    "BPIC2017O": ["bpic2017o", "bpi challenge 2017 o", "bpi_challenge_2017_o", "2017o", "offer graph", "ocel"],
    "BPIC2019": ["bpic2019", "bpi challenge 2019", "bpi_challenge_2019", "purchase order"],
    "BPIC2020D": ["bpic2020d", "bpi_challenge_2020_d", "2020d", "domestic declarations", "domestic_declarations"],
    "BPIC2020I": ["bpic2020i", "bpi_challenge_2020_i", "2020i", "international declarations", "international_declarations"],
    "BPIC2020Pe": ["bpic2020pe", "bpi_challenge_2020_pe", "2020pe", "travel permits", "travel_permits"],
    "BPIC2020Pr": ["bpic2020pr", "bpi_challenge_2020_pr", "2020pr", "prepaid travel", "prepaid_travel"],
    "BPIC2020R": ["bpic2020r", "bpi_challenge_2020_r", "2020r", "request for payment", "request_for_payment"],
    "BPIC2012A": ["bpic2012a", "bpi_challenge_2012_a", "bpic_2012_a"],
    "BPIC2012O": ["bpic2012-o", "bpic2012o", "bpi_challenge_2012_o", "2012-o", "bpic_2012_o"],
    "BPIC2012W": ["bpic2012-w", "bpic2012w", "bpi_challenge_2012_w", "2012-w", "bpic_2012_w"],
    "Hospital-billing": ["hospital-billing", "hospital_billing", "hospital billing"],
    "Receipt": ["receipt", "receiving phase"],
    "Sepsis": ["sepsis"],
    "Helpdesk": ["helpdesk", "help desk"],
    "Production": ["production"],
    "Road_Traffic_Fine": ["road_traffic_fine", "road traffic fine", "traffic fines", "traffic_fines"],
    "Service-process": ["service-process", "service_process", "service process"],
}


CASE_COL_CANDIDATES = [
    "case:concept:name",
    "case_concept_name",
    "Case ID",
    "CaseID",
    "CASE_ID",
    "caseid",
    "case_id",
    "trace_id",
    "traceid",
    "case",
]

ACTIVITY_COL_CANDIDATES = [
    "concept:name",
    "Activity",
    "activity",
    "task",
    "Task",
    "event",
    "Event",
    "act_name",
]

TIMESTAMP_COL_CANDIDATES = [
    "time:timestamp",
    "Complete Timestamp",
    "complete_timestamp",
    "timestamp",
    "Timestamp",
    "event_time",
    "datetime",
    "date",
    "end_timestamp",
]


RECEIPT_NEGATIVE_ACTS = {
    "t07-3 draft intern advice hold for aspect 3",
    "t07-4 draft internal advice to hold for type 4",
    "t08 draft and send request for advice",
    "t10 determine necessity to stop indication",
    "t11 create document x request unlicensed",
    "t12 check document x request unlicensed",
    "t13 adjust document x request unlicensed",
    "t14 determine document x request unlicensed",
    "t15 print document x request unlicensed",
    "t16 report reasons to hold request",
    "t17 check report y to stop indication",
    "t18 adjust report y to stop indicition",
    "t19 determine report y to stop indication",
    "t20 print report y to stop indication",
    # canonicalized variants from mapped_dataset
    "draft internal advice aspect 3",
    "draft internal advice aspect 4",
    "send advice request",
    "determine stop indication need",
    "create unlicensed request document",
    "check unlicensed request document",
    "adjust unlicensed request document",
    "determine unlicensed request document",
    "print unlicensed request document",
    "report hold reasons",
    "check stop indication report",
    "adjust stop indication report",
    "determine stop indication report",
    "print stop indication report",
}

HELPDESK_BAD = {"duplicate", "mark duplicate ticket", "invalid", "mark invalid ticket"}
HELPDESK_CLOSED = {"closed", "close ticket"}
PRODUCTION_NEGATIVE_ACT_KEYWORDS = ["rework", "reject"]
ROAD_CREDIT_COLLECTION = {"send for credit collection", "send for debt collection"}
ROAD_PAYMENT = {"payment", "receive payment"}


def build_rule_descriptions(positive_label: str, negative_label: str) -> Dict[str, str]:
    return {
        dataset: f"label='{positive_label}' if {condition}; otherwise label='{negative_label}'."
        for dataset, condition in BASE_RULE_DESCRIPTIONS.items()
    }


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    text = str(x).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_case_id(x: object) -> str:
    return str(x).strip()


def resolve_dataset_name(path_or_name: str) -> Optional[str]:
    key = str(path_or_name).lower()
    for dataset, aliases in DATASET_ALIASES.items():
        for alias in aliases:
            if alias in key:
                return dataset
    return None


def supported_file(path: Path) -> bool:
    if path.suffix.lower() in {".csv", ".parquet", ".xes"}:
        return True
    if path.suffix.lower() == ".gz" and path.name.lower().endswith(".xes.gz"):
        return True
    return False


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".xes" or path.name.lower().endswith(".xes.gz"):
        try:
            import pm4py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Reading XES files requires pm4py. Please run: pip install pm4py") from exc
        df = pm4py.read_xes(str(path))
        if not isinstance(df, pd.DataFrame):
            df = pm4py.convert_to_dataframe(df)
        return df
    raise ValueError(f"Unsupported file format: {path}")


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    exact = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]
    for cand in candidates:
        cl = cand.lower()
        for col in columns:
            if cl in col.lower():
                return col
    return None


def find_numeric_column(columns: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    for col in columns:
        cl = col.lower()
        if all(k.lower() in cl for k in keywords):
            return col
    for col in columns:
        cl = col.lower()
        if any(k.lower() in cl for k in keywords):
            return col
    return None


def find_booleanish_column(columns: Sequence[str], names: Sequence[str]) -> Optional[str]:
    exact = {c.lower(): c for c in columns}
    for name in names:
        if name.lower() in exact:
            return exact[name.lower()]
    for name in names:
        nl = name.lower()
        for col in columns:
            if nl in col.lower():
                return col
    return None


def to_bool(value: object) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        return bool(int(value))
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "closed", "complete", "completed"}:
        return True
    if s in {"0", "false", "f", "no", "n", "open", "incomplete"}:
        return False
    return None


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def sort_group(group: pd.DataFrame, ts_col: Optional[str]) -> pd.DataFrame:
    if ts_col is None or ts_col not in group.columns:
        return group
    temp = group.copy()
    temp["__tmp_ts__"] = ensure_datetime(temp[ts_col])
    if temp["__tmp_ts__"].notna().any():
        temp = temp.sort_values(["__tmp_ts__"], kind="mergesort")
    return temp.drop(columns=["__tmp_ts__"])


def activity_series(group: pd.DataFrame, activity_col: str) -> pd.Series:
    return group[activity_col].astype(str).map(norm_text)


def case_has_any(group: pd.DataFrame, activity_col: str, acts: Iterable[str]) -> bool:
    target = {norm_text(a) for a in acts}
    values = set(activity_series(group, activity_col).tolist())
    return len(values & target) > 0


def last_activity(group: pd.DataFrame, activity_col: str, ts_col: Optional[str]) -> str:
    ordered = sort_group(group, ts_col)
    return norm_text(ordered.iloc[-1][activity_col]) if len(ordered) > 0 else ""


def build_bpic2012a_accept_map(files: Sequence[Tuple[Path, str]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for path, dataset in files:
        if dataset != "BPIC2012A":
            continue
        df = read_table(path)
        case_col = find_column(df.columns, CASE_COL_CANDIDATES)
        act_col = find_column(df.columns, ACTIVITY_COL_CANDIDATES)
        if case_col is None or act_col is None:
            continue
        for case_id, group in df.groupby(case_col, sort=False):
            label = 1 if case_has_any(group, act_col, ["accept application", "accepted"]) else 0
            result[normalize_case_id(case_id)] = label
    return result


def label_bpic2013(group: pd.DataFrame, activity_col: str, ts_col: Optional[str]) -> int:
    last_act = last_activity(group, activity_col, ts_col)
    return int(last_act in {"complete incident", "complete problem", "complete case", "completed"})


def label_bpic2017o(group: pd.DataFrame, activity_col: str) -> int:
    return int(case_has_any(group, activity_col, ["accept offer", "o_accepted"]))


def label_bpic2019(group: pd.DataFrame, activity_col: str) -> int:
    return int(case_has_any(group, activity_col, ["clear invoice"]))


def label_payment_handled(group: pd.DataFrame, activity_col: str) -> int:
    return int(case_has_any(group, activity_col, ["handle payment", "payment handled"]))


def label_bpic2012_accept(group: pd.DataFrame, activity_col: str) -> int:
    return int(case_has_any(group, activity_col, ["accept application", "accept offer", "accepted"]))


def label_bpic2012w(case_id: object, accept_map: Dict[str, int]) -> Optional[int]:
    return accept_map.get(normalize_case_id(case_id))


def label_hospital_billing(group: pd.DataFrame, activity_col: str, ts_col: Optional[str]) -> int:
    has_reopen = case_has_any(group, activity_col, ["reopen bill", "reopen"])
    is_closed_col = find_booleanish_column(group.columns, ["isClosed", "is_closed", "closed"])
    closed_flag = False
    if is_closed_col is not None:
        vals = group[is_closed_col].map(to_bool)
        closed_flag = any(v is True for v in vals.tolist())
    terminal = last_activity(group, activity_col, ts_col)
    terminal_positive = {
        "finalize bill",
        "issue bill",
        "release bill",
        "delete billing record",
        "set bill status",
        "fin",
        "billed",
        "release",
        "delete",
        "set status",
    }
    has_terminal = terminal in terminal_positive or case_has_any(
        group,
        activity_col,
        ["finalize bill", "issue bill", "release bill", "delete billing record", "set bill status"],
    )
    return int((closed_flag or has_terminal) and (not has_reopen))


def label_receipt(group: pd.DataFrame, activity_col: str) -> int:
    acts = set(activity_series(group, activity_col).tolist())
    return int(len(acts & RECEIPT_NEGATIVE_ACTS) == 0)


def label_sepsis(group: pd.DataFrame, activity_col: str, ts_col: Optional[str]) -> int:
    ordered = sort_group(group, ts_col)
    acts = ordered[activity_col].astype(str).map(norm_text).tolist()
    if "return er" not in acts and "return to emergency" not in acts:
        return 1

    if ts_col is not None and ts_col in ordered.columns:
        ts = ensure_datetime(ordered[ts_col])
        release_set = {
            "release a",
            "release b",
            "release c",
            "release d",
            "release e",
            "release patient a",
            "release patient b",
            "release patient c",
            "release patient d",
            "release patient e",
        }
        return_set = {"return er", "return to emergency"}
        last_release_time = None
        for act, t in zip(acts, ts.tolist()):
            if pd.isna(t):
                continue
            if act in release_set:
                last_release_time = t
            elif act in return_set:
                if last_release_time is not None:
                    delta_days = (t - last_release_time).total_seconds() / 86400.0
                    return int(delta_days > 28.0)
                return 0
    return 0


def label_helpdesk(group: pd.DataFrame, activity_col: str) -> int:
    acts = set(activity_series(group, activity_col).tolist())
    has_closed = len(acts & HELPDESK_CLOSED) > 0
    has_bad = len(acts & HELPDESK_BAD) > 0
    return int(has_closed and not has_bad)


def label_production(group: pd.DataFrame, activity_col: str) -> int:
    qty_col = None
    for cand in ["Qty_Rejected", "qty_rejected", "QTY_REJECTED", "Qty Rejected", "rejected qty"]:
        qty_col = find_booleanish_column(group.columns, [cand])
        if qty_col is not None:
            break
    if qty_col is not None:
        vals = pd.to_numeric(group[qty_col], errors="coerce")
        if (vals.fillna(0) > 0).any():
            return 0
        return 1
    acts = activity_series(group, activity_col)
    if any(any(k in a for k in PRODUCTION_NEGATIVE_ACT_KEYWORDS) for a in acts.tolist()):
        return 0
    return 1


def label_road_traffic_fine(group: pd.DataFrame, activity_col: str) -> int:
    acts = set(activity_series(group, activity_col).tolist())
    if len(acts & ROAD_CREDIT_COLLECTION) > 0:
        return 0
    amount_col = find_numeric_column(group.columns, ["amount"])
    total_payment_col = None
    for col in group.columns:
        cl = col.lower()
        if "totalpaymentamount" in cl or ("total" in cl and "payment" in cl and "amount" in cl):
            total_payment_col = col
            break
    if amount_col is not None and total_payment_col is not None:
        amount = pd.to_numeric(group[amount_col], errors="coerce").dropna()
        paid = pd.to_numeric(group[total_payment_col], errors="coerce").dropna()
        if len(amount) > 0 and len(paid) > 0:
            return int(float(paid.max()) >= float(amount.max()))
    return int(len(acts & ROAD_PAYMENT) > 0)


def label_service_process(group: pd.DataFrame, activity_col: str, ts_col: Optional[str]) -> int:
    bool_col = find_booleanish_column(group.columns, ["REPAIR_IN_TIME_5D", "repair_in_time_5d"])
    if bool_col is not None:
        vals = group[bool_col].map(to_bool)
        for v in vals.tolist():
            if v is not None:
                return int(v)
    acts = set(activity_series(group, activity_col).tolist())
    has_completed = "complete service order" in acts or "completed" in acts
    if has_completed and ts_col is not None and ts_col in group.columns:
        ts = ensure_datetime(group[ts_col]).dropna()
        if len(ts) >= 2:
            duration_days = (ts.max() - ts.min()).total_seconds() / 86400.0
            return int(duration_days <= 5.0)
    return int(has_completed)


def compute_case_labels(
    df: pd.DataFrame,
    dataset: str,
    case_col: str,
    activity_col: str,
    ts_col: Optional[str],
    bpic2012a_accept_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Optional[int]]:
    labels: Dict[str, Optional[int]] = {}
    for case_id, group in df.groupby(case_col, sort=False):
        case_key = normalize_case_id(case_id)
        if dataset in {"BPIC2013I", "BPIC2013O", "BPIC2013C"}:
            label = label_bpic2013(group, activity_col, ts_col)
        elif dataset == "BPIC2017O":
            label = label_bpic2017o(group, activity_col)
        elif dataset == "BPIC2019":
            label = label_bpic2019(group, activity_col)
        elif dataset in {"BPIC2020D", "BPIC2020I", "BPIC2020Pe", "BPIC2020Pr", "BPIC2020R"}:
            label = label_payment_handled(group, activity_col)
        elif dataset in {"BPIC2012A", "BPIC2012O"}:
            label = label_bpic2012_accept(group, activity_col)
        elif dataset == "BPIC2012W":
            label = label_bpic2012w(case_id, bpic2012a_accept_map or {})
        elif dataset == "Hospital-billing":
            label = label_hospital_billing(group, activity_col, ts_col)
        elif dataset == "Receipt":
            label = label_receipt(group, activity_col)
        elif dataset == "Sepsis":
            label = label_sepsis(group, activity_col, ts_col)
        elif dataset == "Helpdesk":
            label = label_helpdesk(group, activity_col)
        elif dataset == "Production":
            label = label_production(group, activity_col)
        elif dataset == "Road_Traffic_Fine":
            label = label_road_traffic_fine(group, activity_col)
        elif dataset == "Service-process":
            label = label_service_process(group, activity_col, ts_col)
        else:
            raise KeyError(f"No outcome rule defined for dataset: {dataset}")
        labels[case_key] = label
    return labels


def numeric_label_to_word(value: Optional[int], positive_label: str, negative_label: str) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    return positive_label if int(value) == 1 else negative_label


def label_dataframe(
    df: pd.DataFrame,
    dataset: str,
    positive_label: str,
    negative_label: str,
    label_col: str,
    bpic2012a_accept_map: Optional[Dict[str, int]] = None,
    drop_unlabeled_cases: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    case_col = find_column(df.columns, CASE_COL_CANDIDATES)
    activity_col = find_column(df.columns, ACTIVITY_COL_CANDIDATES)
    ts_col = find_column(df.columns, TIMESTAMP_COL_CANDIDATES)

    if case_col is None:
        raise RuntimeError("Cannot find case id column.")
    if activity_col is None:
        raise RuntimeError("Cannot find activity column.")

    labeled = df.copy()
    case_labels = compute_case_labels(
        labeled,
        dataset=dataset,
        case_col=case_col,
        activity_col=activity_col,
        ts_col=ts_col,
        bpic2012a_accept_map=bpic2012a_accept_map,
    )
    labeled[label_col] = labeled[case_col].map(
        lambda x: numeric_label_to_word(case_labels.get(normalize_case_id(x)), positive_label, negative_label)
    )
    if drop_unlabeled_cases:
        labeled = labeled[labeled[label_col].notna()].copy()

    values = list(case_labels.values())
    summary = {
        "dataset": dataset,
        "case_col": case_col,
        "activity_col": activity_col,
        "timestamp_col": ts_col,
        "label_col": label_col,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "n_events": int(len(labeled)),
        "n_cases": int(labeled[case_col].nunique(dropna=True)),
        "n_positive_cases": int(sum(v == 1 for v in values if v is not None)),
        "n_negative_cases": int(sum(v == 0 for v in values if v is not None)),
        "n_unlabeled_cases": int(sum(v is None for v in values)),
        "rule": build_rule_descriptions(positive_label, negative_label)[dataset],
    }
    return labeled, summary


def make_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_path.relative_to(input_root)
    out = output_root / rel
    if out.name.lower().endswith(".xes.gz"):
        out = out.with_suffix("")
        out = out.with_suffix(".csv")
    else:
        out = out.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def discover_files(input_dir: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or not supported_file(path):
            continue
        dataset = resolve_dataset_name(str(path))
        if dataset is None:
            continue
        items.append((path, dataset))
    return items


def process_file(
    input_path: Path,
    dataset: str,
    output_path: Path,
    positive_label: str,
    negative_label: str,
    label_col: str,
    bpic2012a_accept_map: Optional[Dict[str, int]],
    drop_unlabeled_cases: bool,
) -> Dict[str, object]:
    df = read_table(input_path)
    labeled, summary = label_dataframe(
        df,
        dataset=dataset,
        positive_label=positive_label,
        negative_label=negative_label,
        label_col=label_col,
        bpic2012a_accept_map=bpic2012a_accept_map,
        drop_unlabeled_cases=drop_unlabeled_cases,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    summary["input_file"] = str(input_path)
    summary["output_file"] = str(output_path)
    return summary


def print_rules(positive_label: str, negative_label: str) -> None:
    rules = build_rule_descriptions(positive_label, negative_label)
    for dataset in rules:
        print(f"{dataset}: {rules[dataset]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate natural-language outcome labels for the mapped PPM datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./mapped_dataset"),
        help="Root directory of mapped datasets. Default: ./mapped_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./labeled_dataset"),
        help="Root directory for labeled outputs. Default: ./labeled_dataset",
    )
    parser.add_argument("--input", type=Path, default=None, help="Single input file.")
    parser.add_argument("--output", type=Path, default=None, help="Single output file.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for single-file mode.")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name. Default: label")
    parser.add_argument(
        "--positive-label",
        type=str,
        default=DEFAULT_POSITIVE_LABEL,
        help=f"Word used for the positive outcome. Default: {DEFAULT_POSITIVE_LABEL}",
    )
    parser.add_argument(
        "--negative-label",
        type=str,
        default=DEFAULT_NEGATIVE_LABEL,
        help=f"Word used for the negative outcome. Default: {DEFAULT_NEGATIVE_LABEL}",
    )
    parser.add_argument("--drop-unlabeled-cases", action="store_true", help="Drop cases whose labels cannot be computed.")
    parser.add_argument("--print-rules", action="store_true", help="Print the rule of every dataset and exit.")
    parser.add_argument("--summary-name", type=str, default="outcome_label_summary.csv", help="Summary CSV filename for batch mode.")
    parser.add_argument("--rules-json-name", type=str, default="outcome_label_rules.json", help="Rules JSON filename for batch mode.")
    args = parser.parse_args()

    if args.print_rules:
        print_rules(args.positive_label, args.negative_label)
        return

    if args.positive_label.strip() == "" or args.negative_label.strip() == "":
        raise SystemExit("--positive-label and --negative-label must be non-empty words.")
    if args.positive_label.strip().lower() == args.negative_label.strip().lower():
        raise SystemExit("--positive-label and --negative-label must be different words.")

    if args.input is not None:
        dataset = args.dataset or resolve_dataset_name(str(args.input))
        if dataset is None:
            raise SystemExit("Cannot resolve dataset name for single-file mode. Please pass --dataset.")
        if args.output is None:
            out = args.input.with_name(args.input.stem + "_labeled.csv")
        else:
            out = args.output
        out.parent.mkdir(parents=True, exist_ok=True)

        bpic_map: Optional[Dict[str, int]] = None
        if dataset == "BPIC2012W":
            search_root = args.input.parent.parent if args.input.parent != args.input else Path(".")
            files = discover_files(search_root)
            bpic_map = build_bpic2012a_accept_map(files)

        summary = process_file(
            args.input,
            dataset,
            out,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            label_col=args.label_col,
            bpic2012a_accept_map=bpic_map,
            drop_unlabeled_cases=args.drop_unlabeled_cases,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_dir)
    if not files:
        raise SystemExit(f"No supported dataset files found under: {input_dir}")

    bpic2012a_accept_map = build_bpic2012a_accept_map(files)

    summaries: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for input_path, dataset in files:
        output_path = make_output_path(input_path, input_dir, output_dir)
        try:
            summary = process_file(
                input_path,
                dataset,
                output_path,
                positive_label=args.positive_label,
                negative_label=args.negative_label,
                label_col=args.label_col,
                bpic2012a_accept_map=bpic2012a_accept_map,
                drop_unlabeled_cases=args.drop_unlabeled_cases,
            )
            summaries.append(summary)
            print(f"[OK] {dataset}: {input_path} -> {output_path}")
        except Exception as exc:
            errors.append({"dataset": dataset, "file": str(input_path), "error": str(exc)})
            print(f"[ERROR] {dataset}: {input_path}: {exc}", file=sys.stderr)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / args.summary_name
    summary_df.to_csv(summary_path, index=False)

    rules_path = output_dir / args.rules_json_name
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(build_rule_descriptions(args.positive_label, args.negative_label), f, ensure_ascii=False, indent=2)

    if errors:
        error_path = output_dir / "outcome_label_errors.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"Finished with {len(errors)} errors. See: {error_path}")
    else:
        print("Finished without errors.")
    print(f"Summary written to: {summary_path}")
    print(f"Rules written to: {rules_path}")


if __name__ == "__main__":
    main()
