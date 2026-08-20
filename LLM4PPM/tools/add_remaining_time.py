from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


UNIT_TO_SECONDS = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
}

CASE_COL_CANDIDATES = [
    "case_id",
    "caseid",
    "case",
    "case:concept:name",
    "case_concept_name",
    "trace_id",
    "traceid",
    "Case ID",
]

TIMESTAMP_COL_CANDIDATES = [
    "timestamp",
    "time:timestamp",
    "time_timestamp",
    "complete_timestamp",
    "event_time",
    "datetime",
    "time",
]


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    exact = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    return None


def add_remaining_time(
    input_csv: str | Path,
    output_csv: str | Path,
    case_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    out_col: str = "remaining_time",
    unit: str = "days",
    round_ndigits: Optional[int] = 0,
) -> pd.DataFrame:
    if unit not in UNIT_TO_SECONDS:
        raise ValueError(f"unit must be one of {sorted(UNIT_TO_SECONDS)}")

    df = pd.read_csv(input_csv)
    case_col = case_col or find_column(df.columns, CASE_COL_CANDIDATES)
    timestamp_col = timestamp_col or find_column(df.columns, TIMESTAMP_COL_CANDIDATES)
    if case_col is None:
        raise ValueError("Cannot find the case id column. Pass --case-col explicitly.")
    if timestamp_col is None:
        raise ValueError("Cannot find the timestamp column. Pass --timestamp-col explicitly.")

    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    if ts.isna().any():
        print(f"[WARN] {int(ts.isna().sum())} timestamps could not be parsed.")
    end_ts = ts.groupby(df[case_col]).transform("max")
    remaining = (end_ts - ts).dt.total_seconds() / UNIT_TO_SECONDS[unit]
    df[out_col] = remaining.clip(lower=0)
    if round_ndigits is not None:
        df[out_col] = df[out_col].round(round_ndigits)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def iter_csv_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(p for p in input_dir.rglob("*.csv") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Add remaining-time labels to event-log CSV files.")
    parser.add_argument("--input", type=Path, default=None, help="Single input CSV file.")
    parser.add_argument("--output", type=Path, default=None, help="Single output CSV file.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for batch output.")
    parser.add_argument("--case-col", type=str, default=None)
    parser.add_argument("--timestamp-col", type=str, default=None)
    parser.add_argument("--out-col", type=str, default="remaining_time")
    parser.add_argument("--unit", choices=sorted(UNIT_TO_SECONDS), default="days")
    parser.add_argument("--round-ndigits", type=int, default=0)
    args = parser.parse_args()

    if args.input:
        if args.output is None:
            raise SystemExit("--output is required when --input is used.")
        add_remaining_time(
            args.input,
            args.output,
            case_col=args.case_col,
            timestamp_col=args.timestamp_col,
            out_col=args.out_col,
            unit=args.unit,
            round_ndigits=args.round_ndigits,
        )
        print(f"[OK] {args.input} -> {args.output}")
        return

    if args.input_dir is None or args.output_dir is None:
        raise SystemExit("Use either --input/--output or --input-dir/--output-dir.")

    for path in iter_csv_files(args.input_dir):
        output = args.output_dir / path.relative_to(args.input_dir)
        add_remaining_time(
            path,
            output,
            case_col=args.case_col,
            timestamp_col=args.timestamp_col,
            out_col=args.out_col,
            unit=args.unit,
            round_ndigits=args.round_ndigits,
        )
        print(f"[OK] {path} -> {output}")


if __name__ == "__main__":
    main()
