from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summarize(df: pd.DataFrame, name: str, date_col: str, target_col: str) -> None:
    dt = pd.to_datetime(df[date_col], errors="coerce")
    pos_rate = None
    if target_col in df.columns:
        # Expect values like Yes/No; treat "Yes" as positive when possible, else use mean if numeric.
        s = df[target_col]
        if pd.api.types.is_numeric_dtype(s):
            pos_rate = float(s.mean())
        else:
            pos_rate = float((s.astype(str).str.lower() == "yes").mean())

    print(f"\n[{name}] rows={len(df)}")
    print(f"[{name}] {date_col} min={dt.min()} max={dt.max()} na_dates={int(dt.isna().sum())}")
    if pos_rate is not None:
        print(f"[{name}] {target_col} positive_rate≈{pos_rate:.4f}")


def split_by_date_threshold(
    df: pd.DataFrame,
    date_col: str,
    threshold: pd.Timestamp,
    inclusive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = pd.to_datetime(df[date_col], errors="coerce")
    if inclusive:
        mask_train = dt <= threshold
    else:
        mask_train = dt < threshold

    train_df = df.loc[mask_train].copy()
    test_df = df.loc[~mask_train].copy()
    return train_df, test_df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split dataset into train/test by Effective To Date threshold (writes CSV files)."
    )
    ap.add_argument(
        "--input",
        type=str,
        default="dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv",
        help="Input CSV path (relative to repo root).",
    )
    ap.add_argument(
        "--date-col",
        type=str,
        default="Effective To Date",
        help="Date column name used for split.",
    )
    ap.add_argument(
        "--target-col",
        type=str,
        default="Response",
        help="Target column name (used only for reporting).",
    )
    ap.add_argument(
        "--cutoff",
        type=str,
        default="2011-02-20",
        help="Cutoff date (YYYY-MM-DD). Dates <= cutoff go to train.",
    )
    ap.add_argument(
        "--inclusive",
        action="store_true",
        default=True,
        help="If set (default), include cutoff date in train (<= cutoff).",
    )
    ap.add_argument(
        "--min-test-rows",
        type=int,
        default=500,
        help="If resulting test rows < this, auto-move cutoff earlier until test is large enough.",
    )
    ap.add_argument(
        "--output-train",
        type=str,
        default="dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—train_timecut.csv",
        help="Output train CSV path (relative to repo root).",
    )
    ap.add_argument(
        "--output-test",
        type=str,
        default="dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—test_timecut.csv",
        help="Output test CSV path (relative to repo root).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    in_path = repo_root / args.input
    out_train = repo_root / args.output_train
    out_test = repo_root / args.output_test

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    if args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found in input.")

    cutoff = pd.to_datetime(args.cutoff)

    # Auto-adjust cutoff if test too small
    # We move cutoff earlier by stepping through unique sorted dates.
    dt_all = pd.to_datetime(df[args.date_col], errors="coerce")
    uniq_dates = sorted([d for d in dt_all.dropna().unique()])

    def _do_split(c: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        return split_by_date_threshold(df, args.date_col, c, inclusive=args.inclusive)

    train_df, test_df = _do_split(cutoff)
    if len(test_df) < int(args.min_test_rows):
        # Find the earliest cutoff that yields enough test rows (move cutoff earlier).
        for d in reversed(uniq_dates):
            c = pd.Timestamp(d)
            tr2, te2 = _do_split(c)
            if len(te2) >= int(args.min_test_rows):
                cutoff = c
                train_df, test_df = tr2, te2
                break

    print("## Split summary")
    print(f"input={in_path}")
    print(f"cutoff_used={cutoff.date()} inclusive={args.inclusive}")
    _summarize(train_df, "train", args.date_col, args.target_col)
    _summarize(test_df, "test", args.date_col, args.target_col)

    # Write outputs
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    print(f"\nWrote train: {out_train}")
    print(f"Wrote test:  {out_test}")


if __name__ == "__main__":
    main()

