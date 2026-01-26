from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summarize(df: pd.DataFrame, name: str, target_col: str) -> None:
    pos_rate = None
    if target_col in df.columns:
        s = df[target_col]
        if pd.api.types.is_numeric_dtype(s):
            pos_rate = float(s.mean())
        else:
            pos_rate = float((s.astype(str).str.lower() == "yes").mean())

    print(f"\n[{name}] rows={len(df)}")
    if pos_rate is not None:
        print(f"[{name}] {target_col} positive_rate≈{pos_rate:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split dataset into train/test by ratio (writes CSV files)."
    )
    ap.add_argument(
        "--input",
        type=str,
        default="dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv",
        help="Input CSV path (relative to repo root).",
    )
    ap.add_argument(
        "--target-col",
        type=str,
        default="Response",
        help="Target column name.",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size ratio (e.g., 0.2).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    ap.add_argument(
        "--stratify",
        action="store_true",
        default=True,
        help="If set (default), stratify split by target column.",
    )
    ap.add_argument(
        "--no-stratify",
        dest="stratify",
        action="store_false",
        help="Disable stratification.",
    )
    ap.add_argument(
        "--output-train",
        type=str,
        default="dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—train_ratio.csv",
        help="Output train CSV path (relative to repo root).",
    )
    ap.add_argument(
        "--output-test",
        type=str,
        default="dataset/WA_Fn-UseC_-Marketing-Customer-Value-Analysis—test_ratio.csv",
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
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in input.")

    from sklearn.model_selection import train_test_split

    stratify = df[args.target_col] if args.stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        shuffle=True,
        stratify=stratify,
    )

    print("## Split summary (ratio)")
    print(f"input={in_path}")
    print(f"test_size={float(args.test_size)} stratify={bool(args.stratify)} random_state={int(args.random_state)}")
    _summarize(train_df, "train", args.target_col)
    _summarize(test_df, "test", args.target_col)

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    print(f"\nWrote train: {out_train}")
    print(f"Wrote test:  {out_test}")


if __name__ == "__main__":
    main()

