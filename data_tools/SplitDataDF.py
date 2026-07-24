import polars as pl
from pathlib import Path


def split_dataframe(
    data_file: str | Path,
    output_dir: str | Path,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    output_format: str = "parquet",
) -> dict[str, pl.DataFrame]:
    """
    Split a Polars DataFrame into train, validation, and test sets.

    Args:
        data_file:     Path to the input data file (parquet, csv, json, ipc).
        output_dir:    Directory to save the split DataFrames.
        train_size:    Fraction for training set   (default 0.7).
        val_size:      Fraction for validation set  (default 0.15).
        test_size:     Fraction for test set         (default 0.15).
        seed:          Random seed for shuffling     (default 42).
        output_format: Output file format — "parquet", "csv", "json", or "ipc".

    Returns:
        Dict with keys "train", "val", "test" mapping to the split DataFrames.
    """
    # --- validate split ratios ---
    total = train_size + val_size + test_size
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {total:.4f} "
            f"(train={train_size}, val={val_size}, test={test_size})"
        )

    # --- read input file ---
    data_file = Path(data_file)
    readers = {
        ".parquet": pl.read_parquet,
        ".csv": pl.read_csv,
        ".json": pl.read_json,
        ".ipc": pl.read_ipc,
        ".arrow": pl.read_ipc,
    }
    reader = readers.get(data_file.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported file format '{data_file.suffix}'. "
            f"Supported: {', '.join(readers.keys())}"
        )
    df = reader(data_file)
    print(f"Loaded {data_file.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- shuffle ---
    # df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    # --- compute split indices ---
    n = df.shape[0]
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    splits = {
        "train": df.slice(0, train_end),
        "val": df.slice(train_end, val_end - train_end),
        "test": df.slice(val_end, n - val_end),
    }

    # --- save ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        "parquet": lambda frame, p: frame.write_parquet(p.with_suffix(".parquet")),
        "csv": lambda frame, p: frame.write_csv(p.with_suffix(".csv")),
        "json": lambda frame, p: frame.write_json(p.with_suffix(".json")),
        "ipc": lambda frame, p: frame.write_ipc(p.with_suffix(".ipc")),
    }
    writer = writers.get(output_format)
    if writer is None:
        raise ValueError(
            f"Unsupported output format '{output_format}'. "
            f"Supported: {', '.join(writers.keys())}"
        )

    for name, split_df in splits.items():
        out_path = output_dir / name
        writer(split_df, out_path)
        print(f"  {name:>5}: {split_df.shape[0]:>7} rows  ->  {out_path.with_suffix('.' + output_format)}")

    return splits


# ── Configure and run ────────────────────────────────────────────────
if __name__ == "__main__":

    DATA_FILE = "data_artefacts/CICAPT-IIoT/CICAPT_Phase1.parquet"   # <-- your input file
    OUTPUT_DIR = "data_artefacts/CICAPT-IIoT/Phase1_split"           # <-- where to save the splits
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    SEED = 42
    OUTPUT_FORMAT = "parquet"            # parquet | csv | json | ipc

    splits = split_dataframe(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        seed=SEED,
        output_format=OUTPUT_FORMAT,
    )