"""
Strict temporal splitter for flow-level models.

Each split is a contiguous wall-clock interval of the capture. A flow that crosses a
boundary is cut at that boundary — every flow, uniformly, with no exceptions.

    time ->
    |------- train -------|-- test --|-- ad_fit --|-- ad_calib --|-- val --|- late -|
      a flow spanning the capture contributes packets to each interval it covers

Boundaries sit at **packet quantiles**, not at even divisions of the clock: the boundary
for a cumulative share q is the timestamp of the packet at index round(q x N) in global
time order. That hits the ratios in packets while keeping every split a clean time
interval, which even wall-clock division would not do when the traffic rate varies.

Why no flow selection
---------------------
There is deliberately no `long_flow_duration_s`, no `min_packets_per_piece`, and no
"keep short flows whole" path. A previous version cut only flows that passed such gates
and assigned the rest whole, ordered by first packet. That silently stops being a
temporal split as soon as flows are long-lived: on CICAPT-IIoT Phase 1 the median
conversation lasts 61% of the capture, so every split spanned all four days and nothing
was held out in time at all. Uniform cutting at the boundary is the only sound option,
and conditioning it on flow length or duration reintroduces exactly that failure.

A flow appearing in several splits is therefore intended, not leakage. The invariant that
matters is that the intervals are disjoint and ordered in time, which `_check_intervals`
asserts before anything is written.

SCOPE: for *unsupervised anomaly detection* on captures containing **no attacks**. Splits
are time periods, so an attack in a labelled capture would land wholly inside whichever
split covers its window. Attack captures stay whole and are used as evaluation sets on
their own.

Packets sharing a timestamp are never separated: a boundary is snapped forward to the
first packet of the next distinct timestamp.

Run from the repo root, after editing the constants in the __main__ block:
    python -m data_tools.SplitFlowsDF
"""
import polars as pl
import numpy as np
import json
from pathlib import Path


# `src_ip:src_port -> dst_ip:dst_port (proto)`. Not used by the split itself — it is kept
# here because `DataUtils._flow_key_pattern` imports it, and the greedy `.*` binding each
# port to the *last* colon of its endpoint is what lets IPv6 keys parse:
#   `fe80::b067:5f59:5094:9ba5:0 -> ff02::fb:0 (ICMPv6)`
FLOW_KEY_RE = r"^(.*):(\d+) -> (.*):(\d+) \((.*)\)$"


def _boundary_indices(ts: np.ndarray, fractions: list[float]) -> list[int]:
    """
    Row indices where each split ends, given time-sorted timestamps.

    The index for a cumulative share is snapped forward to the start of the next distinct
    timestamp, so packets recorded at the same instant always land in the same split. That
    snapping can make a split slightly larger than its target, which is why the report
    prints achieved ratios rather than assuming the requested ones.
    """
    n = ts.shape[0]
    indices, cumulative = [], 0.0
    for fraction in fractions[:-1]:
        cumulative += fraction
        raw = min(int(round(cumulative * n)), n - 1)
        indices.append(int(np.searchsorted(ts, ts[raw], side="left")))
    return indices


def _check_intervals(bounds: list[int], ts: np.ndarray, names: list[str]) -> None:
    """Assert the splits are non-empty, disjoint, and ordered along the time axis."""
    edges = [0, *bounds, ts.shape[0]]
    for name, start, end in zip(names, edges, edges[1:]):
        if end <= start:
            raise AssertionError(
                f"split {name!r} is empty (rows {start}:{end}). Its share is too small for "
                f"this capture, or several splits fell inside one timestamp.")
    for i, cut in enumerate(bounds):
        if ts[cut - 1] >= ts[cut]:
            raise AssertionError(
                f"boundary between {names[i]!r} and {names[i + 1]!r} falls inside a single "
                f"timestamp ({ts[cut]}); the intervals would overlap in time.")


def split_flows(
    data_file: str | Path,
    output_dir: str | Path,
    splits: tuple[tuple[str, float], ...],
    dry_run: bool = False,
) -> dict[str, pl.DataFrame]:
    """
    Split a packet parquet into contiguous time intervals.

    Args:
        data_file:  Input packet parquet (proto_hierarchy, flow_key, timestamp_s,
                    timestamp_us, data, mask, header_len).
        output_dir: Directory for <name>.parquet + split_report.json.
        splits:     Ordered `(name, fraction)` pairs in time order; fractions are packet
                    shares and must sum to 1.0.
        dry_run:    Print the report, write nothing.

    Returns:
        Dict mapping each split name to its DataFrame (empty on a dry run).
    """
    names = [name for name, _ in splits]
    fractions = [fraction for _, fraction in splits]
    if len(set(names)) != len(names):
        raise ValueError(f"split names must be unique, got {names}")
    if not (0.999 <= sum(fractions) <= 1.001):
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions):.4f}")

    data_file = Path(data_file)
    df = pl.read_parquet(data_file)
    missing = {"flow_key", "timestamp_s", "timestamp_us"} - set(df.columns)
    if missing:
        raise ValueError(f"{data_file.name} is missing column(s): {sorted(missing)}")

    columns = df.columns
    df = df.with_columns(
        (pl.col("timestamp_s") * 1_000_000 + pl.col("timestamp_us")).alias("_ts")
    ).sort("_ts")
    ts = df["_ts"].to_numpy()
    print(f"Loaded {data_file.name}: {ts.shape[0]} packets spanning "
          f"{(ts[-1] - ts[0]) / 1e6:.0f} s")

    bounds = _boundary_indices(ts, fractions)
    _check_intervals(bounds, ts, names)

    edges = [0, *bounds, ts.shape[0]]
    lengths = [end - start for start, end in zip(edges, edges[1:])]
    df = df.with_columns(pl.Series("split", np.repeat(names, lengths)))

    report = {
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "requested": dict(splits),
        "total_packets": ts.shape[0],
        "flow_keys": df["flow_key"].n_unique(),
        "capture_span_s": (ts[-1] - ts[0]) / 1e6,
        "achieved": {},
    }
    for name in names:
        part = df.filter(pl.col("split") == name)
        first, last = int(part["_ts"].min()), int(part["_ts"].max())
        report["achieved"][name] = {
            "packets": part.height,
            "ratio": part.height / ts.shape[0],
            "flow_keys": part["flow_key"].n_unique(),
            "first_ts_us": first,
            "last_ts_us": last,
            "span_s": (last - first) / 1e6,
        }
    # How many flows the boundaries cut, i.e. appear in more than one interval. Expected to
    # be large on long-lived traffic; reported so the scale of the cutting is visible.
    spans = (df.group_by("flow_key").agg(pl.col("split").n_unique().alias("n"))
             .filter(pl.col("n") > 1).height)
    report["flow_keys_crossing_a_boundary"] = spans

    print(json.dumps(report, indent=2))
    if dry_run:
        print("Dry run: nothing written.")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in names:
        part = df.filter(pl.col("split") == name).select(columns)
        part.write_parquet(output_dir / f"{name}.parquet")
        out[name] = part
        print(f"  {name:>9}: {part.height:>9} rows  ->  {output_dir / f'{name}.parquet'}")
    with open(output_dir / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report -> {output_dir / 'split_report.json'}")
    return out


# ── Configure and run ────────────────────────────────────────────────
if __name__ == "__main__":

    # Normal-only capture — see the SCOPE note in the module docstring.
    DATA_FILE = "data_artefacts/merged_extractor/CICAPT-IIoT/CICAPT_Phase1.parquet"
    OUTPUT_DIR = "data_artefacts/merged_extractor/CICAPT-IIoT/flow_split"

    # One role per split, in time order, so none has to be spent twice:
    #   train     packet/sequence AE pretraining
    #   test      monitored during AE training, picks the checkpoint
    #   ad_fit    fits the anomaly detectors
    #   ad_calib  score quantiles -> thresholds (no labels)
    #   val       final held-out normals; the reported false-positive rate
    #   late      unseen tail, the temporal confound floor
    SPLITS = (
        ("train",    0.55),
        ("test",     0.10),
        ("ad_fit",   0.10),
        ("ad_calib", 0.10),
        ("val",      0.10),
        ("late",     0.05),
    )
    DRY_RUN = False   # set True to print the report without writing

    split_flows(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        splits=SPLITS,
        dry_run=DRY_RUN,
    )
