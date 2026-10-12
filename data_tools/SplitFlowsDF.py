"""
Strict temporal splitter for flow-level models.

The goal is that each split is one contiguous wall-clock interval of the capture. A flow
that crosses a boundary is cut at that boundary — every flow, uniformly, with no
exceptions.

    time ->
    |------- train -------|-- test --|-- ad_fit --|-- ad_calib --|-- val --|- late -|
      a flow spanning the capture contributes packets to each interval it covers

Exception: separate recording sessions (`session_gap_s`). When a capture is several
sessions recorded one after another, possibly with different settings or simulated
environments, a single interval per role hands whole sessions to roles, and evaluation
then measures the jump between sessions rather than held-out time. Then every session is
cut into the roles on its own clock, and each role is the union of its piece of every
session:

    |-- session A: train | test | ad_fit | ad_calib | val --| ... |-- session B: train | ... |

Within a session the boundaries are clean time boundaries, as above. Across sessions a
role's later piece can come after another role's earlier piece, so `val` measures held-out
time within sessions, not generalisation to an unseen session. Flows break at session
edges (a `session` column is written for that). Sessions below `min_session_share` of the
packets are left out of every split and listed in the report.

The fractions are shares of either packets or time (`boundary_mode`):

  packets  the boundary for a cumulative share q is the timestamp of the packet at index
           round(q x N) in global time order. The ratios hold in packet counts; the time
           each split covers follows the traffic rate.
  time     the boundary is the instant t_first + q x (t_last - t_first). Each split gets
           its share of the clock; its packet count follows the traffic rate.
  active_time  as `time`, on a clock where every pause between packets counts as at most
           `pause_cap_s`. Each split gets its share of the time traffic was recorded, and
           a pause inside a recording cannot leave a split empty.

All keep every split a clean time interval (per session). With sessions, t_first and
t_last are the session's own first and last packet.

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
matters is that the intervals are disjoint and ordered in time (within each session),
which `_check_intervals` asserts before anything is written. Dropping a whole tiny
session (`min_session_share`) is a rule about sessions, not about flows: no flow inside a
kept session is ever selected or excluded.

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


BOUNDARY_MODES = ("packets", "time", "active_time")


def _active_clock(ts: np.ndarray, pause_cap_s: float) -> np.ndarray:
    """Microseconds of *active* time up to each packet: every gap between consecutive
    packets counts as at most `pause_cap_s`, so pauses in the recording take no room."""
    gaps = np.minimum(np.diff(ts), pause_cap_s * 1e6)
    return np.concatenate([[0.0], np.cumsum(gaps, dtype=np.float64)])


def _boundary_indices(ts: np.ndarray, fractions: list[float],
                      mode: str = "packets", pause_cap_s: float = 60.0) -> list[int]:
    """
    Row indices where each split ends, given time-sorted timestamps.

    `packets`:     the boundary for cumulative share q is the timestamp of packet
                   round(q x N), so the shares are met in packet counts.
    `time`:        the boundary is the instant t_first + q x (t_last - t_first), so the
                   shares are met in wall-clock time and packet counts follow the rate.
    `active_time`: as `time`, but on the active clock (`_active_clock`): pauses longer
                   than `pause_cap_s` count as `pause_cap_s`. Shares of the time traffic
                   was actually recorded, and a pause can never swallow a whole split.

    Either way the index is the first packet at or after the boundary timestamp
    (searchsorted, side="left"), so packets recorded at the same instant always land in
    the same split. That snapping can make a split slightly larger than its target, which
    is why the report prints achieved shares rather than assuming the requested ones.
    """
    if mode not in BOUNDARY_MODES:
        raise ValueError(f"boundary mode must be one of {BOUNDARY_MODES}, got {mode!r}")
    n = ts.shape[0]
    active = _active_clock(ts, pause_cap_s) if mode == "active_time" else None
    indices, cumulative = [], 0.0
    for fraction in fractions[:-1]:
        cumulative += fraction
        if mode == "packets":
            raw = min(int(round(cumulative * n)), n - 1)
            boundary_ts = ts[raw]
        elif mode == "time":
            boundary_ts = ts[0] + cumulative * (ts[-1] - ts[0])
        else:
            raw = min(int(np.searchsorted(active, cumulative * active[-1], side="left")), n - 1)
            boundary_ts = ts[raw]
        indices.append(int(np.searchsorted(ts, boundary_ts, side="left")))
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




def _sessions(ts: np.ndarray, session_gap_s: float | None) -> list[tuple[int, int]]:
    """(start, end) row ranges of the recording sessions in time-sorted timestamps.

    A new session starts wherever two consecutive packets are more than `session_gap_s`
    apart. `None` means one session, i.e. the whole capture -- the strict single-interval
    split. This is a cut on the clock only; no flow is looked at.
    """
    if session_gap_s is None:
        return [(0, ts.shape[0])]
    breaks = np.flatnonzero(np.diff(ts) > session_gap_s * 1e6) + 1
    edges = [0, *breaks.tolist(), ts.shape[0]]
    return list(zip(edges, edges[1:]))


def split_flows(
    data_file: str | Path,
    output_dir: str | Path,
    splits: tuple[tuple[str, float], ...],
    dry_run: bool = False,
    boundary_mode: str = "packets",
    session_gap_s: float | None = None,
    min_session_share: float = 0.0,
    pause_cap_s: float = 60.0,
) -> dict[str, pl.DataFrame]:
    """
    Split a packet parquet into time intervals, one per role (per recording session).

    Args:
        data_file:         Input packet parquet (proto_hierarchy, flow_key, timestamp_s,
                           timestamp_us, data, mask, header_len).
        output_dir:        Directory for <name>.parquet + split_report.json.
        splits:            Ordered `(name, fraction)` pairs in time order; fractions must
                           sum to 1.0.
        dry_run:           Print the report, write nothing.
        boundary_mode:     What the fractions are shares of: `packets` (packet counts),
                           `time` (the wall-clock span) or `active_time` (the span with
                           pauses capped at `pause_cap_s`). See `_boundary_indices`.
        session_gap_s:     None (default): the strict split -- one interval per role over
                           the whole capture. A number: the capture is a sequence of
                           separate recording sessions, separated by pauses longer than
                           this; every session is cut into the roles on its own clock and
                           each role is the union of its piece of every session. A
                           `session` column is then written, so flows break at sessions.
        min_session_share: Sessions holding fewer than this share of all packets are left
                           out of every split, and listed as dropped in the report. Only
                           meaningful with `session_gap_s`.
        pause_cap_s:       For `active_time`: the longest a pause between two packets
                           counts on the active clock.

    Returns:
        Dict mapping each split name to its DataFrame (empty on a dry run).
    """
    names = [name for name, _ in splits]
    fractions = [fraction for _, fraction in splits]
    if len(set(names)) != len(names):
        raise ValueError(f"split names must be unique, got {names}")
    if not (0.999 <= sum(fractions) <= 1.001):
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions):.4f}")
    if min_session_share and session_gap_s is None:
        raise ValueError("min_session_share needs session_gap_s; without sessions it would "
                         "drop the whole capture or nothing")

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
    n = ts.shape[0]
    print(f"Loaded {data_file.name}: {n} packets spanning {(ts[-1] - ts[0]) / 1e6:.0f} s")

    # Cut every kept session into the roles on its own clock. With one session this is
    # exactly the strict split over the whole capture.
    split_of_row = np.full(n, None, dtype=object)       # None = dropped session
    session_of_row = np.zeros(n, dtype=np.int32)
    sessions_report = []
    pieces = {name: [] for name in names}     # per role: (start_ts, end_ts, rows, active_us)
    kept_active = 0.0
    for s, (lo, hi) in enumerate(_sessions(ts, session_gap_s)):
        session_of_row[lo:hi] = s
        entry = {"session": s, "first_ts_us": int(ts[lo]), "last_ts_us": int(ts[hi - 1]),
                 "span_s": (ts[hi - 1] - ts[lo]) / 1e6, "packets": hi - lo,
                 "share": (hi - lo) / n}
        if (hi - lo) / n < min_session_share:
            entry["dropped"] = True
            sessions_report.append(entry)
            print(f"  session {s}: {hi - lo} packets ({(hi - lo) / n:.4%}) below "
                  f"min_session_share {min_session_share} -- left out of every split")
            continue
        ts_s = ts[lo:hi]
        bounds = _boundary_indices(ts_s, fractions, boundary_mode, pause_cap_s)
        _check_intervals(bounds, ts_s, [f"{name} (session {s})" for name in names])
        edges = [0, *bounds, hi - lo]
        # A piece's clock share runs from its first packet to the next piece's first packet
        # (the last runs to the session's final packet), so shares sum to 1 per session.
        starts = [int(ts_s[e]) for e in edges[:-1]] + [int(ts_s[-1])]
        active = _active_clock(ts_s, pause_cap_s)
        a_starts = [float(active[e]) for e in edges[:-1]] + [float(active[-1])]
        kept_active += active[-1]
        for i, name in enumerate(names):
            split_of_row[lo + edges[i]:lo + edges[i + 1]] = name
            pieces[name].append((starts[i], starts[i + 1], edges[i + 1] - edges[i],
                                 a_starts[i + 1] - a_starts[i]))
        entry["boundaries_ts_us"] = starts[1:-1]
        sessions_report.append(entry)

    df = df.with_columns(pl.Series("split", split_of_row.tolist(), dtype=pl.Utf8),
                         pl.Series("session", session_of_row))
    df = df.filter(pl.col("split").is_not_null())
    kept_span = sum(e["last_ts_us"] - e["first_ts_us"]
                    for e in sessions_report if not e.get("dropped"))

    report = {
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "boundary_mode": boundary_mode,
        "session_gap_s": session_gap_s,
        "min_session_share": min_session_share,
        "pause_cap_s": pause_cap_s,
        "requested": dict(splits),
        "total_packets": n,
        "kept_packets": df.height,
        "flow_keys": df["flow_key"].n_unique(),
        "capture_span_s": (ts[-1] - ts[0]) / 1e6,
        "sessions": sessions_report,
        "achieved": {},
    }
    for name in names:
        part = df.filter(pl.col("split") == name)
        first, last = int(part["_ts"].min()), int(part["_ts"].max())
        report["achieved"][name] = {
            "packets": part.height,
            "ratio": part.height / df.height,
            "time_share": sum(p[1] - p[0] for p in pieces[name]) / kept_span,
            "active_time_share": sum(p[3] for p in pieces[name]) / kept_active,
            "flow_keys": part["flow_key"].n_unique(),
            # Outer extent. With several sessions the role is a union of pieces, so the
            # span between its first and last packet includes other roles' time.
            "first_ts_us": first,
            "last_ts_us": last,
            "span_s": (last - first) / 1e6,
            "pieces": [{"first_ts_us": start, "span_s": (end - start) / 1e6,
                        "active_s": active_us / 1e6, "packets": rows}
                       for start, end, rows, active_us in pieces[name]],
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

    # The session column is written only for a session split, so a strict split keeps the
    # input schema exactly. Its presence is what makes the flow index break flows at
    # sessions (DataUtils.PreTrainingDatasetHandler.build_flow_index).
    out_columns = columns + (["session"] if session_gap_s is not None else [])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in names:
        part = df.filter(pl.col("split") == name).select(out_columns)
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
    DATA_FILE = "data_artefacts/merged_extractor/IIoTset-Ferrag/NormalMerged.parquet"
    OUTPUT_DIR = "data_artefacts/merged_extractor/IIoTset-Ferrag/flow_split"

    # One role per split, in time order, so none has to be spent twice:
    #   train     packet/sequence AE pretraining
    #   test      monitored during AE training, picks the checkpoint
    #   ad_fit    fits the anomaly detectors
    #   ad_calib  score quantiles -> thresholds (no labels)
    #   val       final held-out normals; the reported false-positive rate
    #   late      unseen tail, the temporal confound floor (CICAPT only for now)
    #
    # Must match DATASETS[...].splits. CICAPT was cut strictly (one session) with
    # train .55 / test .10 / ad_fit .10 / ad_calib .10 / val .10 / late .05, "packets".
    SPLITS = (
        ("train",    0.40),
        ("test",     0.10),
        ("ad_fit",   0.20),
        ("ad_calib", 0.20),
        ("val",      0.10),
    )
    # IIoTset was recorded as a series of separate runs with different settings, hours to
    # weeks apart (docs/experiments/2026-10-07-iiotset-session-split.md, 2026-10-11
    # follow-up). A strict split handed whole runs to roles: the first six-split run fitted
    # the AEs on days 0-36 and took every AD split from day 44. So each session is cut into
    # the roles on its own clock instead.
    BOUNDARY_MODE = "active_time"   # within each session: shares of its recorded time
    PAUSE_CAP_S = 60                # pauses inside a session count as 60 s
    # 90 min: the high-rate run at hour 881 (segment 9) follows a 98.0-min pause, while the
    # longest pauses inside runs that belong together are 80.3 and 80.1 min. The 1-day gap
    # of the 2026-10-07 run merged it into session 2, so ad_fit never saw high-rate
    # traffic. Gives 8 kept sessions.
    SESSION_GAP_S = 5_400           # a pause longer than this starts a new session
    MIN_SESSION_SHARE = 0.01        # day 28 (7,020 packets, 0.07%) is too small to cut
    DRY_RUN = False                 # set True to print the report without writing

    split_flows(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        splits=SPLITS,
        dry_run=DRY_RUN,
        boundary_mode=BOUNDARY_MODE,
        session_gap_s=SESSION_GAP_S,
        min_session_share=MIN_SESSION_SHARE,
        pause_cap_s=PAUSE_CAP_S,
    )
