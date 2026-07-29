"""
Flow-aware train/test/val splitter for flow-level models.

`SplitDataDF.py` cuts a packet parquet with contiguous `df.slice()` calls. For packet-level
pretraining that is fine — a row is an independent sample — but for flow-level models it cuts
flows at arbitrary points and scatters one flow over several splits.

This splitter works at flow granularity instead:

  * **Short flows are kept whole** and land in exactly one split.
  * **Long-lived flows are cut by packet timestamp**, earliest packets first, in the order
    train -> test -> val.
  * Split ratios are hit **in packets**, not in flows.

SCOPE: this tool is for *unsupervised anomaly detection*, on captures that contain **no
attacks**. Short flows are assigned chronologically, so train/test/val correspond to different
phases of the capture. Split a labelled attack capture with this and the attack classes end up
concentrated in whichever split covers their time window. Attack captures stay whole and are
used as evaluation sets on their own (that is how `AnomalyDetection/` already consumes the
per-attack parquets).

Flow identity
-------------
The `flow_key` column written by the Rust extractor cannot be grouped on directly:

  1. It is **not normalized**. `feature_extraction/src/main.rs:104-105` stringifies the key
     built by `FlowKey::from_parsed_packet` (which ends in `Self::new`, not
     `Self::new_normalized`); `key.normalize()` on the next line only feeds the FlowTracker
     HashMap. So `A:59573 -> B:1883 (...)` and `B:1883 -> A:59573 (...)` are two keys for one
     conversation.
  2. Its protocol component is the **whole proto_hierarchy** (`flow_tracker.rs:24`), so one TCP
     connection fragments into `... (Ethernet->IPv4->TCP)` for bare ACKs and
     `... (Ethernet->IPv4->TCP->MQTT)` for payload-bearing packets.

Either defect would leak half a conversation across the split boundary. We therefore group on a
canonical *conversation key* derived here in Python (see `conversation_key_map`), while writing
the `flow_key` column out **untouched** — `PreTrainingDatasetHandler.build_flow_index`,
`CachePacketLatents.py` and the existing latent caches keep working unchanged. Once the
extractor is fixed (see TODO.md) this canonicalization collapses to a plain
`group_by("flow_key")`.

Run from the repo root, after editing the constants in the __main__ block:
    python -m data_tools.SplitFlowsDF
"""
import polars as pl
import json
from pathlib import Path


# `src_ip:src_port -> dst_ip:dst_port (proto_hierarchy)`. The greedy `.*` binds each port to
# the *last* colon of its endpoint, which is what lets IPv6 keys parse:
#   `fe80::b067:5f59:5094:9ba5:0 -> ff02::fb:0 (Ethernet->IPv6->ICMPv6)`
FLOW_KEY_RE = r"^(.*):(\d+) -> (.*):(\d+) \((.*)\)$"

# Hierarchy tokens at which we stop when reducing a proto_hierarchy to its transport prefix,
# so `Ethernet->IPv4->TCP` and `Ethernet->IPv4->TCP->MQTT` collapse to the same conversation.
TRANSPORT_TOKENS = {"TCP", "UDP", "ICMP", "ICMPv6", "IGMP", "ARP"}

SPLIT_NAMES = ("train", "test", "val")


# ── flow identity ────────────────────────────────────────────────────

def transport_prefix(proto_hierarchy: str) -> str:
    """
    Reduce a proto_hierarchy to everything up to and including its transport token.

    `Ethernet->IPv4->TCP->MQTT` -> `Ethernet->IPv4->TCP`. A hierarchy with no known transport
    token is returned unchanged, so unrecognised stacks simply stay as specific as they were.
    """
    tokens = proto_hierarchy.split("->")
    for i, token in enumerate(tokens):
        if token in TRANSPORT_TOKENS:
            return "->".join(tokens[: i + 1])
    return proto_hierarchy


def conversation_key_map(flow_keys: pl.Series) -> pl.DataFrame:
    """
    Build a `flow_key -> conv_key` lookup table.

    Runs on the *unique* flow keys (tens of thousands) rather than on every packet row, so the
    regex cost is independent of the dataset size.

    The two endpoints are ordered by comparing the strings `ip:port` with the port zero-padded.
    That order only has to be deterministic and direction-symmetric — it deliberately does not
    reproduce the numeric `IpAddr` ordering the Rust side uses, because nothing here depends on
    matching it.

    Args:
        flow_keys: the `flow_key` column (duplicates fine).

    Returns:
        pl.DataFrame: columns `flow_key` and `conv_key`, one row per distinct flow key. Keys the
                      regex cannot parse keep their raw `flow_key` as `conv_key`.
    """
    uniq = flow_keys.unique().to_frame("flow_key").with_columns(
        pl.col("flow_key").str.extract(FLOW_KEY_RE, 1).alias("ip_a"),
        pl.col("flow_key").str.extract(FLOW_KEY_RE, 2).alias("port_a"),
        pl.col("flow_key").str.extract(FLOW_KEY_RE, 3).alias("ip_b"),
        pl.col("flow_key").str.extract(FLOW_KEY_RE, 4).alias("port_b"),
        pl.col("flow_key").str.extract(FLOW_KEY_RE, 5).alias("proto"),
    )

    unparsed = uniq.filter(pl.col("ip_a").is_null())
    if unparsed.height:
        examples = unparsed["flow_key"].head(3).to_list()
        print(
            f"  WARNING: {unparsed.height} of {uniq.height} distinct flow keys did not match "
            f"{FLOW_KEY_RE!r}; they keep their raw flow_key as conversation key. "
            f"Examples: {examples}"
        )

    # A handful of distinct hierarchies -> resolve the transport prefix in python, then map.
    protos = uniq["proto"].drop_nulls().unique().to_list()
    proto_map = {p: transport_prefix(p) for p in protos}

    endpoint_a = pl.col("ip_a") + ":" + pl.col("port_a").str.zfill(5)
    endpoint_b = pl.col("ip_b") + ":" + pl.col("port_b").str.zfill(5)

    return uniq.with_columns(
        pl.when(endpoint_a <= endpoint_b).then(endpoint_a).otherwise(endpoint_b).alias("ep_lo"),
        pl.when(endpoint_a <= endpoint_b).then(endpoint_b).otherwise(endpoint_a).alias("ep_hi"),
        pl.col("proto").replace_strict(proto_map, default=None).alias("transport"),
    ).with_columns(
        pl.when(pl.col("ip_a").is_null())
        .then(pl.col("flow_key"))
        .otherwise(pl.concat_str(["ep_lo", "ep_hi", "transport"], separator="|"))
        .alias("conv_key")
    ).select("flow_key", "conv_key")


# ── the split ────────────────────────────────────────────────────────

def _flow_stats(df: pl.DataFrame, ratios: tuple[float, float, float],
                long_flow_duration_s: float, min_packets_per_piece: int) -> pl.DataFrame:
    """
    One row per conversation: extent, packet count, and whether it may be cut.

    A conversation is *long* (cuttable) when it spans more than `long_flow_duration_s` **and**
    the cut it would produce leaves every piece with at least `min_packets_per_piece` packets.
    A long-lived but sparse flow that would shatter into unusable fragments is demoted to short
    and kept whole.

    The packet guard does most of the work on IoT captures, where nearly every conversation is
    long-lived: it implies a floor of `min_packets_per_piece / min(ratios)` packets before a
    flow can be cut at all (32 / 0.15 ~= 213 with the defaults). See `min_packets_per_piece` in
    `split_flows` for why that matters.
    """
    cut1 = (pl.col("n_packets") * ratios[0]).round().cast(pl.Int64)
    cut2 = (pl.col("n_packets") * (ratios[0] + ratios[1])).round().cast(pl.Int64)

    return (
        df.group_by("conv_key")
        .agg(
            pl.col("_ts").min().alias("first_ts"),
            pl.col("_ts").max().alias("last_ts"),
            pl.len().alias("n_packets"),
        )
        .with_columns(((pl.col("last_ts") - pl.col("first_ts")) / 1e6).alias("duration_s"))
        .with_columns(_cut1=cut1, _cut2=cut2)
        .with_columns(
            pl.min_horizontal(
                pl.col("_cut1"),
                pl.col("_cut2") - pl.col("_cut1"),
                pl.col("n_packets") - pl.col("_cut2"),
            ).alias("_piece_min")
        )
        .with_columns(
            (
                (pl.col("duration_s") > long_flow_duration_s)
                & (pl.col("_piece_min") >= min_packets_per_piece)
            ).alias("is_long")
        )
    )


def _assign_short_flows(stats: pl.DataFrame, n_total: int,
                        ratios: tuple[float, float, float]) -> tuple[pl.DataFrame, dict]:
    """
    Assign every whole-kept conversation to one split, chronologically by first packet.

    The long flows have already contributed packets to each split; short flows fill what is left
    of each packet quota, in the order train -> test -> val. Because the long-flow cuts use the
    same ratios, the residual quotas stay close to the global ratios.

    Returns:
        (short flows with a `_short_split` column, quota bookkeeping for the report)
    """
    long_stats = stats.filter(pl.col("is_long"))
    from_long = {
        "train": int(long_stats["_cut1"].sum()),
        "test": int((long_stats["_cut2"] - long_stats["_cut1"]).sum()),
        "val": int((long_stats["n_packets"] - long_stats["_cut2"]).sum()),
    }
    targets = {name: n_total * r for name, r in zip(SPLIT_NAMES, ratios)}
    residual = {name: max(0.0, targets[name] - from_long[name]) for name in SPLIT_NAMES}

    boundary1 = residual["train"]
    boundary2 = residual["train"] + residual["test"]

    # Packets already claimed by *earlier* short flows -> which bucket this flow starts in.
    short = (
        stats.filter(~pl.col("is_long"))
        .sort("first_ts")
        .with_columns((pl.col("n_packets").cum_sum() - pl.col("n_packets")).alias("_cum_before"))
        .with_columns(
            pl.when(pl.col("_cum_before") < boundary1)
            .then(pl.lit("train"))
            .when(pl.col("_cum_before") < boundary2)
            .then(pl.lit("test"))
            .otherwise(pl.lit("val"))
            .alias("_short_split")
        )
    )

    bookkeeping = {
        "target_packets": {k: round(v) for k, v in targets.items()},
        "packets_from_cut_flows": from_long,
        "residual_quota_for_whole_flows": {k: round(v) for k, v in residual.items()},
    }
    return short, bookkeeping


def _check_no_leakage(df: pl.DataFrame, cut_keys: set[str]) -> None:
    """
    Two invariants, raised (not warned) before anything is written:

    (a) a conversation may only appear in more than one split if it was deliberately cut;
    (b) within a cut conversation, split order along the time axis is train -> test -> val,
        i.e. the per-split time ranges do not interleave.
    """
    multi = (
        df.group_by("conv_key")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)["conv_key"]
        .to_list()
    )
    stray = set(multi) - cut_keys
    if stray:
        raise AssertionError(
            f"{len(stray)} conversation(s) span several splits without having been cut, "
            f"e.g. {sorted(stray)[:3]}"
        )

    if not cut_keys:
        return

    split_order = (
        pl.when(pl.col("split") == "train").then(0)
        .when(pl.col("split") == "test").then(1)
        .otherwise(2)
    )
    out_of_order = (
        df.filter(pl.col("conv_key").is_in(list(cut_keys)))
        .sort(["conv_key", "_ts"])
        .with_columns(split_order.alias("_order"))
        .select((pl.col("_order").diff().over("conv_key") < 0).any())
        .item()
    )
    if out_of_order:
        raise AssertionError(
            "a cut conversation has interleaved splits along the time axis "
            "(expected all train packets, then all test, then all val)"
        )


def split_flows(
    data_file: str | Path,
    output_dir: str | Path,
    train_size: float = 0.70,
    test_size: float = 0.15,
    val_size: float = 0.15,
    long_flow_duration_s: float = 600.0,
    min_packets_per_piece: int = 32,
    dry_run: bool = False,
) -> dict[str, pl.DataFrame]:
    """
    Split a packet parquet into train/test/val at flow granularity.

    Args:
        data_file:             Input packet parquet (schema: proto_hierarchy, flow_key,
                               timestamp_s, timestamp_us, data, mask, header_len).
        output_dir:            Directory for train/test/val.parquet + split_report.json.
        train_size:            Packet fraction for training   (default 0.70).
        test_size:             Packet fraction for test       (default 0.15).
        val_size:              Packet fraction for validation (default 0.15).
        long_flow_duration_s:  Flows spanning longer than this may be cut across splits.
        min_packets_per_piece: A cut is only performed if every piece keeps at least this many
                               packets; otherwise the flow is kept whole. Deliberately
                               independent of PACKETS_PER_SEQUENCE in the sequence-level
                               training script — it is about not producing degenerate
                               fragments, not about the model's window size.
                               This is the knob that separates "short but long-lived" from
                               "unwieldy" on IoT captures, and it is sharp: on
                               NormalMerged.parquet (9.7M packets, median conversation 69
                               packets over 1565 s) a value of 8 cuts 98810 of 129073
                               conversations, while 32 cuts 28 of them — the handful of
                               static-port device channels, up to 264k packets each, holding
                               4.4% of all packets. Re-check with a dry run on new data.
        dry_run:               Compute and print the report, write nothing.

    Returns:
        Dict with keys "train", "test", "val" mapping to the split DataFrames (empty dict on a
        dry run).
    """
    total = train_size + test_size + val_size
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {total:.4f} "
            f"(train={train_size}, test={test_size}, val={val_size})"
        )
    ratios = (train_size, test_size, val_size)

    data_file = Path(data_file)
    df = pl.read_parquet(data_file)
    original_columns = df.columns
    required = {"flow_key", "timestamp_s", "timestamp_us"}
    missing = required - set(original_columns)
    if missing:
        raise ValueError(f"{data_file.name} is missing required column(s): {sorted(missing)}")
    n_total = df.height
    print(f"Loaded {data_file.name}: {n_total} rows, {len(original_columns)} columns")

    # --- flow identity + a single integer time axis ---
    print("Deriving conversation keys")
    key_map = conversation_key_map(df["flow_key"])
    df = df.join(key_map, on="flow_key", how="left").with_columns(
        (pl.col("timestamp_s") * 1_000_000 + pl.col("timestamp_us")).alias("_ts")
    )
    print(
        f"  {df['flow_key'].n_unique()} raw flow keys -> {df['conv_key'].n_unique()} conversations"
    )

    # --- who gets cut, who stays whole ---
    stats = _flow_stats(df, ratios, long_flow_duration_s, min_packets_per_piece)
    n_long = int(stats["is_long"].sum())
    cut_keys = set(stats.filter(pl.col("is_long"))["conv_key"].to_list())
    print(
        f"  {n_long} conversation(s) exceed {long_flow_duration_s}s and are cut across splits; "
        f"{stats.height - n_long} kept whole"
    )

    short, quota_report = _assign_short_flows(stats, n_total, ratios)

    # --- per-packet assignment ---
    df = (
        df.join(stats.filter(pl.col("is_long")).select("conv_key", "_cut1", "_cut2"),
                on="conv_key", how="left")
        .join(short.select("conv_key", "_short_split"), on="conv_key", how="left")
        .sort(["conv_key", "_ts"])
        .with_columns(pl.int_range(pl.len()).over("conv_key").alias("_rank"))
        .with_columns(
            pl.when(pl.col("_cut1").is_null())
            .then(pl.col("_short_split"))
            .when(pl.col("_rank") < pl.col("_cut1"))
            .then(pl.lit("train"))
            .when(pl.col("_rank") < pl.col("_cut2"))
            .then(pl.lit("test"))
            .otherwise(pl.lit("val"))
            .alias("split")
        )
    )

    _check_no_leakage(df, cut_keys)

    # --- report ---
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    report = {
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "ratios": {"train": train_size, "test": test_size, "val": val_size},
        "long_flow_duration_s": long_flow_duration_s,
        "min_packets_per_piece": min_packets_per_piece,
        "total_packets": n_total,
        "raw_flow_keys": df["flow_key"].n_unique(),
        "conversations": stats.height,
        "conversations_cut": n_long,
        "conversations_whole": stats.height - n_long,
        "packets_in_cut_conversations": int(stats.filter(pl.col("is_long"))["n_packets"].sum()),
        "conversation_duration_s_quantiles": {
            str(q): float(stats["duration_s"].quantile(q)) for q in quantiles
        },
        "conversation_packets_quantiles": {
            str(q): float(stats["n_packets"].quantile(q)) for q in quantiles
        },
        **quota_report,
        "splits": {},
    }
    for name in SPLIT_NAMES:
        part = df.filter(pl.col("split") == name)
        report["splits"][name] = {
            "packets": part.height,
            "achieved_ratio": part.height / n_total if n_total else 0.0,
            "conversations": part["conv_key"].n_unique(),
            "raw_flow_keys": part["flow_key"].n_unique(),
            "first_ts_us": int(part["_ts"].min()) if part.height else None,
            "last_ts_us": int(part["_ts"].max()) if part.height else None,
        }

    print(json.dumps(report, indent=2))

    if dry_run:
        print("Dry run: nothing written.")
        return {}

    # --- write ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, pl.DataFrame] = {}
    for name in SPLIT_NAMES:
        part = df.filter(pl.col("split") == name).sort("_ts").select(original_columns)
        out_path = output_dir / f"{name}.parquet"
        part.write_parquet(out_path)
        splits[name] = part
        print(f"  {name:>5}: {part.height:>9} rows  ->  {out_path}")

    with open(output_dir / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report -> {output_dir / 'split_report.json'}")

    return splits


# ── Configure and run ────────────────────────────────────────────────
if __name__ == "__main__":

    # Normal-only capture — see the SCOPE note in the module docstring.
    DATA_FILE = "data_artefacts/IIoTset-Ferrag/NormalMerged.parquet"
    OUTPUT_DIR = "data_artefacts/IIoTset-Ferrag/flow_split"
    TRAIN_SIZE = 0.70
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    LONG_FLOW_DURATION_S = 600.0   # 10 min; check the quantiles in a dry run before trusting it
    MIN_PACKETS_PER_PIECE = 32     # ~213 packets before a flow is cut at all -- see docstring
    DRY_RUN = False                # set True to print the report without writing

    split_flows(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        long_flow_duration_s=LONG_FLOW_DURATION_S,
        min_packets_per_piece=MIN_PACKETS_PER_PIECE,
        dry_run=DRY_RUN,
    )
