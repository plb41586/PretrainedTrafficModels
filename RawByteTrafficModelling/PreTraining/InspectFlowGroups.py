"""
Group-size profile for the flow-context keys the structure fine-tune will use.

Read-only. It loads `flow_offsets.parquet` from each latent cache and, for the
application-protocol keying only, the `flow_key` / `proto_hierarchy` columns of the
split parquet -- never the packet bytes.

Run this before SequenceAEStructureFinetune.py. It decides COARSE_KEY and FINE_KEY,
and it catches the failure modes that would otherwise be invisible in the training
logs: a keying that produces too few groups (a contrastive term over a handful of
classes flattens `z` onto a low-rank simplex), a keying dominated by one group (no
negatives left in the batch), and -- the one this capture actually has -- a fine
keying that does not partition its parent at all.

What to read in the output:

  groups              too few and the coarse term degenerates into an n-way
                      classification, which is the shape that collapses `z`.
  windows p50 / max   the max is the mega-group check.
  >=8 flows           groups smaller than this contribute few or no in-batch
                      positives, so this is the share of the data the contrastive
                      term can actually act on.
  sub-groups          for a fine keying: how many pieces it cuts each coarse group
                      into. A fine level that averages ~1 is not a hierarchy level,
                      it is the coarse level again.
  test coverage       share of test flows whose group was also seen in train. A group
                      with no training signal says nothing about generalisation.

Run from the repo root:
    python -m RawByteTrafficModelling.PreTraining.InspectFlowGroups
"""
from RawByteTrafficModelling.ModelComponents.DataUtils import (
    flow_group_labels,
    parse_flow_key,
)
from RawByteTrafficModelling.PreTraining.RunConfig import DATASETS
import polars as pl
import numpy as np
import os

DATASET = DATASETS["IIoTset-Ferrag"]
CACHE_TAG = "PacketAE_d128_best"        # must match CachePacketLatents
SPLITS = ("train", "test")
SPLIT_FILES = {"train": DATASET.train, "test": DATASET.test}
SEQ_LEN = 64                            # PACKETS_PER_SEQUENCE - 1, real packets per window

# The scalar keyings flow_group_labels supports, plus "any_host" -- profiled here but
# deliberately not a training option: a flow belongs to both of its hosts, so it has no
# single label a contrastive loss could use. It is in the table as the sharpest test
# for a broker/gateway mega-group.
COARSE_CANDIDATES = ("endpoint_pair", "host_lo", "host_hi")
MIN_FLOWS_FOR_POSITIVES = 8
TOP_GROUPS_SHOWN = 5


def windows_per_flow(lengths: np.ndarray) -> np.ndarray:
    """Matches CachedLatentSequenceHandler.enumerate_windows: floor(L/SEQ_LEN), min 1."""
    return np.maximum(1, lengths // SEQ_LEN)


def app_protocol_per_flow(split_file: str) -> dict:
    """flow_key -> richest proto_hierarchy observed on that flow.

    "Richest" (most hierarchy tokens) rather than modal, mirroring what the Rust side
    stores as FlowStats::proto_hierarchy: a flow that carries even one MQTT packet is
    an MQTT flow, and the TCP handshake packets that outnumber it are not what
    identifies it. Counts break ties.

    Aggregated lazily on (flow_key, proto_hierarchy) so the 6.8M-row split never
    materialises -- the grouped frame is one row per flow per distinct hierarchy.
    """
    counts = (pl.scan_parquet(split_file)
              .select(["flow_key", "proto_hierarchy"])
              .group_by(["flow_key", "proto_hierarchy"])
              .agg(pl.len().alias("n"))
              .collect())
    counts = counts.with_columns(
        pl.col("proto_hierarchy").str.count_matches("->").alias("depth"))
    richest = (counts.sort(["depth", "n"], descending=True)
                     .group_by("flow_key", maintain_order=True)
                     .first())
    return dict(zip(richest["flow_key"].to_list(),
                    richest["proto_hierarchy"].to_list()))


def profile(name: str, labels: list[str], windows: np.ndarray,
            parents: list[str] = None) -> dict:
    """Per-group flow and window counts, and how finely it cuts its parent keying."""
    frame = pl.DataFrame({"label": labels, "windows": windows}).group_by("label").agg(
        pl.len().alias("flows"), pl.col("windows").sum().alias("windows"))
    flows = frame["flows"].to_numpy()
    win = frame["windows"].to_numpy()

    sub_per_parent = float("nan")
    if parents is not None:
        # Weighted by windows: what matters is how finely the *bulk* of the data is
        # cut, not how many pieces a singleton group splits into.
        pairs = pl.DataFrame({"parent": parents, "label": labels, "windows": windows})
        per_parent = pairs.group_by("parent").agg(
            pl.col("label").n_unique().alias("subs"),
            pl.col("windows").sum().alias("windows"))
        sub_per_parent = float(
            (per_parent["subs"] * per_parent["windows"]).sum() / per_parent["windows"].sum())

    return {
        "name": name,
        "groups": frame.height,
        "flows_p50": int(np.median(flows)),
        "flows_max": int(flows.max()),
        "windows_p50": int(np.median(win)),
        "windows_p90": int(np.quantile(win, 0.9)),
        "windows_max": int(win.max()),
        "largest_share": float(win.max() / win.sum()),
        "usable_groups": int((flows >= MIN_FLOWS_FOR_POSITIVES).sum()),
        "usable_window_share": float(win[flows >= MIN_FLOWS_FOR_POSITIVES].sum() / win.sum()),
        "sub_per_parent": sub_per_parent,
    }


def print_table(title: str, rows: list[dict], show_subs: bool = False):
    print(f"\n{title}")
    header = (f"  {'keying':<30} {'groups':>7} {'flows/grp':>12} {'windows/grp':>22} "
              f"{'largest':>9} {'>=' + str(MIN_FLOWS_FOR_POSITIVES) + ' flows':>16}")
    sub_header = (f"  {'':<30} {'':>7} {'p50':>5} {'max':>6} {'p50':>6} {'p90':>7} {'max':>7} "
                  f"{'share':>9} {'groups':>7} {'wins':>8}")
    if show_subs:
        header += f" {'subs/parent':>12}"
    print(header)
    print(sub_header)
    for r in rows:
        line = (f"  {r['name']:<30} {r['groups']:>7} {r['flows_p50']:>5} {r['flows_max']:>6} "
                f"{r['windows_p50']:>6} {r['windows_p90']:>7} {r['windows_max']:>7} "
                f"{r['largest_share']:>8.1%} {r['usable_groups']:>7} "
                f"{r['usable_window_share']:>7.1%}")
        if show_subs:
            line += f" {r['sub_per_parent']:>12.2f}"
        print(line)


split_labels = {}       # split -> keying name -> labels

for split in SPLITS:
    cache_dir = DATASET.latent_cache(CACHE_TAG, split)
    offsets = pl.read_parquet(os.path.join(cache_dir, "flow_offsets.parquet"))
    flow_keys = offsets["flow_key"].to_list()
    lengths = offsets["length"].to_numpy().astype(np.int64)
    windows = windows_per_flow(lengths)

    print(f"\n=== {split}: {len(flow_keys)} flows, {int(lengths.sum())} packets, "
          f"{int(windows.sum())} windows ({cache_dir})")

    parsed = [parse_flow_key(k) for k in flow_keys]
    transports = [p[4] for p in parsed]
    print("  transports: " + ", ".join(
        f"{t} {c}" for t, c in
        pl.Series(transports).value_counts(sort=True).iter_rows()))

    # Application protocol, the only axis that needs the split parquet.
    proto_map = app_protocol_per_flow(SPLIT_FILES[split])
    app_protos = [proto_map.get(k, "?") for k in flow_keys]
    proto_counts = pl.DataFrame({"proto": app_protos, "windows": windows}).group_by(
        "proto").agg(pl.len().alias("flows"), pl.col("windows").sum().alias("windows")
                     ).sort("windows", descending=True)
    print(f"  proto_hierarchy ({proto_counts.height} distinct):")
    for proto, flows, wins in proto_counts.head(12).iter_rows():
        print(f"    {proto:<48} {flows:>7} flows {wins:>8} windows")

    # Service side of the port pair: the lower port. Ephemeral ports are high, so this
    # is the listening service in every ordinary client/server flow (and 0 for ARP/ICMP).
    service_ports = [str(min(p[1], p[3])) for p in parsed]

    coarse_rows, fine_rows, keyed = [], [], {}
    for coarse in COARSE_CANDIDATES:
        coarse_labels, _fine = flow_group_labels(flow_keys, coarse=coarse)
        keyed[coarse] = coarse_labels
        coarse_rows.append(profile(coarse, coarse_labels, windows))

        for fine_name, fine_values in (("transport", transports),
                                       ("service_port", service_ports),
                                       ("app_proto", app_protos)):
            labels = [f"{c}|{v}" for c, v in zip(coarse_labels, fine_values)]
            keyed[f"{coarse} x {fine_name}"] = labels
            fine_rows.append(profile(f"{coarse} x {fine_name}", labels, windows,
                                     parents=coarse_labels))

    # any_host: a flow counts towards both of its endpoints, so its rows are duplicated.
    # Not usable as a training label (no single group per flow) -- the mega-group probe.
    any_labels, any_windows = [], []
    for (ip_a, _pa, ip_b, _pb, _t), w in zip(parsed, windows):
        any_labels += [ip_a, ip_b]
        any_windows += [w, w]
    coarse_rows.append(profile("any_host (probe only)", any_labels,
                               np.array(any_windows, dtype=np.int64)))

    print_table(f"coarse keyings ({split})", coarse_rows)
    print_table(f"fine keyings ({split})", fine_rows, show_subs=True)

    # The biggest coarse groups, and how the fine keyings cut them. This is where a
    # fine level either earns its place or does not.
    top = (pl.DataFrame({"label": keyed["endpoint_pair"], "windows": windows,
                         "port": service_ports, "proto": app_protos})
           .group_by("label").agg(pl.len().alias("flows"),
                                  pl.col("windows").sum().alias("windows"),
                                  pl.col("port").n_unique().alias("ports"),
                                  pl.col("proto").n_unique().alias("protos"))
           .sort("windows", descending=True).head(TOP_GROUPS_SHOWN))
    print(f"\n  top {TOP_GROUPS_SHOWN} endpoint_pair groups by windows")
    print(f"    {'pair':<44} {'flows':>8} {'windows':>9} {'ports':>7} {'protos':>7}")
    for label, flows, wins, ports, protos in top.iter_rows():
        print(f"    {label:<44} {flows:>8} {wins:>9} {ports:>7} {protos:>7}")

    split_labels[split] = keyed

# --- Train -> test coverage -------------------------------------------------
# The structure is only evidence of anything on groups the model was trained on. A test
# group unseen in train has had no contrastive signal, so its position is whatever
# reconstruction alone put it at.
if "train" in split_labels and "test" in split_labels:
    print("\n=== train -> test group coverage")
    print(f"  {'keying':<30} {'test groups':>12} {'seen in train':>14} {'test flows seen':>17}")
    for name in split_labels["test"]:
        train_set = set(split_labels["train"][name])
        test_labels = split_labels["test"][name]
        seen = [l in train_set for l in test_labels]
        print(f"  {name:<30} {len(set(test_labels)):>12} "
              f"{sum(1 for l in set(test_labels) if l in train_set):>14} "
              f"{np.mean(seen):>16.1%}")

print("\nCOARSE_KEY: prefer many groups, a large '>=8 flows' window share, and no single "
      "group holding an outsized 'largest share'.")
print("FINE_KEY:   'subs/parent' near 1 means the keying does not partition its parent "
      "and is not a hierarchy level at all.")
