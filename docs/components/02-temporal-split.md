# 02 · Temporal split (packet parquet → split roles)

**What it does:** cuts an **attack-free** capture into contiguous, disjoint wall-clock
intervals, one per consumer role, so that every later stage is evaluated on time it has
never seen.

**Entry point:** `data_tools/SplitFlowsDF.py` (set the constants in its `__main__` block).

```
python -m data_tools.SplitFlowsDF
```

**Consumes:** the normal packet parquet (`DatasetPaths.normal`).
**Produces:** `flow_split/<role>.parquet` (same schema as the input) and `split_report.json`
([artefacts](../reference/artefacts.md#split-report)).

## Where the logic lives

| Concern | Symbol in `data_tools/SplitFlowsDF.py` |
|---|---|
| The whole split | `split_flows(data_file, output_dir, splits, dry_run)` |
| Boundary placement | `_boundary_indices` |
| Disjoint, ordered, non-empty assertion | `_check_intervals` |
| `flow_key` regex, shared with `DataUtils.parse_flow_key` | `FLOW_KEY_RE` |

## How it works

1. Sort all packets by `timestamp_s * 1e6 + timestamp_us`.
2. Place each boundary at a **packet quantile**: the boundary for cumulative share *q* is
   the timestamp of packet `round(q·N)`. The ratios are therefore met in packet counts,
   while each split remains a clean time interval.
3. Move each boundary forward to the first packet of the next distinct timestamp, so
   packets with the same timestamp are never separated.
4. Cut **every** flow at every boundary it crosses. A long-lived flow therefore appears in
   several splits. This is intended, and `split_report.json` reports the count as
   `flow_keys_crossing_a_boundary`.

**No flow is selected or excluded by length or duration, and no flow is kept whole.** An
earlier splitter cut only flows that passed a `long_flow_duration_s` /
`min_packets_per_piece` gate, kept the rest whole, and ordered them by first packet. On
CICAPT-IIoT Phase 1 the median conversation lasts 61% of the capture, so under that rule
every split covered all four days and nothing was held out in time. Cutting every flow
at the boundaries, the same way, is the only approach that works for this kind of
traffic.

**This is the only splitter.** An older `SplitDataDF` took contiguous row slices of the
packet table without regard to flows. It was deleted because it leaked: a single flow
could be split between train, test and val. A packet-level split would bring that leak
back.

## Split roles

`SPLITS` is an ordered tuple of `(name, fraction)` pairs, listed in time order. The
six-role layout gives each consumer its own split, so no split is used for two purposes:

| Role (time order) | Consumer |
|---|---|
| `train` | packet and sequence AE training |
| `test` | monitored during AE training; selects `_best` checkpoints |
| `ad_fit` | fitting the AD detectors |
| `ad_calib` | calibrating thresholds from score quantiles, without labels |
| `val` | final held-out normals, i.e. the reported false-positive rate |
| `late` | the tail after `val`, used to measure temporal drift |

The rest of the pipeline expects the names `train` and `test`. Other names are free-form,
but they must also be listed in the capture's `DatasetPaths.splits`.

## Current state

| Capture | Layout | Produced by |
|---|---|---|
| `CICAPT-IIoT` | six roles, strict temporal | the current `split_flows` |
| `IIoTset-Ferrag` | `train`/`test`/`val` | the **earlier flow-selection splitter**, so each split spans the whole capture: it holds out a population of flows, not a period of time. Re-splitting it with the current splitter would invalidate its latent cache and every model trained on it |

## Scope

The splitter only applies to captures that contain no attacks. A split is a time period,
so an attack inside a split capture would fall entirely within whichever split covers
its time window. Attack captures are therefore kept whole under `attacks/` and used only
as evaluation positives.

## Reuse

The splitter knows nothing about the models. Any task that needs a temporal hold-out with
no leakage can call `split_flows` with its own role names and fractions.
