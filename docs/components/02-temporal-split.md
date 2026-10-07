# 02 · Temporal split (packet parquet → split roles)

**What it does:** cuts an **attack-free** capture into disjoint wall-clock intervals, one
per consumer role (one per role per recording session, see [§ Sessions](#sessions)), so
that every later stage is evaluated on time it has never seen.

**Entry point:** `data_tools/SplitFlowsDF.py` (set the constants in its `__main__` block).

```
python -m data_tools.SplitFlowsDF
```

**Consumes:** the normal packet parquet (`DatasetPaths.normal`).
**Produces:** `flow_split/<role>.parquet` (the input schema, plus `session` for a session split) and `split_report.json`
([artefacts](../reference/artefacts.md#split-report)).

## Where the logic lives

| Concern | Symbol in `data_tools/SplitFlowsDF.py` |
|---|---|
| The whole split | `split_flows(data_file, output_dir, splits, dry_run, boundary_mode, session_gap_s, min_session_share, pause_cap_s)` |
| Boundary placement (packet, time or active-time quantiles) | `_boundary_indices`, `_active_clock`, `BOUNDARY_MODES` |
| Recording sessions (pauses longer than `session_gap_s`) | `_sessions` |
| Disjoint, ordered, non-empty assertion | `_check_intervals` |
| `flow_key` regex, shared with `DataUtils.parse_flow_key` | `FLOW_KEY_RE` |

## How it works

1. Sort all packets by `timestamp_s * 1e6 + timestamp_us`.
2. Place each boundary according to `boundary_mode`, which sets what the fractions are
   shares of:
   - `packets` (the default): at a **packet quantile**. The boundary for cumulative share
     *q* is the timestamp of packet `round(q·N)`, so the ratios hold in packet counts.
   - `time`: at a **time quantile**. The boundary is the instant
     `t_first + q·(t_last − t_first)`, so each split gets its share of the clock and its
     packet count follows the traffic rate.
   - `active_time`: as `time`, but on an **active clock** where every pause between two
     packets counts as at most `pause_cap_s`. Each split gets its share of the time traffic
     was actually recorded, and a pause inside a recording cannot swallow a whole split.

   Every mode keeps each split a clean time interval. The modes differ where the traffic
   rate varies or the recording pauses. None of them helps when the capture is several
   recording sessions with long pauses between them, because every mode would hand whole
   sessions to roles. Use [§ Sessions](#sessions) for that case.
3. Move each boundary forward to the first packet at or after it, so packets with the
   same timestamp are never separated. A time window that contains no packets raises
   an error instead of producing an empty split.
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

## Sessions

**The goal is one contiguous time interval per split.** Follow it whenever the capture
is one continuous recording. A strict split's `val` is then a genuinely later period,
which is the most honest test.

**Exception: a capture made of separate recording sessions** that were run one after
another, possibly with different settings or simulated environments. A strict split
then hands whole sessions to roles: the AEs train on the early sessions and every AD
split comes from the last one. The evaluation measures the jump between environments
instead of held-out time. That is what happened on IIoTset
([experiment diary](../experiments/2026-09-25-iiotset-six-split-first-run.md)). For such a
capture, set `session_gap_s`:

1. A new session starts wherever two consecutive packets are more than `session_gap_s`
   apart. This is a cut on the clock only; no flow is looked at.
2. A session holding less than `min_session_share` of all packets is left out of every
   split and marked as dropped in `split_report.json`. This is the one sanctioned
   exclusion: it is decided for the whole session by its size, never for individual flows.
3. Every remaining session is cut into the roles with the same fractions, on its **own**
   clock (`t_first`/`t_last` are the session's). Within a session, every boundary is a
   clean time boundary and the pieces are disjoint and in role order.
4. Each role's parquet is the union of its piece of every session. A `session` column is
   written so the flow index treats (`flow_key`, `session`) as the flow
   ([data-representation § Flow index](../reference/data-representation.md#flow-index)):
   flows break at sessions, and no window spans the pause between recordings.

What this changes in meaning: across sessions, a role's later piece can come after
another role's earlier piece. For example, `train` contains a session recorded after
another session's `val`. No packet is shared between splits, but `val` measures held-out
time **within** sessions, not generalisation to an unseen session. Say so when reporting
results from a session split.

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

The splits run in the six-role order, which is the `DatasetPaths.splits` default
(`RunConfig.SIX_ROLE_SPLITS`). A capture may leave out `late`.

| Capture | Splits | Produced by |
|---|---|---|
| `CICAPT-IIoT` | all six | the current `split_flows` |
| `IIoTset-Ferrag` | five, without `late` | a session split ([§ Sessions](#sessions)). The capture is four recording sessions (days 0–2, 28, 36 and 44 of a 45-day span); day 28 (0.07% of packets) is dropped, and each other session is cut by quantiles of its own active time (`boundary_mode="active_time"`, pauses capped at 60 s), because the sessions have internal pauses of hours. Results: [experiment diary](../experiments/2026-10-07-iiotset-session-split.md). The first, strict six-split run took every AD split from the day-44 session ([experiment diary](../experiments/2026-09-25-iiotset-six-split-first-run.md)). Earlier generations are in `archive/` ([artefacts § Directory layout](../reference/artefacts.md#directory-layout)) |

## Scope

The splitter only applies to captures that contain no attacks. A split is a time period,
so an attack inside a split capture would fall entirely within whichever split covers
its time window. Attack captures are therefore kept whole under `attacks/` and used only
as evaluation positives.

## Reuse

The splitter knows nothing about the models. Any task that needs a temporal hold-out with
no leakage can call `split_flows` with its own role names and fractions.
