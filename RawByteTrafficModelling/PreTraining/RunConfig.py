"""
Shared configuration and run scaffolding for the pretraining scripts.

The scripts in this package are hardcoded-config scripts by design -- their
hyperparameters stay as constants at the top of each file. What lives here is the
part that is *not* per-experiment and was being copy-pasted: where the data is,
what the special token IDs are, how a run directory and its logger are set up,
and the small helpers (LR schedule, metrics CSV, fixed eval subset, curves) that
SequenceLevelAutoEncoder.py worked out and the packet-level scripts should not
reinvent.

One place to repoint when a capture is re-extracted: DATA_ROOT / DATASETS.
"""
from RawByteTrafficModelling.ModelComponents.ModelDefinitions import EncoderParams
from RawByteTrafficModelling.ModelComponents.BackBones import (
    MambaBackboneParams,
    TransformerBackboneParams,
)
from RawByteTrafficModelling.ModelComponents.DataUtils import ID_Encoder
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import math
import csv
import os
from pathlib import Path

# --- Token / packet conventions --------------------------------------------
# Defined once here rather than ad hoc per script: CachePacketLatents and the
# byte-level eval in SequenceLevelAutoEncoder assert that the encoder they load
# was built with exactly these, so a drifting copy is a silent misalignment.
SPECIAL_IDS = {"<pad>": 256, "</s>": 257, "<CLS>": 258,
               "<mask>": 259, "<EndPointMasking>": 260, "<BOS>": 261}
VOCAB_SIZE = 262          # 256 byte values + the six specials above
PACKET_ID_LEN = 1520      # ID_Encoder hardcodes this length too
CLS_PLACEMENT = "EOS"     # load-bearing: the latent cache was built this way

TRAINING_OUTPUT_ROOT = "RawByteTrafficModelling/PreTraining/TrainingOutputs"


# --- Data artefacts ---------------------------------------------------------
DATA_ROOT = "data_artefacts/merged_extractor"

# The one split layout, in time order, one consumer per split (docs/components/
# 02-temporal-split.md § Split roles): train pretrains the AEs, test picks their
# checkpoint, ad_fit fits the detectors, ad_calib turns scores into thresholds, val is
# the reported false-positive rate, late is the drift floor.
SIX_ROLE_SPLITS = ("train", "test", "ad_fit", "ad_calib", "val", "late")


@dataclass(frozen=True)
class DatasetPaths:
    """Where one capture's artefacts live, relative to the workspace root.

    Only the `flow_split/` partition is exposed. The old packet-level `split/`
    partition (contiguous row slices) is gone along with the splitter that
    produced it.

    `SplitFlowsDF` is a strict temporal splitter, so a split is a contiguous
    wall-clock interval and a flow crossing a boundary appears on both sides.
    Do not assume a flow lives in exactly one split -- an earlier version of
    this docstring claimed that, and it is false by design (see docs/components/02-temporal-split.md).

    `splits` is the authority, and `split()` validates against it, so a name that
    does not exist fails here rather than as an unreadable-parquet error later.

    Args:
        name:        directory under DATA_ROOT.
        source:      the attack-free capture parquet the split was cut from.
        splits:      every split name present in `flow_split/`.
        has_attacks: whether an `attacks/` directory exists. `attacks` is None
                     when it does not, so callers branch on it rather than on a
                     path that silently does not resolve.
    """
    name: str
    source: str = "NormalMerged.parquet"
    splits: tuple[str, ...] = SIX_ROLE_SPLITS
    has_attacks: bool = True

    @property
    def root(self) -> str:
        return f"{DATA_ROOT}/{self.name}"

    @property
    def normal(self) -> str:
        """The attack-free source capture the split was cut from."""
        return f"{self.root}/{self.source}"

    @property
    def attacks(self) -> str | None:
        return f"{self.root}/attacks" if self.has_attacks else None

    def attack_names(self) -> tuple[str, ...]:
        """Sorted stems of `attacks/*.parquet`; the stem is the attack class. Empty when
        the capture has no attacks, so a role mapped to ALL_ATTACKS resolves to nothing
        instead of listing a directory that does not exist."""
        if not self.has_attacks:
            return ()
        return tuple(sorted(p.stem for p in Path(self.attacks).glob("*.parquet")))

    def attack(self, name: str) -> str:
        """Path to one attack parquet, checked against `attack_names()`."""
        if name not in self.attack_names():
            raise KeyError(f"{self.name} has no attack capture {name!r}; "
                           f"it has {self.attack_names()}")
        return f"{self.attacks}/{name}.parquet"

    def sections(self) -> dict[str, str]:
        """Every readable section of this capture -> its kind, for `DataRoles`."""
        return ({s: NORMAL for s in self.splits}
                | {a: ATTACK for a in self.attack_names()})

    def split(self, which: str) -> str:
        """Path to one split parquet, checked against `splits`."""
        if which not in self.splits:
            raise KeyError(f"{self.name} has no split {which!r}; it has {self.splits}")
        return f"{self.split_dir}/{which}.parquet"

    @property
    def split_dir(self) -> str:
        return f"{self.root}/flow_split"

    # The three original roles keep their properties so the existing scripts read
    # unchanged; anything else goes through split().
    @property
    def train(self) -> str:
        return self.split("train")

    @property
    def test(self) -> str:
        """Monitored during training."""
        return self.split("test")

    @property
    def val(self) -> str:
        """Final held-out set -- not read by the pretraining scripts."""
        return self.split("val")

    @property
    def split_report(self) -> str:
        return f"{self.split_dir}/split_report.json"

    def latent_cache(self, tag: str, split: str) -> str:
        return f"{self.split_dir}/latents_{tag}/{split}"


DATASETS = {
    # No `late` for now: its share went to the AD splits, to test whether their narrow
    # time span (and the shorter windows it produces) drove the first six-split AD result
    # (docs/experiments/2026-09-25-iiotset-six-split-first-run.md).
    "IIoTset-Ferrag": DatasetPaths(
        "IIoTset-Ferrag",
        splits=("train", "test", "ad_fit", "ad_calib", "val"),
    ),
    "CICAPT-IIoT": DatasetPaths(
        "CICAPT-IIoT",
        source="CICAPT_Phase1.parquet",
        has_attacks=False,
    ),
}


# --- Data roles -------------------------------------------------------------
# Section kinds. A section is anything a script can read as one unit: a split of the
# normal capture, or one attack capture (stem = attack class).
NORMAL = "normal"
ATTACK = "attack"
# Expands to every attack section available, so a role can take all attacks without
# listing them (the list differs per capture and is empty for CICAPT).
ALL_ATTACKS = "<all attacks>"


class DataRoles:
    """The one place a script says which data it reads, and a log of every read.

    A script declares `DATA_ROLES = {script_role: (section, ...)}` near its top. This
    class checks the sections against what actually exists (`available`: section ->
    NORMAL | ATTACK), logs the resolved mapping once as a `DATA ROLES` block (including
    which available sections are left unread), and then logs one `DATA ACCESS` line per
    `touch()` -- which the scripts call at every disk read of a section. The run log is
    then a complete record of what data a run saw.

    Roles are disjoint: a section mapped to two roles raises, unless that exact pair of
    roles is listed in `allow_shared` (the embedding export reads `test` both as a cached
    set and for its cross-path check).
    """

    def __init__(self, role_map: dict[str, tuple[str, ...]], available: dict[str, str],
                 logger: logging.Logger,
                 allow_shared: tuple[tuple[str, str], ...] = ()):
        self.available = dict(available)
        self.logger = logger
        self.roles: dict[str, tuple[str, ...]] = {}
        for role, sections in role_map.items():
            if isinstance(sections, str):
                raise TypeError(f"role {role!r} maps to the string {sections!r}; "
                                f"use a tuple, e.g. ({sections!r},)")
            resolved = []
            for s in sections:
                if s == ALL_ATTACKS:
                    resolved += [a for a, kind in self.available.items() if kind == ATTACK]
                elif s not in self.available:
                    raise KeyError(f"role {role!r} maps to unknown section {s!r}; "
                                   f"available: {sorted(self.available)}")
                else:
                    resolved.append(s)
            self.roles[role] = tuple(dict.fromkeys(resolved))   # dedupe, keep order

        shared = {frozenset(p) for p in allow_shared}
        names = list(self.roles)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = set(self.roles[a]) & set(self.roles[b])
                if both and frozenset((a, b)) not in shared:
                    raise ValueError(f"roles {a!r} and {b!r} both map {sorted(both)}; "
                                     f"roles must not share a section")

        width = max(len(r) for r in self.roles) if self.roles else 0
        lines = [f"  {r:<{width}} -> {', '.join(s) if s else '(none)'}"
                 for r, s in self.roles.items()]
        mapped = {s for secs in self.roles.values() for s in secs}
        unread = [s for s in self.available if s not in mapped]
        lines.append(f"  not read this run: {', '.join(unread) if unread else '(none)'}")
        self.logger.info("DATA ROLES\n" + "\n".join(lines))

    def sections(self, role: str) -> tuple[str, ...]:
        return self.roles[role]

    def kind(self, section: str) -> str:
        return self.available[section]

    def touch(self, role: str, section: str, path: str, rows: int | None = None):
        """Log one read of `section` in `role`. Call it at the read, not ahead of it."""
        if section not in self.roles.get(role, ()):
            raise KeyError(f"reading section {section!r} as role {role!r}, which the "
                           f"DATA_ROLES map does not allow")
        rows_part = f" rows={rows}" if rows is not None else ""
        self.logger.info(f"DATA ACCESS role={role} section={section}{rows_part} path={path}")


# --- Run scaffolding --------------------------------------------------------
def setup_run(run_name: str, log_filename: str) -> tuple[str, logging.Logger]:
    """Create the run's output dir and wire logging to file + stdout.

    The scripts do not mkdir their own output dir, and a logging.FileHandler on a
    missing directory raises before anything else runs -- so this does both.
    """
    output_dir = f"{TRAINING_OUTPUT_ROOT}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/{log_filename}'),
            logging.StreamHandler()
        ]
    )
    return output_dir, logging.getLogger(run_name)


def resolve_device(index: int = 0) -> torch.device:
    """Pick one GPU by index.

    Two identical GPUs are attached to the container, and the packet-level AE and
    MLM runs are independent, so they take one device each and run concurrently
    rather than queueing behind each other.
    """
    assert torch.cuda.is_available(), "no CUDA device -- these scripts are GPU-only"
    count = torch.cuda.device_count()
    assert index < count, f"asked for cuda:{index} but only {count} device(s) visible"
    return torch.device(f"cuda:{index}")


def make_id_encoder() -> ID_Encoder:
    """The one encoder configuration every level of the hierarchy assumes."""
    return ID_Encoder(SpecialIDs=SPECIAL_IDS, CLS_Placement=CLS_PLACEMENT)


def packet_encoder_params(dim: int, num_layers: int = 2,
                          backbone: str = "Mamba", **backbone_kwargs) -> EncoderParams:
    """Build the packet-level EncoderParams.

    Both packet-level scripts go through here, so the AE's encoder and the MLM's
    encoder are structurally identical and either can be dropped into the
    sequence level as its packet "embedding layer".
    """
    if backbone == "Mamba":
        backbone_params = MambaBackboneParams(dim=dim, num_layers=num_layers, **backbone_kwargs)
    elif backbone == "Transformer":
        backbone_params = TransformerBackboneParams(dim=dim, num_layers=num_layers,
                                                    max_len=PACKET_ID_LEN, **backbone_kwargs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return EncoderParams(
        vocab_size=VOCAB_SIZE,
        EncoderDim=dim,
        packet_id_len=PACKET_ID_LEN,
        pooling_type="DynamicCLS",
        BackboneType=backbone,
        BackboneParams=backbone_params,
        CLS_ID=SPECIAL_IDS["<CLS>"],
        SpecialTokens=SPECIAL_IDS,
    )


# --- Training helpers -------------------------------------------------------
def cosine_warmup_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to zero, for LambdaLR."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return lr_lambda


def fixed_eval_batches(height: int, n_batches: int, batch_size: int,
                       seed: int) -> list[np.ndarray]:
    """Row-index batches for a held-out subset, drawn once.

    A full pass over a 1.46M-packet test split per eval is far too expensive, and
    a fresh sample each time makes the curve a scatter plot. Same rows every eval.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(height, size=min(n_batches * batch_size, height), replace=False)
    return [rows[i:i + batch_size] for i in range(0, rows.shape[0], batch_size)]


class MetricsCsv:
    """Append-only metrics CSV with a guard against stale columns."""

    def __init__(self, path: str, fields: list[str], logger: logging.Logger = None):
        self.path = path
        self.fields = fields
        # A metrics.csv from an earlier run with different columns would be
        # appended to with rows that no longer line up with its header, so retire
        # it instead of corrupting it.
        if os.path.exists(path):
            with open(path, newline="") as f:
                stale_header = next(csv.reader(f), [])
            if stale_header != fields:
                backup_path = f"{path}.{len(stale_header)}col.bak"
                os.rename(path, backup_path)
                if logger:
                    logger.warning(f"metrics.csv had stale columns, moved to {backup_path}")
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def append(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields).writerow(row)

    def read(self) -> list[dict]:
        with open(self.path, newline="") as f:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def plot_curves(metrics: MetricsCsv, out_png: str, x_field: str,
                panels: list[dict], logger: logging.Logger = None):
    """Render one row of panels from the metrics CSV.

    Reads the CSV back rather than an in-memory history, so a resumed run still
    plots the epochs that ran before the interruption.

    Each panel is a dict:
        {"title": str, "ylabel": str,
         "series": [(field, label), ...],
         "hlines": [(field, label, linestyle), ...],   # optional, last row's value
         "ylim": (lo, hi)}                             # optional
    """
    history = metrics.read()
    if not history:
        if logger:
            logger.warning(f"{metrics.path} is empty, no curves to plot")
        return

    x = [r[x_field] for r in history]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        for field, label in panel["series"]:
            ax.plot(x, [r[field] for r in history], label=label)
        for field, label, style in panel.get("hlines", []):
            ax.axhline(history[-1][field], ls=style, c="grey", label=label)
        ax.set_xlabel(x_field)
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        if "ylim" in panel:
            ax.set_ylim(*panel["ylim"])
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if logger:
        logger.info(f"Saved curves to {out_png}")
