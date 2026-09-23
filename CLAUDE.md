# CLAUDE.md

This file gives Claude Code (claude.ai/code) its working rules for this repository.
It holds **rules only**. The project itself is described once, in `docs/`, and this file
does not repeat that description.

## Read the docs first

Dissertation codebase: raw-byte traffic models (packet- and flow-level autoencoders, MLM,
Mamba vs. Transformer backbones) pretrained for network anomaly detection, plus a Rust
PCAP feature extractor.

- **Start at `docs/PIPELINE.md`.** It gives the order of the stages, the entry-point
  script and inputs/outputs of each one, the checklist for adding a dataset, and the
  points where other downstream tasks can reuse a stage.
- **Before changing a stage, read its `docs/components/NN-*.md`.** The
  "Where the logic lives" table maps each concern to `file · symbol`.
- **Shared facts are in `docs/reference/`:**
  - `artefacts.md`: file formats, layout, checkpoints, lineage checks
  - `data-representation.md`: tokens and windows
  - `model-library.md`: `ModelComponents/`
  - `conventions.md`: container environment, script anatomy, `RunConfig`, split names

## Keeping the docs true

- **Every change that affects documented behaviour updates the owning doc in the same
  change.** That covers a renamed symbol, a moved file, a new or changed artefact, a new
  stage, a new constant, and a new known gap.
- **Each fact lives in exactly one doc; other docs link to it.** Do not add project
  description to this file, to `Readme.md` or to `TODO.md`. If a fact is written in two
  places, remove one copy and link to the other.
- **Reference code as `path · symbol`, never by line number.** Do not copy hyperparameter
  values or results into the docs: values stay as script constants, and results stay with
  their run outputs.
- **When the code and the docs disagree, the code is right.** Fix the doc (or ask the user)
  instead of acting on the stale text.

## Running commands (read before executing anything)

Nothing in this project runs on the host. Every Python invocation, script run, smoke test,
`cargo` build and data-tool run happens inside the `devcontainer`. A host
`python`/`python3`/`py_compile` call is always wrong. The environment details and the
canonical `docker exec -w /home/plb41586/workspace devcontainer …` form are in
`docs/reference/conventions.md § Running anything`.

**Ask before every single container command.** Do not batch them, do not treat one
approval as covering the next, and never run one "just to check something quickly". For
each command, before running it:

1. Show the **entire** command verbatim, with nothing abbreviated and no `...`.
2. Give a one- or two-sentence reason why it is the right thing to run now.
3. Wait for explicit confirmation.

Reading files, `grep`, `ls` and `git` on the host need no confirmation. The rule is about
*executing* project code.

- **Hand edits over unverified.** A syntax check is itself a container command, so expect
  edits to go back without one. Say plainly what has not been run, and give the exact
  command that would check it.
- **Prefer one meaningful run** (the real script with `SMOKE = True` or a small
  `MAX_STEPS_PER_EPOCH`) over a series of small probes.
- **Let the user launch long training runs** with the `! <command>` prefix. Propose the
  exact command instead of running it.
- **Don't invent test or lint commands.** The Python code has neither. The Rust crate has
  `cargo test`.

## Writing code here

- **Scripts are configured by constants at the top of the file.** Change the constant;
  do not add argparse.
- **New scripts** import shared config and helpers from `PreTraining/RunConfig.py`, and
  use `PacketLevelAutoEncoder.py` as the template (`docs/reference/conventions.md`).
- **Paths** come from `RunConfig.DATASETS`, never hardcoded.
- **Tokenisation** goes through `RunConfig.make_id_encoder()`, never an inline `ID_Encoder`
  (`docs/reference/data-representation.md`).
- **A new backbone** is registered in both `BACKBONES` and `unpack_backbone_params`.
- **Checkpoints** are loaded through the matching `load_*` function, never by building a
  params dataclass from `ckpt["config"]` (`docs/reference/model-library.md`).
- **Padding masks** come from `build_padding_mask`. Do not rebuild them elsewhere.
- **One run per GPU** via `DEVICE_INDEX`, no DDP. Pin TensorFlow to the CPU before any
  `keras_hub` import (`docs/reference/conventions.md`).

## Invariants

Never break these. The linked doc explains why each one exists.

- **Never select or exclude flows by duration or packet count, under any circumstances.**
  The split cuts every flow uniformly at the time boundaries. Do not reintroduce
  length- or duration-conditional handling, quantile gates, or "keep short flows whole"
  shortcuts. (`docs/components/02-temporal-split.md`)
- **`SplitFlowsDF` is the only splitter.** Do not reintroduce a packet-level (row-slice)
  split. A flow appearing in several splits is intended; a boundary that is not a clean
  time boundary is a bug. (`docs/components/02-temporal-split.md`)
- **Do not mix `flow_key` formats within one artefact lineage.** A re-extracted capture
  needs its split, caches, checkpoints and embeddings regenerated together, and the
  provenance asserts must stay in place. (`docs/reference/artefacts.md § Lineage`)
- **`CLS_PLACEMENT = "EOS"` and the token constants are fixed** for every existing cache and
  checkpoint. (`docs/reference/data-representation.md`)
- **Split names:** `test` is monitored during training, and `val` is the final held-out set,
  spent once. Never tune against `val`. (`docs/reference/conventions.md § Split role naming`)
- **AD is fitted on normals only**, thresholds are calibrated without labels, and the
  focus detector is fixed in advance, never chosen from the evaluation metrics. Rows that
  play no role are dropped before scoring or plotting. (`docs/components/07-anomaly-detection.md`)
- **Do not read dataset README or label files** when designing anomaly detection.
- **Do not reintroduce `retrieval_accuracy`.** (`docs/reference/model-library.md § Evaluation helpers`)
- **Ask before algorithmic decisions** (splitting, sampling, windowing, detector
  protocol). Confirm the mechanism step by step with the user. An existing plan does not
  count as approval.
