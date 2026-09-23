# Conventions

## Running anything

Everything runs inside the devcontainer (`.devcontainer/`: compose service and container
both named `devcontainer`, user `plb41586`, base image
`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel`). The host has no working Python
environment. Any `python` call on the host is wrong: it either fails or finds an
unrelated interpreter.

The repo is bind-mounted from the host at `/home/plb41586/projects/PretrainedTrafficModels`
to `/home/plb41586/workspace` in the container, so host edits show up immediately and
nothing needs copying or rebuilding. Scripts are run as modules, from the workspace
directory:

```
docker exec -w /home/plb41586/workspace devcontainer python -m RawByteTrafficModelling.PreTraining.<Script>
```

- **`-w` is required.** Imports such as `from RawByteTrafficModelling.ModelComponents...`
  resolve through the working directory, not `PYTHONPATH`. The `.bashrc` exports
  `/workspace`, a leftover from an older mount point that does not exist.
- **Python** is the venv at `/home/plb41586/app/venv`, which a Dockerfile `ENV` puts first on
  `PATH`, so plain `python` works in a non-interactive `docker exec`. It contains GPU-only
  packages (`mamba-ssm`, `causal-conv1d`, `keras_hub`, `torch`) that `requirements.txt`
  does not fully list.
- **Shell features** (pipes, redirection, globs) need `bash -lc '<command>'`. The container
  shell is bash and the host shell is fish, so quoting differs.
- **Rust:** `cargo` is in `~/.cargo/bin`, which is only on `PATH` in a login shell:
  `docker exec -w /home/plb41586/workspace/feature_extraction devcontainer bash -lc 'cargo build --release'`.
  The built binary's `PATH` entry uses the same stale `/workspace` prefix, so call it by its
  full path.
- **Redis** runs as a second compose service, reachable at host `redis`. There is no
  FalkorDB service.
- **No test suite, lint or CI** is set up for the Python code. To check a change, run the
  relevant script with `SMOKE = True` or a small `MAX_STEPS_PER_EPOCH`. The Rust crate has
  unit tests (`cargo test`).
- Start long training runs yourself from a terminal. The coding agent has its own rules
  for container commands, in `CLAUDE.md`.

**GPUs.** The container has two A5000s (24 GB each). Run independent jobs on one GPU each
(`DEVICE_INDEX` → `RunConfig.resolve_device`). DDP does not help here: the data pipeline
runs single-threaded on the CPU in the main thread, and DDP would not speed up that part.
Any script that imports `keras_hub` (currently only `PacketLevelMLM.py`) must call
`tf.config.set_visible_devices([], "GPU")` before that import, or TensorFlow takes almost
all the memory on both GPUs.

## Script anatomy

The scripts have no CLI. Each one is configured by constants at the top of the file, and
changing a setting means editing that constant. Most scripts share these constants:

| Constant | Meaning |
|---|---|
| `RUN_NAME` | names the output directory and checkpoint files. It usually includes capture, width and backbone, e.g. `SeqAE_CICAPT_d128_Mamba_s512` |
| `DATASET` | `DATASETS["<capture>"]` ([artefacts](artefacts.md#dataset-registry)) |
| `DEVICE_INDEX` | which GPU |
| `SMOKE` | runs the same code path at a tiny scale and appends `_smoke` to the output dir, so a smoke run cannot overwrite a real one. Run it once before every real run |
| `MAX_STEPS_PER_EPOCH` | caps the number of batches per epoch (`None` = full epoch) |
| `RESUME_FROM` | a checkpoint to resume from. Restores optimizer, scheduler, best metric and step |
| `Epochs`, `WARMUP_STEPS`, … | the cosine schedule spans the whole run, so changing `Epochs` changes the shape of the LR curve, not only where training stops |

## `PreTraining/RunConfig.py`

This file holds everything that is the same across experiments: the dataset registry,
the token constants, `make_id_encoder`, `packet_encoder_params`, `setup_run` (creates
the output dir and sets up file and stdout logging), `resolve_device`,
`cosine_warmup_lambda`, `fixed_eval_batches` (a fixed held-out subset, so eval numbers
can be compared across evals), `MetricsCsv` (moves aside a CSV whose columns are stale)
and `plot_curves` (reads the CSV back, so a resumed run still plots its earlier epochs).

New scripts should import these helpers. `PacketLevelAutoEncoder.py` is the cleanest
template. `SequenceLevelAutoEncoder.py` is older than `RunConfig` and still has its own
copies of the logging, metrics and curve code. Scripts that do not use `setup_run` have to
create their own output directory, because a `logging.FileHandler` raises if the
directory is missing.

## Split role naming

The naming here is the reverse of the usual convention:

| Role | Used for |
|---|---|
| `train` | fitting the packet and sequence AEs |
| `test` | **monitored during training**, and used to select `_best` checkpoints |
| `val` | **final held-out set**, read once for the reported numbers |
| `ad_fit`, `ad_calib`, `late` | six-role layout only; see [02](../components/02-temporal-split.md#split-roles) |

Every rerun that reads `val` uses it up again. Run with `SMOKE` to check the wiring
before any run that reads `val`.

## Invariants

The rules that changes must not break are listed once, in `CLAUDE.md § Invariants`.
Each rule there links to the doc that explains why it exists.
