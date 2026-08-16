# AGENTS.md

## Scope

This file applies to the entire repository.

This repository contains Python code for 3D medical-image experiments built with
PyTorch, Lightning, MONAI, and related libraries. Most training and inference
workloads require an NVIDIA GPU and local datasets that are not stored in Git.

## Repository layout

- `main.py` is the command-line entry point. It dynamically imports a module
  below `experiments` and invokes its `analyze`, `prune`, `train`, or
  `inference` callable.
- `experiments/` contains shared experiment infrastructure, data modules,
  callbacks, network implementations, and method-specific packages.
- `experiments/<method>/experiments/` contains named experiment variants that
  can be selected with `--experiment_name`.
- `experiments/nets/` contains reusable network components and builders.
- `experiments/utils/` contains shared filesystem and utility functions.
- `plans/` contains JSON planning/configuration data used by experiments.
- `data/` contains local datasets, preprocessing outputs, checkpoints, and
  inference results. It is ignored by Git and must not be treated as source.

## Environment and dependencies

- Use Python 3.12 or newer, as declared in `pyproject.toml`.
- Treat `pyproject.toml` and `uv.lock` as the primary dependency definition.
  Environment setup and dependency commands are performed manually by the user
  inside the project container.
- Agents must not edit `pyproject.toml`. If a dependency or project metadata
  change is needed, describe the exact proposed change and ask the user to make
  it manually. Do not run `uv sync`, package installation, lockfile generation,
  or dependency-update commands.
- PyTorch is pinned to the CUDA 12.1 package index. Do not change the PyTorch,
  CUDA, or other pinned dependency versions unless the task explicitly requires
  it.
- `requirements.txt` exists alongside the uv configuration, but it is not an
  exact substitute: some optional compiled dependencies differ. Do not update
  one dependency file mechanically from the other.
- Never commit credentials, machine-specific paths, datasets, model weights,
  logs, SQLite databases, or generated reports.

## Running the project

The top-level command has this shape:

```bash
uv run python main.py <lib> <method> [--experiment_name <name>] [method arguments]
```

For example, selecting `lib=munet` imports `experiments.munet`; adding
`--experiment_name overlap_0` imports
`experiments.munet.experiments.overlap_0`.

The method-specific parser receives all remaining arguments. Inspect the
selected module and its base class before running a command because training,
inference, analysis, and pruning can require positional data/output paths and
can write substantial artifacts. `--help-main` only displays the top-level
parser help; method arguments are defined by the selected implementation.

Do not launch a full training, inference, preprocessing, or pruning job merely
to validate a code change. These operations may require unavailable data,
multiple GPUs, significant time, and large disk writes.

## Implementation conventions

- Before editing, creating, moving, or deleting any file below `experiments/`,
  explain the exact intended change and obtain explicit user approval
  immediately before editing. This confirmation is always required, even when
  the user's original request already asks for an edit in `experiments/`. Prior
  approval for a different edit does not carry over.
- Keep changes focused on the requested behavior. Avoid unrelated architecture
  changes or broad formatting rewrites.
- Follow the existing module organization and naming style in the files being
  changed. There is currently no repository-wide formatter or linter
  configuration.
- Use `pathlib.Path` for paths and reuse helpers from `experiments.utils` where
  applicable. Avoid embedding absolute, user-specific paths in source code.
- Reuse the existing abstractions (`ArgumentAdaptor`, `Experiment`, data-module
  classes, `Builder`, callbacks, and configuration helpers) instead of
  duplicating their behavior.
- Put a new named experiment variant in the relevant
  `experiments/<method>/experiments/` package when it only changes one method's
  configuration. Change shared infrastructure only when the behavior is truly
  shared.
- Preserve checkpoint compatibility when editing model builders, module names,
  state-dict keys, or `UNetTrainingModule.CKPT_BUILDER_KEY`.
- Preserve tensor layout conventions in surrounding code. This project works
  primarily with 3D tensors, and silent channel/spatial-axis changes are
  especially risky.
- Keep CLI compatibility unless the user explicitly asks for a breaking change.
  Existing experiment modules are loaded dynamically by import path.

## Data and artifact safety

- Do not edit, delete, relocate, or commit files below `data/` unless the user
  explicitly requests the exact operation.
- Treat checkpoints, medical-image files, preprocessing outputs, MLflow data,
  TensorBoard logs, and generated visualizations as potentially large and
  expensive to reproduce.
- Before running code that writes output, identify the resolved destination and
  ensure it does not overwrite an existing experiment.
- Be especially careful with `prune`: its implementation creates symbolic links
  between target and save paths. Verify path pairing and destinations before
  executing it.
- Do not assume local medical data may be displayed, uploaded, or included in
  logs. Use synthetic metadata or tensors for tests whenever possible.

## Validation

There is currently no committed automated test suite and no configured pytest,
lint, type-check, or formatting command.

Agents must not execute Python commands in this repository. Python must be run
manually by the user inside the project's container environment. This includes
direct `python` invocations, `uv run python ...`, test runners, import checks,
training, inference, analysis, pruning, and Python-based formatting or linting
tools. When Python validation is needed, provide the exact command and explain
its purpose, then ask the user to run it and share the result.

Recommend validation proportionally to the change:

1. Ask the user to run a syntax check for changed Python files, for example:

   ```bash
   uv run python -m py_compile path/to/changed_file.py
   ```

2. Suggest focused imports or small CPU-only checks when they do not initialize
   a dataset, trainer, logger, or CUDA-only extension.
3. For tensor or model changes, prefer a minimal synthetic-input check. Keep it
   CPU-only unless CUDA behavior is the subject of the change.
4. Ask the user to run a real experiment only when it is necessary and the
   required data, hardware, runtime, and output destination are known.

Report which non-Python checks the agent performed and which Python commands
were delegated to the user. If validation was not completed because results
were not provided, or was limited by missing data, CUDA, compiled extensions,
or runtime cost, state that clearly rather than implying the workflow passed.

## Command execution policy

- Agents may run read-only commands that inspect files or repository state and
  do not modify the filesystem, repository, environment, data, or external
  state. Examples include `rg`, `sed`, `find`, `git status`, and `git diff`.
- Before running any command that may create, edit, move, delete, format, stage,
  install, generate, or otherwise modify state, explain the command's intended
  effect and obtain explicit user approval.
- Do not hide a state-changing operation inside an otherwise read-only command,
  script, pipeline, hook, or tool invocation.

## Change review

Before finishing:

- inspect `git diff` and keep unrelated user changes intact;
- confirm generated artifacts and local data are not staged;
- check that new experiment modules are importable through the path constructed
  in `main.py`;
- document any new required command, dependency, or data layout in the relevant
  repository documentation.
