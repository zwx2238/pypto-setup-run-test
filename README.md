# pypto-setup-run-test

Standalone helper repo for setting up an isolated local PyPTO environment and running the AKG Agents PyPTO KernelBench pytest target.

## What It Does

The main script `setup_and_run_test.sh` automates the workflow in a repo-local layout:

1. Create a local runtime area inside this repo
2. Download PyPTO third-party dependencies via `tools/prepare_env.sh`
3. Install local CANN packages if needed
4. Export `PYPTO_THIRD_PARTY_PATH`, `PTO_TILE_LIB_CODE_PATH`, `DEVICE_ID`, and `TILE_FWK_DEVICE_ID`
5. Install PyPTO into a repo-local virtualenv by default
6. Install `akg_agents` and initialize `KernelBench`
7. Run:

```bash
tests/op/bench/test_bench_kernelgen_only.py::test_kernelbench_torch_pypto_kernelgen_only_ascend910b4
```

## Default Layout

By default, the script keeps sources and runtime state inside this repo instead of relying on sibling directories under your home directory:

```text
pypto-setup-run-test/
  setup_and_run_test.sh
  repos/
    pypto/                 # put an existing checkout here, or let the script clone it
    akg/                   # cloned automatically by default if missing
  .local/
    venv/                  # default Python install target
    downloads/             # prepare_env.sh download root
    toolkit/               # local CANN install root
    cache/                 # pip / XDG caches
    state/
      home/
        .akg/settings.json # default AKG settings location for this script
```

You can override any of those paths with command-line arguments, but the defaults are now repo-local and self-contained.

## Requirements

- Linux shell environment
- `python3` with `venv`
- `git`
- PyPTO source code, either:
  - already present at `repos/pypto`, or
  - provided via `--pypto-root`, or
  - clonable via `--pypto-git-url`
- Optional but typical:
  - local `akg` checkout, otherwise the script clones it into `repos/akg`
  - CANN environment or permission to install it locally
  - `.local/state/home/.akg/settings.json` or `AKG_AGENTS_*` / `AIKG_*` model env vars

## Usage

Run with defaults:

```bash
bash setup_and_run_test.sh
```

Use a different `pypto` path and device id:

```bash
bash setup_and_run_test.sh --pypto-root /path/to/pypto --device-id 1
```

Clone PyPTO into the repo-local `repos/pypto` path:

```bash
bash setup_and_run_test.sh --pypto-git-url <pypto_git_url> --pypto-git-ref main
```

Editable install and extra pytest args:

```bash
bash setup_and_run_test.sh --pypto-install-mode editable -- --maxfail=1 -x
```

Show help:

```bash
bash setup_and_run_test.sh --help
```

## Notes

- The script prefers reusing existing downloads and installs unless `--force-prepare` is specified.
- By default, `pip install` goes into `.local/venv`, not into your user-level Python environment.
- The script exports a repo-local `HOME` so tools that would normally write under `~` use `.local/state/home` instead.
- If `akg_agents/.env` exists, it is loaded automatically.
- The script checks out KernelBench commit `21fbe5a642898cd60b8f60c7aefb43d475e11f33` to match the recorded workflow.
