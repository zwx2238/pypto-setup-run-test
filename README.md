# pypto-setup-run-test

Standalone helper repo for setting up a local PyPTO environment and running the AKG Agents PyPTO KernelBench pytest target.

## What It Does

The main script `setup_and_run_test.sh` automates the workflow that was originally captured from a shell history:

1. Download PyPTO third-party dependencies via `tools/prepare_env.sh`
2. Install local CANN packages if needed
3. Export `PYPTO_THIRD_PARTY_PATH`, `PTO_TILE_LIB_CODE_PATH`, `DEVICE_ID`, and `TILE_FWK_DEVICE_ID`
4. Install the local `pypto` checkout
5. Install `akg_agents` and initialize `KernelBench`
6. Run:

```bash
tests/op/bench/test_bench_kernelgen_only.py::test_kernelbench_torch_pypto_kernelgen_only_ascend910b4
```

## Default Layout

By default, the script assumes the following sibling checkout layout:

```text
/home/zwx/
  pypto/
  akg/
  pypto_download/
  pypto_toolkit/
  pypto-setup-run-test/
```

You can override any of those paths with command-line arguments.

## Requirements

- Linux shell environment
- `python3`
- `git`
- Local `pypto` source checkout
- Optional but typical:
  - local `akg` checkout
  - CANN environment or permission to install it locally
  - `~/.akg/settings.json` or `AKG_AGENTS_*` / `AIKG_*` model env vars

## Usage

Run with defaults:

```bash
bash setup_and_run_test.sh
```

Use a different `pypto` path and device id:

```bash
bash setup_and_run_test.sh --pypto-root /path/to/pypto --device-id 1
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
- If `akg_agents/.env` exists, it is loaded automatically.
- The script checks out KernelBench commit `21fbe5a642898cd60b8f60c7aefb43d475e11f33` to match the recorded workflow.
