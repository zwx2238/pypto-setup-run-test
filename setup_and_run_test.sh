#!/usr/bin/env bash

set -euo pipefail

SELF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${SELF_ROOT}/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYPTO_ROOT="${PYPTO_ROOT:-${WORK_ROOT}/pypto}"
DEVICE_TYPE="a2"
DEVICE_ID="${DEVICE_ID:-0}"
PYPTO_INSTALL_MODE="wheel"
RUN_HELLO_WORLD=false
SKIP_THIRD_PARTY=false
SKIP_CANN=false
SKIP_PYPTO_REQUIREMENTS=false
SKIP_PYPTO_INSTALL=false
SKIP_AKG_INSTALL=false
SKIP_LLM_CHECK=false
FORCE_PREPARE=false

TOOLKIT_ROOT="${TOOLKIT_ROOT:-${WORK_ROOT}/pypto_toolkit}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${WORK_ROOT}/pypto_download}"
THIRD_PARTY_PATH="${THIRD_PARTY_PATH:-${DOWNLOAD_ROOT}/third_party_packages}"
AKG_ROOT="${AKG_ROOT:-${WORK_ROOT}/akg}"
AKG_AGENTS_ROOT="${AKG_AGENTS_ROOT:-${AKG_ROOT}/akg_agents}"
PTO_ISA_PATH="${PTO_ISA_PATH:-${PYPTO_ROOT}/pto_isa/pto-isa}"
KERNELBENCH_COMMIT="${KERNELBENCH_COMMIT:-21fbe5a642898cd60b8f60c7aefb43d475e11f33}"
TEST_TARGET="${TEST_TARGET:-tests/op/bench/test_bench_kernelgen_only.py::test_kernelbench_torch_pypto_kernelgen_only_ascend910b4}"

PYTEST_EXTRA_ARGS=()

log_info() {
    printf '[INFO] %s\n' "$*"
}

log_warn() {
    printf '[WARN] %s\n' "$*" >&2
}

log_error() {
    printf '[ERROR] %s\n' "$*" >&2
}

die() {
    log_error "$*"
    exit 1
}

show_help() {
    cat <<EOF
Usage: $(basename "$0") [options] [-- <extra pytest args>]

This script automates a local PyPTO + AKG Agents benchmark workflow:
1. Prepare PyPTO third-party packages
2. Install CANN toolkit into a local path when needed
3. Export PyPTO-related environment variables
4. Install PyPTO
5. Install AKG Agents and initialize KernelBench
6. Run the target pytest case

Options:
  --python PATH                 Python executable to use (default: ${PYTHON_BIN})
  --pypto-root PATH             Local pypto repository root (default: ${PYPTO_ROOT})
  --device-type a2|a3           Device type for prepare_env.sh (default: ${DEVICE_TYPE})
  --device-id N                 Export DEVICE_ID and TILE_FWK_DEVICE_ID (default: ${DEVICE_ID})
  --pypto-install-mode MODE     wheel|editable (default: ${PYPTO_INSTALL_MODE})
  --toolkit-root PATH           CANN install root (default: ${TOOLKIT_ROOT})
  --download-root PATH          Download root used by prepare_env.sh (default: ${DOWNLOAD_ROOT})
  --third-party-path PATH       PYPTO_THIRD_PARTY_PATH (default: ${THIRD_PARTY_PATH})
  --pto-isa-path PATH           PTO_TILE_LIB_CODE_PATH (default: ${PTO_ISA_PATH})
  --akg-root PATH               AKG repo root (default: ${AKG_ROOT})
  --akg-agents-root PATH        AKG Agents root (default: ${AKG_AGENTS_ROOT})
  --test-target NODEID          Pytest node id (default: ${TEST_TARGET})
  --run-hello-world             Run examples/00_hello_world/hello_world.py before pytest
  --skip-third-party            Skip prepare_env third_party step
  --skip-cann                   Skip prepare_env cann step
  --skip-pypto-requirements     Skip pip install -r python/requirements.txt
  --skip-pypto-install          Skip pip install for PyPTO
  --skip-akg-install            Skip AKG Agents dependency/install steps
  --skip-llm-check              Do not require ~/.akg/settings.json
  --force-prepare               Re-run prepare steps even if target paths already exist
  -h, --help                    Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --pypto-root /path/to/pypto --device-id 1
  $(basename "$0") --pypto-install-mode editable -- --maxfail=1 -x
EOF
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Command not found: ${cmd}"
}

ensure_dir() {
    local dir="$1"
    mkdir -p "$dir"
}

run_cmd() {
    log_info "Running: $*"
    "$@"
}

resolve_cann_env_script() {
    local candidates=(
        "${TOOLKIT_ROOT}/ascend-toolkit/set_env.sh"
        "${TOOLKIT_ROOT}/cann/set_env.sh"
        "${TOOLKIT_ROOT}/Ascend/ascend-toolkit/set_env.sh"
        "${TOOLKIT_ROOT}/Ascend/cann/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        "/usr/local/Ascend/cann/set_env.sh"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --python)
                shift
                [[ $# -gt 0 ]] || die "--python requires a value"
                PYTHON_BIN="$1"
                ;;
            --pypto-root)
                shift
                [[ $# -gt 0 ]] || die "--pypto-root requires a value"
                PYPTO_ROOT="$1"
                PTO_ISA_PATH="${PYPTO_ROOT}/pto_isa/pto-isa"
                ;;
            --device-type)
                shift
                [[ $# -gt 0 ]] || die "--device-type requires a value"
                DEVICE_TYPE="$1"
                ;;
            --device-id)
                shift
                [[ $# -gt 0 ]] || die "--device-id requires a value"
                DEVICE_ID="$1"
                ;;
            --pypto-install-mode)
                shift
                [[ $# -gt 0 ]] || die "--pypto-install-mode requires a value"
                PYPTO_INSTALL_MODE="$1"
                ;;
            --toolkit-root)
                shift
                [[ $# -gt 0 ]] || die "--toolkit-root requires a value"
                TOOLKIT_ROOT="$1"
                ;;
            --download-root)
                shift
                [[ $# -gt 0 ]] || die "--download-root requires a value"
                DOWNLOAD_ROOT="$1"
                THIRD_PARTY_PATH="${DOWNLOAD_ROOT}/third_party_packages"
                ;;
            --third-party-path)
                shift
                [[ $# -gt 0 ]] || die "--third-party-path requires a value"
                THIRD_PARTY_PATH="$1"
                ;;
            --pto-isa-path)
                shift
                [[ $# -gt 0 ]] || die "--pto-isa-path requires a value"
                PTO_ISA_PATH="$1"
                ;;
            --akg-root)
                shift
                [[ $# -gt 0 ]] || die "--akg-root requires a value"
                AKG_ROOT="$1"
                AKG_AGENTS_ROOT="${AKG_ROOT}/akg_agents"
                ;;
            --akg-agents-root)
                shift
                [[ $# -gt 0 ]] || die "--akg-agents-root requires a value"
                AKG_AGENTS_ROOT="$1"
                ;;
            --test-target)
                shift
                [[ $# -gt 0 ]] || die "--test-target requires a value"
                TEST_TARGET="$1"
                ;;
            --run-hello-world)
                RUN_HELLO_WORLD=true
                ;;
            --skip-third-party)
                SKIP_THIRD_PARTY=true
                ;;
            --skip-cann)
                SKIP_CANN=true
                ;;
            --skip-pypto-requirements)
                SKIP_PYPTO_REQUIREMENTS=true
                ;;
            --skip-pypto-install)
                SKIP_PYPTO_INSTALL=true
                ;;
            --skip-akg-install)
                SKIP_AKG_INSTALL=true
                ;;
            --skip-llm-check)
                SKIP_LLM_CHECK=true
                ;;
            --force-prepare)
                FORCE_PREPARE=true
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            --)
                shift
                PYTEST_EXTRA_ARGS=("$@")
                break
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
        shift
    done
}

validate_args() {
    [[ -d "${PYPTO_ROOT}" ]] || die "PyPTO root does not exist: ${PYPTO_ROOT}"
    [[ -f "${PYPTO_ROOT}/tools/prepare_env.sh" ]] || die "Missing ${PYPTO_ROOT}/tools/prepare_env.sh"

    case "${DEVICE_TYPE}" in
        a2|a3) ;;
        *)
            die "--device-type must be a2 or a3"
            ;;
    esac

    case "${PYPTO_INSTALL_MODE}" in
        wheel|editable) ;;
        *)
            die "--pypto-install-mode must be wheel or editable"
            ;;
    esac
}

prepare_third_party() {
    if [[ "${SKIP_THIRD_PARTY}" == true ]]; then
        log_info "Skipping third-party prepare step"
        return
    fi

    if [[ -d "${THIRD_PARTY_PATH}" && -n "$(ls -A "${THIRD_PARTY_PATH}" 2>/dev/null)" && "${FORCE_PREPARE}" != true ]]; then
        log_info "Using existing third-party packages at ${THIRD_PARTY_PATH}"
        return
    fi

    ensure_dir "${DOWNLOAD_ROOT}"
    run_cmd bash "${PYPTO_ROOT}/tools/prepare_env.sh" \
        --type=third_party \
        --download-path="${DOWNLOAD_ROOT}"
}

prepare_cann() {
    local env_script=""

    if [[ "${SKIP_CANN}" == true ]]; then
        log_info "Skipping CANN prepare step"
        return
    fi

    if env_script="$(resolve_cann_env_script 2>/dev/null)" && [[ "${FORCE_PREPARE}" != true ]]; then
        log_info "Using existing CANN environment script: ${env_script}"
        return
    fi

    ensure_dir "${TOOLKIT_ROOT}"
    ensure_dir "${DOWNLOAD_ROOT}"
    run_cmd bash "${PYPTO_ROOT}/tools/prepare_env.sh" \
        --type=cann \
        --device-type="${DEVICE_TYPE}" \
        --download-path="${DOWNLOAD_ROOT}" \
        --install-path="${TOOLKIT_ROOT}" \
        --quiet
}

source_cann_env() {
    local env_script=""

    if [[ -n "${ASCEND_HOME_PATH:-}" || -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
        log_info "Ascend environment already present in current shell"
        return
    fi

    if env_script="$(resolve_cann_env_script 2>/dev/null)"; then
        log_info "Sourcing CANN environment: ${env_script}"
        # shellcheck disable=SC1090
        source "${env_script}"
        return
    fi

    if [[ "${SKIP_CANN}" == true ]]; then
        log_warn "No local CANN env script found. Assuming the shell has already been configured."
        return
    fi

    die "Cannot find a CANN set_env.sh under ${TOOLKIT_ROOT} or /usr/local/Ascend"
}

export_pypto_env() {
    if [[ ! -d "${THIRD_PARTY_PATH}" ]]; then
        die "Third-party path does not exist: ${THIRD_PARTY_PATH}"
    fi

    export PYPTO_THIRD_PARTY_PATH="${THIRD_PARTY_PATH}"
    export DEVICE_ID="${DEVICE_ID}"
    export TILE_FWK_DEVICE_ID="${DEVICE_ID}"

    if [[ -d "${PTO_ISA_PATH}" ]]; then
        export PTO_TILE_LIB_CODE_PATH="${PTO_ISA_PATH}"
    else
        log_warn "PTO ISA path does not exist, leaving PTO_TILE_LIB_CODE_PATH unchanged: ${PTO_ISA_PATH}"
    fi

    log_info "PYPTO_THIRD_PARTY_PATH=${PYPTO_THIRD_PARTY_PATH}"
    log_info "DEVICE_ID=${DEVICE_ID}"
    log_info "TILE_FWK_DEVICE_ID=${TILE_FWK_DEVICE_ID}"
    if [[ -n "${PTO_TILE_LIB_CODE_PATH:-}" ]]; then
        log_info "PTO_TILE_LIB_CODE_PATH=${PTO_TILE_LIB_CODE_PATH}"
    fi
}

install_pypto() {
    pushd "${PYPTO_ROOT}" >/dev/null

    if [[ "${SKIP_PYPTO_REQUIREMENTS}" != true ]]; then
        run_cmd "${PYTHON_BIN}" -m pip install -r python/requirements.txt
    else
        log_info "Skipping PyPTO requirements install"
    fi

    if [[ "${SKIP_PYPTO_INSTALL}" != true ]]; then
        if [[ "${PYPTO_INSTALL_MODE}" == "editable" ]]; then
            run_cmd "${PYTHON_BIN}" -m pip install -e . --verbose
        else
            run_cmd "${PYTHON_BIN}" -m pip install . --verbose
        fi
    else
        log_info "Skipping PyPTO install"
    fi

    popd >/dev/null
}

run_hello_world_example() {
    if [[ "${RUN_HELLO_WORLD}" != true ]]; then
        return
    fi

    pushd "${PYPTO_ROOT}" >/dev/null
    run_cmd "${PYTHON_BIN}" examples/00_hello_world/hello_world.py
    popd >/dev/null
}

ensure_akg_repo() {
    if [[ -d "${AKG_ROOT}/.git" ]]; then
        log_info "Using existing AKG repo at ${AKG_ROOT}"
        return
    fi

    ensure_dir "$(dirname "${AKG_ROOT}")"
    run_cmd git clone https://gitcode.com/mindspore/akg -b br_agents "${AKG_ROOT}"
}

ensure_kernelbench() {
    local kernelbench_root="${AKG_AGENTS_ROOT}/thirdparty/KernelBench"

    run_cmd git -C "${AKG_ROOT}" submodule update --init "akg_agents/thirdparty/*"

    if [[ ! -d "${kernelbench_root}" ]]; then
        die "KernelBench not found after submodule init: ${kernelbench_root}"
    fi

    if [[ -d "${kernelbench_root}/.git" ]]; then
        local current_commit
        current_commit="$(git -C "${kernelbench_root}" rev-parse HEAD)"
        if [[ "${current_commit}" != "${KERNELBENCH_COMMIT}" ]]; then
            run_cmd git -C "${kernelbench_root}" checkout "${KERNELBENCH_COMMIT}"
        fi
    fi
}

source_akg_env_file() {
    local env_file="${AKG_AGENTS_ROOT}/.env"

    if [[ -f "${env_file}" ]]; then
        log_info "Sourcing AKG env file: ${env_file}"
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi
}

check_llm_config() {
    if [[ "${SKIP_LLM_CHECK}" == true ]]; then
        return
    fi

    if [[ -f "${HOME}/.akg/settings.json" ]]; then
        log_info "Found AKG settings at ${HOME}/.akg/settings.json"
        return
    fi

    if [[ -n "${AKG_AGENTS_API_KEY:-}" || -n "${AKG_AGENTS_STANDARD_API_KEY:-}" || -n "${AKG_AGENTS_COMPLEX_API_KEY:-}" || -n "${AIKG_API_KEY:-}" || -n "${AIKG_STANDARD_API_KEY:-}" || -n "${AIKG_COMPLEX_API_KEY:-}" ]]; then
        log_info "Using AKG model configuration from environment variables"
        return
    fi

    die "Missing model configuration. Provide ${HOME}/.akg/settings.json or export AKG_AGENTS_*/AIKG_* model environment variables."
}

install_akg_agents() {
    if [[ ! -d "${AKG_AGENTS_ROOT}" ]]; then
        die "AKG Agents root does not exist: ${AKG_AGENTS_ROOT}"
    fi

    if [[ "${SKIP_AKG_INSTALL}" != true ]]; then
        pushd "${AKG_AGENTS_ROOT}" >/dev/null
        run_cmd "${PYTHON_BIN}" -m pip install -r requirements.txt
        run_cmd "${PYTHON_BIN}" -m pip install -e . --no-build-isolation
        popd >/dev/null
    else
        log_info "Skipping AKG Agents install"
    fi
}

run_test() {
    pushd "${AKG_AGENTS_ROOT}" >/dev/null

    if [[ -f "${AKG_AGENTS_ROOT}/env.sh" ]]; then
        # shellcheck disable=SC1091
        source "${AKG_AGENTS_ROOT}/env.sh"
    fi

    source_akg_env_file

    log_info "Running pytest target: ${TEST_TARGET}"
    if [[ ${#PYTEST_EXTRA_ARGS[@]} -gt 0 ]]; then
        run_cmd "${PYTHON_BIN}" -m pytest -sv "${TEST_TARGET}" "${PYTEST_EXTRA_ARGS[@]}"
    else
        run_cmd "${PYTHON_BIN}" -m pytest -sv "${TEST_TARGET}"
    fi

    popd >/dev/null
}

main() {
    parse_args "$@"
    validate_args

    require_cmd bash
    require_cmd git
    require_cmd "${PYTHON_BIN}"

    prepare_third_party
    prepare_cann
    source_cann_env
    export_pypto_env
    install_pypto
    run_hello_world_example

    ensure_akg_repo
    ensure_kernelbench
    source_akg_env_file
    check_llm_config
    install_akg_agents
    run_test
}

main "$@"
