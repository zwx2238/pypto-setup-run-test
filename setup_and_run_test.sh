#!/usr/bin/env bash

set -euo pipefail

SELF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-${SELF_ROOT}}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${WORK_ROOT}/.local}"
SOURCE_ROOT="${SOURCE_ROOT:-${WORK_ROOT}/repos}"
STATE_ROOT="${STATE_ROOT:-${RUNTIME_ROOT}/state}"
CACHE_ROOT="${CACHE_ROOT:-${RUNTIME_ROOT}/cache}"
HOME_ROOT="${HOME_ROOT:-${STATE_ROOT}/home}"
XDG_CACHE_ROOT="${XDG_CACHE_ROOT:-${CACHE_ROOT}/xdg}"
XDG_CONFIG_ROOT="${XDG_CONFIG_ROOT:-${STATE_ROOT}/config}"
XDG_DATA_ROOT="${XDG_DATA_ROOT:-${STATE_ROOT}/share}"
BOOTSTRAP_PYTHON_BIN="${PYTHON_BIN:-${BOOTSTRAP_PYTHON_BIN:-python3}}"
PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN}"
VENV_ROOT="${VENV_ROOT:-${RUNTIME_ROOT}/venv}"
USE_VENV=true

PYPTO_ROOT="${PYPTO_ROOT:-${SOURCE_ROOT}/pypto}"
PYPTO_GIT_REF_REQUESTED=false
if [[ -n "${PYPTO_GIT_REF+x}" ]]; then
    PYPTO_GIT_REF_REQUESTED=true
fi
PYPTO_GIT_URL="${PYPTO_GIT_URL:-}"
PYPTO_GIT_REF="${PYPTO_GIT_REF:-}"
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

TOOLKIT_ROOT="${TOOLKIT_ROOT:-${RUNTIME_ROOT}/toolkit}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${RUNTIME_ROOT}/downloads}"
THIRD_PARTY_PATH="${THIRD_PARTY_PATH:-${DOWNLOAD_ROOT}/third_party_packages}"
AKG_ROOT="${AKG_ROOT:-${SOURCE_ROOT}/akg}"
AKG_GIT_REF_REQUESTED=false
if [[ -n "${AKG_GIT_REF+x}" ]]; then
    AKG_GIT_REF_REQUESTED=true
fi
AKG_GIT_URL="${AKG_GIT_URL:-https://gitcode.com/mindspore/akg}"
AKG_GIT_REF="${AKG_GIT_REF:-br_agents}"
AKG_AGENTS_ROOT="${AKG_AGENTS_ROOT:-${AKG_ROOT}/akg_agents}"
PTO_ISA_PATH="${PTO_ISA_PATH:-${PYPTO_ROOT}/pto_isa/pto-isa}"
KERNELBENCH_COMMIT="${KERNELBENCH_COMMIT:-21fbe5a642898cd60b8f60c7aefb43d475e11f33}"
TEST_TARGET="${TEST_TARGET:-tests/op/bench/test_bench_kernelgen_only.py::test_kernelbench_torch_pypto_kernelgen_only_ascend910b4}"
AKG_SETTINGS_PATH="${AKG_SETTINGS_PATH:-${HOME_ROOT}/.akg/settings.json}"

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

This script automates an isolated PyPTO + AKG Agents benchmark workflow:
1. Prepare local runtime directories inside this repo
2. Prepare PyPTO third-party packages
3. Install CANN toolkit into a local path when needed
4. Export PyPTO-related environment variables
5. Install PyPTO into a local virtualenv by default
6. Install AKG Agents and initialize KernelBench
7. Run the target pytest case

Options:
  --python PATH                 Bootstrap Python used to create the local venv (default: ${BOOTSTRAP_PYTHON_BIN})
  --venv-root PATH              Local virtualenv directory (default: ${VENV_ROOT})
  --no-venv                     Disable the local virtualenv and use --python directly
  --pypto-root PATH             Local pypto repository root (default: ${PYPTO_ROOT})
  --pypto-git-url URL           Clone PyPTO into --pypto-root when missing
  --pypto-git-ref REF           Checkout this PyPTO branch/tag/commit after clone or on existing git repo
  --device-type a2|a3           Device type for prepare_env.sh (default: ${DEVICE_TYPE})
  --device-id N                 Export DEVICE_ID and TILE_FWK_DEVICE_ID (default: ${DEVICE_ID})
  --pypto-install-mode MODE     wheel|editable (default: ${PYPTO_INSTALL_MODE})
  --toolkit-root PATH           CANN install root (default: ${TOOLKIT_ROOT})
  --download-root PATH          Download root used by prepare_env.sh (default: ${DOWNLOAD_ROOT})
  --third-party-path PATH       PYPTO_THIRD_PARTY_PATH (default: ${THIRD_PARTY_PATH})
  --pto-isa-path PATH           PTO_TILE_LIB_CODE_PATH (default: ${PTO_ISA_PATH})
  --akg-root PATH               AKG repo root (default: ${AKG_ROOT})
  --akg-git-url URL             AKG clone URL when --akg-root is missing (default: ${AKG_GIT_URL})
  --akg-git-ref REF             AKG branch/tag/commit to use (default: ${AKG_GIT_REF})
  --akg-agents-root PATH        AKG Agents root (default: ${AKG_AGENTS_ROOT})
  --home-root PATH              Local HOME used by this script (default: ${HOME_ROOT})
  --test-target NODEID          Pytest node id (default: ${TEST_TARGET})
  --run-hello-world             Run examples/00_hello_world/hello_world.py before pytest
  --skip-third-party            Skip prepare_env third_party step
  --skip-cann                   Skip prepare_env cann step
  --skip-pypto-requirements     Skip pip install -r python/requirements.txt
  --skip-pypto-install          Skip pip install for PyPTO
  --skip-akg-install            Skip AKG Agents dependency/install steps
  --skip-llm-check              Do not require local AKG settings or model env vars
  --force-prepare               Re-run prepare steps even if target paths already exist
  -h, --help                    Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --pypto-root /path/to/pypto --device-id 1
  $(basename "$0") --pypto-git-url <pypto_git_url> --pypto-git-ref main
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

configure_runtime_env() {
    ensure_dir "${RUNTIME_ROOT}"
    ensure_dir "${STATE_ROOT}"
    ensure_dir "${CACHE_ROOT}"
    ensure_dir "${HOME_ROOT}"
    ensure_dir "${XDG_CACHE_ROOT}"
    ensure_dir "${XDG_CONFIG_ROOT}"
    ensure_dir "${XDG_DATA_ROOT}"
    ensure_dir "$(dirname "${AKG_SETTINGS_PATH}")"

    export HOME="${HOME_ROOT}"
    export XDG_CACHE_HOME="${XDG_CACHE_ROOT}"
    export XDG_CONFIG_HOME="${XDG_CONFIG_ROOT}"
    export XDG_DATA_HOME="${XDG_DATA_ROOT}"
    export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export AKG_SETTINGS_PATH="${AKG_SETTINGS_PATH}"

    ensure_dir "${PIP_CACHE_DIR}"

    log_info "Using local runtime root: ${RUNTIME_ROOT}"
    log_info "Using local HOME: ${HOME}"
}

ensure_python_env() {
    if [[ "${USE_VENV}" != true ]]; then
        PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN}"
        log_warn "Local virtualenv disabled; installs will use ${PYTHON_BIN}"
        return
    fi

    if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
        ensure_dir "$(dirname "${VENV_ROOT}")"
        run_cmd "${BOOTSTRAP_PYTHON_BIN}" -m venv "${VENV_ROOT}"
    fi

    export VIRTUAL_ENV="${VENV_ROOT}"
    export PATH="${VENV_ROOT}/bin:${PATH}"
    PYTHON_BIN="${VENV_ROOT}/bin/python"
    log_info "Using local virtualenv: ${VENV_ROOT}"
}

resolve_cann_env_script() {
    local candidates=(
        "${TOOLKIT_ROOT}/ascend-toolkit/set_env.sh"
        "${TOOLKIT_ROOT}/cann/set_env.sh"
        "${TOOLKIT_ROOT}/Ascend/ascend-toolkit/set_env.sh"
        "${TOOLKIT_ROOT}/Ascend/cann/set_env.sh"
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
                BOOTSTRAP_PYTHON_BIN="$1"
                ;;
            --venv-root)
                shift
                [[ $# -gt 0 ]] || die "--venv-root requires a value"
                VENV_ROOT="$1"
                ;;
            --no-venv)
                USE_VENV=false
                ;;
            --pypto-root)
                shift
                [[ $# -gt 0 ]] || die "--pypto-root requires a value"
                PYPTO_ROOT="$1"
                PTO_ISA_PATH="${PYPTO_ROOT}/pto_isa/pto-isa"
                ;;
            --pypto-git-url)
                shift
                [[ $# -gt 0 ]] || die "--pypto-git-url requires a value"
                PYPTO_GIT_URL="$1"
                ;;
            --pypto-git-ref)
                shift
                [[ $# -gt 0 ]] || die "--pypto-git-ref requires a value"
                PYPTO_GIT_REF_REQUESTED=true
                PYPTO_GIT_REF="$1"
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
            --akg-git-url)
                shift
                [[ $# -gt 0 ]] || die "--akg-git-url requires a value"
                AKG_GIT_URL="$1"
                ;;
            --akg-git-ref)
                shift
                [[ $# -gt 0 ]] || die "--akg-git-ref requires a value"
                AKG_GIT_REF_REQUESTED=true
                AKG_GIT_REF="$1"
                ;;
            --akg-agents-root)
                shift
                [[ $# -gt 0 ]] || die "--akg-agents-root requires a value"
                AKG_AGENTS_ROOT="$1"
                ;;
            --home-root)
                shift
                [[ $# -gt 0 ]] || die "--home-root requires a value"
                HOME_ROOT="$1"
                AKG_SETTINGS_PATH="${HOME_ROOT}/.akg/settings.json"
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

ensure_git_repo() {
    local repo_root="$1"
    local repo_url="$2"
    local repo_ref="$3"
    local repo_name="$4"
    local repo_ref_requested="$5"

    if [[ -d "${repo_root}/.git" ]]; then
        log_info "Using existing ${repo_name} repo at ${repo_root}"
        if [[ "${repo_ref_requested}" == true && -n "${repo_ref}" ]]; then
            run_cmd git -C "${repo_root}" checkout "${repo_ref}"
        fi
        return
    fi

    if [[ -d "${repo_root}" && -n "$(ls -A "${repo_root}" 2>/dev/null)" ]]; then
        die "${repo_name} root exists but is not a git repo: ${repo_root}"
    fi

    [[ -n "${repo_url}" ]] || die "${repo_name} repo not found at ${repo_root}. Provide --${repo_name}-git-url or --${repo_name}-root."

    ensure_dir "$(dirname "${repo_root}")"
    if [[ -n "${repo_ref}" ]]; then
        run_cmd git clone --branch "${repo_ref}" "${repo_url}" "${repo_root}"
    else
        run_cmd git clone "${repo_url}" "${repo_root}"
    fi
}

ensure_pypto_repo() {
    if [[ -f "${PYPTO_ROOT}/tools/prepare_env.sh" ]]; then
        log_info "Using existing PyPTO repo at ${PYPTO_ROOT}"
        if [[ "${PYPTO_GIT_REF_REQUESTED}" == true && -n "${PYPTO_GIT_REF}" && -d "${PYPTO_ROOT}/.git" ]]; then
            run_cmd git -C "${PYPTO_ROOT}" checkout "${PYPTO_GIT_REF}"
        fi
        return
    fi

    if [[ -d "${PYPTO_ROOT}" && -n "$(ls -A "${PYPTO_ROOT}" 2>/dev/null)" ]]; then
        die "PyPTO root exists but does not look usable: ${PYPTO_ROOT}"
    fi

    [[ -n "${PYPTO_GIT_URL}" ]] || die "PyPTO repo not found at ${PYPTO_ROOT}. Put a checkout there, pass --pypto-root, or provide --pypto-git-url."

    ensure_dir "$(dirname "${PYPTO_ROOT}")"
    if [[ -n "${PYPTO_GIT_REF}" ]]; then
        run_cmd git clone --branch "${PYPTO_GIT_REF}" "${PYPTO_GIT_URL}" "${PYPTO_ROOT}"
    else
        run_cmd git clone "${PYPTO_GIT_URL}" "${PYPTO_ROOT}"
    fi

    [[ -f "${PYPTO_ROOT}/tools/prepare_env.sh" ]] || die "Missing ${PYPTO_ROOT}/tools/prepare_env.sh after PyPTO clone"
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
        log_warn "No local CANN env script found under ${TOOLKIT_ROOT}. Assuming the shell has already been configured."
        return
    fi

    die "Cannot find a CANN set_env.sh under ${TOOLKIT_ROOT}"
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
    ensure_git_repo "${AKG_ROOT}" "${AKG_GIT_URL}" "${AKG_GIT_REF}" "akg" "${AKG_GIT_REF_REQUESTED}"
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

    if [[ -f "${AKG_SETTINGS_PATH}" ]]; then
        log_info "Found AKG settings at ${AKG_SETTINGS_PATH}"
        return
    fi

    if [[ -n "${AKG_AGENTS_API_KEY:-}" || -n "${AKG_AGENTS_STANDARD_API_KEY:-}" || -n "${AKG_AGENTS_COMPLEX_API_KEY:-}" || -n "${AIKG_API_KEY:-}" || -n "${AIKG_STANDARD_API_KEY:-}" || -n "${AIKG_COMPLEX_API_KEY:-}" ]]; then
        log_info "Using AKG model configuration from environment variables"
        return
    fi

    die "Missing model configuration. Provide ${AKG_SETTINGS_PATH} or export AKG_AGENTS_*/AIKG_* model environment variables."
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
    require_cmd "${BOOTSTRAP_PYTHON_BIN}"

    configure_runtime_env
    ensure_python_env

    require_cmd "${PYTHON_BIN}"

    ensure_pypto_repo
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
