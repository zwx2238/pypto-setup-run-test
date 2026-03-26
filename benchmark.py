import os
import csv

_PREPEND_PATH_VARS = frozenset({"PATH", "LD_LIBRARY_PATH", "PYTHONPATH"})

def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key in _PREPEND_PATH_VARS:
                existing = os.environ.get(key, "")
                existing_parts = set(existing.split(os.pathsep))
                new_parts = [p for p in value.split(os.pathsep)
                             if p and p not in existing_parts]
                if new_parts:
                    prefix = os.pathsep.join(new_parts)
                    os.environ[key] = (
                        f"{prefix}{os.pathsep}{existing}" if existing else prefix
                    )
            else:
                os.environ.setdefault(key, value)

_load_env_file()

os.environ['AIKG_STREAM_OUTPUT'] = 'on'

import textwrap
import math
import asyncio
import argparse
import time
from typing import Optional, List
from tabulate import tabulate
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.utils.common_utils import create_log_dir
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import (
    register_local_worker,
    register_remote_worker,
    get_worker_manager,
)
from akg_agents.utils.environment_check import check_env_for_task
from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask


def get_device_id(default=0):
    return int(os.getenv("DEVICE_ID", default))


def add_op_prefix(op_name, benchmark="KernelBench"):
    return f"akg_agents_{benchmark.lower()}_{op_name}"

DEFAULT_REMOTE_WORKER_URL = "http://127.0.0.1:19001"
DEFAULT_PROFILE_RUN_TIMES = 50


def _resolve_npu_device_ids(default_device_id, cli_device_ids=None):
    if cli_device_ids:
        return list(cli_device_ids)

    try:
        import torch
    except Exception:
        return [default_device_id]

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return [default_device_id]

    try:
        device_count = int(torch.npu.device_count())
    except Exception:
        return [default_device_id]

    if device_count <= 0:
        return [default_device_id]
    return list(range(device_count))


def _get_akg_agents_root() -> str:
    root = os.environ.get("AKG_AGENTS_ROOT", "").strip()
    if not root:
        raise RuntimeError(
            "AKG_AGENTS_ROOT environment variable is not set. "
            "Export it to the akg_agents project root, e.g. "
            "export AKG_AGENTS_ROOT=/path/to/akg/akg_agents"
        )
    return os.path.abspath(root)


def _resolve_kernelbench_flat_level_dir(level: str = "level1") -> str:
    base_dir = os.path.join(
        _get_akg_agents_root(), "thirdparty", "KernelBench", "KernelBench", level
    )
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"KernelBench level dir not found: {base_dir}")
    return base_dir


def _resolve_kernelbench_pypto_level_dir(
    kernelbench_pypto_root: Optional[str] = None, level: str = "level1"
) -> str:
    repo_root = (kernelbench_pypto_root or os.getenv("KERNELBENCH_PYPTO_ROOT", "")).strip()
    if repo_root:
        repo_root = os.path.expanduser(repo_root)
        if not os.path.isabs(repo_root):
            repo_root = os.path.abspath(os.path.join(_get_akg_agents_root(), repo_root))
    else:
        repo_root = os.path.join(_get_akg_agents_root(), "thirdparty", "KernelBench-pypto")

    preferred_level_dir = os.path.join(repo_root, "KernelBench", level)
    if os.path.isdir(preferred_level_dir):
        return preferred_level_dir

    raise FileNotFoundError(
        "KernelBench-pypto level dir not found. "
        f"checked={preferred_level_dir}. "
        "Set --kernelbench-pypto-root or KERNELBENCH_PYPTO_ROOT."
    )


def _find_kernelbench_flat_task_file(task_index: int, level: str = "level1") -> str:
    flat_level_dir = _resolve_kernelbench_flat_level_dir(level)
    task_prefix = f"{int(task_index)}_"
    task_files = sorted(
        file
        for file in os.listdir(flat_level_dir)
        if file.endswith(".py") and file.startswith(task_prefix)
    )
    if not task_files:
        raise FileNotFoundError(
            f"KernelBench task file not found for index={task_index} in {flat_level_dir}"
        )
    return os.path.join(flat_level_dir, task_files[0])


def _collect_pypto_indices_in_category(category_dir: str) -> List[int]:
    indices: List[int] = []
    for entry in os.listdir(category_dir):
        entry_path = os.path.join(category_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            idx = int(entry)
        except ValueError:
            continue
        if any(file.endswith("_pypto.py") for file in os.listdir(entry_path)):
            indices.append(idx)
    return sorted(indices)


def _get_kernelbench_structured_indices_from_pypto(
    pypto_level_dir: str, category: Optional[str] = None
):
    categories = sorted(
        d for d in os.listdir(pypto_level_dir)
        if os.path.isdir(os.path.join(pypto_level_dir, d))
    )
    if category:
        if category not in categories:
            raise ValueError(
                f"无效的 KernelBench 类别: {category}，可选值: {', '.join(categories)}"
            )
        return _collect_pypto_indices_in_category(os.path.join(pypto_level_dir, category))

    return {
        cat: _collect_pypto_indices_in_category(os.path.join(pypto_level_dir, cat))
        for cat in categories
    }


def _find_pypto_case_dir(pypto_level_dir: str, task_index: int, category: Optional[str] = None) -> str:
    search_dir = pypto_level_dir
    if category:
        categories = sorted(
            d for d in os.listdir(pypto_level_dir)
            if os.path.isdir(os.path.join(pypto_level_dir, d))
        )
        if category not in categories:
            raise ValueError(
                f"无效的 KernelBench 类别: {category}，可选值: {', '.join(categories)}"
            )
        search_dir = os.path.join(pypto_level_dir, category)

    for root, _, files in os.walk(search_dir):
        if os.path.basename(root) == str(task_index):
            if any(file.endswith("_pypto.py") for file in files):
                return root
    raise FileNotFoundError(
        f"KernelBench pypto case dir not found: index={task_index}, search_dir={search_dir}"
    )


def _get_kernelbench_task_and_kernel_from_pypto_repo(
    task_index: int,
    pypto_level_dir: str,
    dsl: str = "pypto",
    level: str = "level1",
    category: Optional[str] = None,
):
    case_dir = _find_pypto_case_dir(pypto_level_dir, task_index, category=category)
    kernel_files = sorted(
        file for file in os.listdir(case_dir) if file.endswith(f"_{dsl}.py")
    )
    if not kernel_files:
        raise FileNotFoundError(f"Kernel file (*_{dsl}.py) not found in {case_dir}")
    kernel_file = os.path.join(case_dir, kernel_files[0])
    with open(kernel_file, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    task_file = _find_kernelbench_flat_task_file(task_index, level=level)
    with open(task_file, "r", encoding="utf-8") as f:
        task_desc = f.read()

    op_name = os.path.splitext(os.path.basename(task_file))[0]
    return op_name, task_desc, kernel_code


def _save_results_to_csv(results, run_profile, csv_path):
    fieldnames = [
        "category",
        "idx",
        "op_name",
        "pass_count",
        "run_count",
        "e2e_time_s",
        "result",
        "error_log",
    ]
    if run_profile:
        fieldnames.extend(
            [
                "gen_time_us",
                "base_time_us",
                "speedup_x",
            ]
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(results, key=lambda x: (x["category"], x["idx"])):
            row = {
                "category": item.get("category", ""),
                "idx": item.get("idx", ""),
                "op_name": item.get("op_name", ""),
                "pass_count": item.get("pass_count", 0),
                "run_count": item.get("run_count", 0),
                "e2e_time_s": f"{item.get('e2e_time_s', 0.0):.3f}",
                "result": "PASS" if item.get("result") else "FAIL",
                "error_log": item.get("error_log", ""),
            }
            if run_profile:
                profile_info = item.get("profile") or {}
                row.update(
                    {
                        "gen_time_us": profile_info.get("gen_time"),
                        "base_time_us": profile_info.get("base_time"),
                        "speedup_x": profile_info.get("speedup"),
                    }
                )
            writer.writerow(row)


async def benchmark_pypto_ascend910b4_torch(
    idx,
    category=None,
    parallel=1,
    run_profile=False,
    use_llm=False,
    llm_workflow="default",
    pass_times=1,
    worker_mode="local",
    pypto_run_mode: int = 0,
    devices: Optional[list] = None,
    kernelbench_pypto_root: Optional[str] = None,
    evolve_rounds: int = 1,
    evolve_parallel: int = 2,
    evolve_islands: int = 1,
    as_max_tasks: int = 2,
    as_concurrent: int = 2,
):
    pypto_level_dir = _resolve_kernelbench_pypto_level_dir(
        kernelbench_pypto_root=kernelbench_pypto_root, level="level1"
    )
    print(f"run settings: pypto_level_dir={pypto_level_dir}")

    device_id = get_device_id()
    framework = "torch"
    dsl = "pypto"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)
    config["pypto_run_mode"] = int(pypto_run_mode)
    if run_profile:
        config["profile_settings"] = {
            "run_times": DEFAULT_PROFILE_RUN_TIMES,
            "warmup_times": 5,
            "pypto_use_msprof_base": True,
        }
    if use_llm:
        check_env_for_task(
            framework,
            backend,
            dsl,
            config,
            is_remote=(worker_mode == "remote"),
        )
    if pass_times <= 0:
        raise ValueError("pass_times must be greater than 0")

    if worker_mode == "remote":
        resolved_worker_url = DEFAULT_REMOTE_WORKER_URL
        print(
            f"run settings: worker_mode=remote parallel={parallel} "
            f"worker_url={resolved_worker_url}"
        )
        await register_remote_worker(
            backend=backend, arch=arch, worker_url=resolved_worker_url
        )
    else:
        device_ids = [device_id]
        if backend == "ascend":
            device_ids = _resolve_npu_device_ids(device_id, cli_device_ids=devices)
        elif devices:
            device_ids = list(devices)
        print(
            f"run settings: worker_mode=local parallel={parallel} "
            f"device_ids={device_ids}"
        )
        await register_local_worker(device_ids, backend=backend, arch=arch)

    worker = None
    if not use_llm:
        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            raise RuntimeError(
                f"No available worker for backend={backend}, arch={arch}. Please register a worker first."
            )

    task_type = "profile" if run_profile else "precision_only"

    idx_list = None if idx is None else ([idx] if isinstance(idx, int) else list(idx))
    if category is None:
        category_list = None
    elif isinstance(category, str):
        category_list = [category]
    else:
        category_list = list(category)

    if idx_list is None:
        if category_list is None:
            category_indices_map = _get_kernelbench_structured_indices_from_pypto(
                pypto_level_dir
            )
            category_items = sorted(category_indices_map.items())
        else:
            category_items = [
                (
                    cat,
                    _get_kernelbench_structured_indices_from_pypto(
                        pypto_level_dir, category=cat
                    ),
                )
                for cat in category_list
            ]
    else:
        if category_list is None:
            category_items = [(None, idx_list)]
        else:
            category_items = [(cat, idx_list) for cat in category_list]

    def _shorten_error(msg, limit=200):
        if not msg:
            return ""
        msg = " ".join(str(msg).strip().split())
        return msg if len(msg) <= limit else f"{msg[:limit - 3]}..."

    results = []
    results_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(parallel)

    async def _run_case(category_name, case_idx):
        async with semaphore:
            category_label = category_name or "all"
            op_name = ""
            pass_count = 0
            run_count = 0
            error_log = "all runs failed"
            profile_result = None
            case_start_time = time.perf_counter()
            for run_idx in range(1, pass_times + 1):
                run_count += 1
                result = False
                run_error = ""
                run_profile_result = None
                try:
                    op_name, op_task_str, kernel_code = _get_kernelbench_task_and_kernel_from_pypto_repo(
                        case_idx,
                        pypto_level_dir=pypto_level_dir,
                        dsl=dsl,
                        level="level1",
                        category=category_name,
                    )
                    op_task_str = textwrap.dedent(op_task_str)

                    if use_llm:
                        task_op_name = add_op_prefix(op_name, benchmark="KernelBench")
                        task_id = f"{category_label}_{case_idx}_run{run_idx}"
                        workflow_map = {
                            "default": "default_workflow",
                            "kernelgenonly": "kernelgen_only_workflow",
                            "evolve": "evolve_workflow",
                            "adaptive_search": "adaptive_search_workflow",
                        }
                        task_config = dict(config)
                        if llm_workflow == "evolve":
                            task_config["max_rounds"] = evolve_rounds
                            task_config["parallel_num"] = evolve_parallel
                            task_config["num_islands"] = evolve_islands
                            task_config["migration_interval"] = 0
                            task_config["elite_size"] = 0
                        elif llm_workflow == "adaptive_search":
                            task_config["max_total_tasks"] = as_max_tasks
                            task_config["max_concurrent"] = as_concurrent
                            task_config["initial_task_count"] = as_concurrent

                        task = AIKGTask(
                            op_name=task_op_name,
                            task_desc=op_task_str,
                            task_id=task_id,
                            dsl=dsl,
                            backend=backend,
                            arch=arch,
                            config=task_config,
                            framework=framework,
                            task_type=task_type,
                            workflow=workflow_map.get(llm_workflow, "default_workflow"),
                        )
                        _, result, task_info = await task.run()
                        run_error = task_info.get("verifier_error", "")
                        if not result and not run_error:
                            run_error = task_info.get("error", "")
                        if result and run_profile:
                            run_profile_result = task_info.get("profile_res") or {}
                    else:
                        create_log_dir(
                            f"{op_name}_{framework}_{backend}_{arch}_{dsl}_test"
                        )

                        verifier = KernelVerifier(
                            op_name=op_name,
                            framework_code=op_task_str,
                            framework=framework,
                            dsl=dsl,
                            backend=backend,
                            arch=arch,
                            impl_func_name="ModelNew",
                            config=config,
                            worker=worker,
                        )
                        task_info = {"coder_code": kernel_code}
                        result, run_error = await verifier.run(task_info, device_id=device_id)

                        if result and run_profile:
                            profile_settings = {
                                "run_times": DEFAULT_PROFILE_RUN_TIMES,
                                "warmup_times": 5,
                                "pypto_use_msprof_base": True,
                            }
                            run_profile_result = await verifier.run_profile(
                                task_info,
                                current_step=0,
                                device_id=device_id,
                                profile_settings=profile_settings,
                            )

                    if result and run_profile:
                        gen_time = run_profile_result.get("gen_time")
                        base_time = run_profile_result.get("base_time")
                        speedup = run_profile_result.get("speedup", 0.0)
                        profile_error = run_profile_result.get("error", "")
                        debug_extract_dir = run_profile_result.get("debug_extract_dir")
                        verify_dir = run_profile_result.get("verify_dir")
                        if verify_dir:
                            print(
                                f"category {category_label} idx {case_idx} run {run_idx}/{pass_times} "
                                f"verify_dir={verify_dir}"
                            )
                        if debug_extract_dir:
                            print(
                                f"category {category_label} idx {case_idx} run {run_idx}/{pass_times} "
                                f"debug_extract_dir={debug_extract_dir}"
                            )
                        if gen_time is None or (
                            isinstance(gen_time, (int, float)) and math.isinf(gen_time)
                        ):
                            result = False
                            run_error = profile_error or "profile failed"
                            if debug_extract_dir:
                                run_error = f"{run_error} [debug_extract_dir={debug_extract_dir}]"
                            if verify_dir:
                                run_error = f"{run_error} [verify_dir={verify_dir}]"
                        else:
                            base_time_str = "N/A"
                            if base_time is not None and not (
                                isinstance(base_time, (int, float)) and math.isinf(base_time)
                            ):
                                base_time_str = f"{base_time:.2f}"
                            print(
                                f"category {category_label} idx {case_idx} run {run_idx}/{pass_times} "
                                f"profile gen={gen_time:.2f} us base={base_time_str} us "
                                f"speedup={speedup:.2f}x"
                            )
                except Exception as exc:
                    run_error = str(exc)

                if result:
                    pass_count += 1
                    profile_result = run_profile_result
                    print(
                        f"category {category_label} idx {case_idx} run {run_idx}/{pass_times} 验证成功"
                    )
                    break
                else:
                    error_log = run_error
                    print(
                        f"category {category_label} idx {case_idx} run {run_idx}/{pass_times} 验证失败: {run_error}"
                    )

            case_result = pass_count > 0
            case_elapsed_s = time.perf_counter() - case_start_time
            case_error_log = "" if case_result else error_log

            async with results_lock:
                results.append(
                    {
                        "category": category_label,
                        "idx": case_idx,
                        "op_name": op_name,
                        "result": case_result,
                        "pass_count": pass_count,
                        "run_count": run_count,
                        "e2e_time_s": case_elapsed_s,
                        "error_log": case_error_log,
                        "profile": profile_result,
                    }
                )

            if case_result:
                print(
                    f"category {category_label} idx {case_idx} case通过 ({pass_count}/{run_count})"
                )
            else:
                print(
                    f"category {category_label} idx {case_idx} case失败 (0/{run_count}): {case_error_log}"
                )

    tasks = []
    for category_name, case_idx_list in category_items:
        for case_idx in case_idx_list:
            tasks.append(asyncio.create_task(_run_case(category_name, case_idx)))

    if tasks:
        await asyncio.gather(*tasks)

    headers = [
        "category",
        "idx",
        "op_name",
        "pass_count",
        "run_count",
        "e2e_time(s)",
        "result",
        "error",
    ]
    if run_profile:
        headers.extend(["gen_time(us)", "base_time(us)", "speedup(x)"])

    rows = []
    for item in sorted(results, key=lambda x: (x["category"], x["idx"])):
        profile_info = item.get("profile") or {}
        row = [
            item["category"],
            item["idx"],
            item["op_name"],
            item["pass_count"],
            item["run_count"],
            f"{item.get('e2e_time_s', 0.0):.3f}",
            "PASS" if item["result"] else "FAIL",
            _shorten_error(item["error_log"]),
        ]
        if run_profile:
            gen_time = profile_info.get("gen_time")
            base_time = profile_info.get("base_time")
            speedup = profile_info.get("speedup", 0.0)
            base_time_str = (
                f"{base_time:.2f}"
                if isinstance(base_time, (int, float)) and not math.isinf(base_time)
                else ""
            )
            gen_time_str = (
                f"{gen_time:.2f}"
                if isinstance(gen_time, (int, float)) and not math.isinf(gen_time)
                else ""
            )
            speedup_str = (
                f"{speedup:.2f}"
                if isinstance(speedup, (int, float)) and not math.isinf(speedup)
                else ""
            )
            row.extend([gen_time_str, base_time_str, speedup_str])
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

    csv_output_path = os.path.abspath("benchmark_results.csv")

    _save_results_to_csv(results, run_profile, csv_output_path)
    print(f"结果已保存到 CSV: {csv_output_path}")

    total = len(results)
    passed = sum(1 for item in results if item["result"])
    failed = total - passed
    total_runs = sum(item["run_count"] for item in results)
    passed_runs = sum(item["pass_count"] for item in results)
    failed_runs = total_runs - passed_runs
    summary_rows = [
        ["total_cases", total],
        ["passed_cases", passed],
        ["failed_cases", failed],
        ["case_pass_rate", f"{(passed / total * 100):.2f}%" if total else "0.00%"],
        ["total_runs", total_runs],
        ["passed_runs", passed_runs],
        ["failed_runs", failed_runs],
        [
            "run_pass_rate",
            f"{(passed_runs / total_runs * 100):.2f}%" if total_runs else "0.00%",
        ],
    ]
    print(tabulate(summary_rows, headers=["metric", "value"], tablefmt="fancy_grid"))

    if failed:
        raise RuntimeError(f"验证失败: {failed} case(s) failed (pass@{pass_times})") 


def _parse_idx_list(raw_values, parser):
    idx_list = []
    for value in raw_values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx_list.append(int(part))
            except ValueError:
                parser.error(f"Invalid idx '{part}'")
    if not idx_list:
        parser.error("idx list is empty")
    return idx_list


def _parse_category_list(raw_values, parser):
    if raw_values is None:
        return None
    category_list = []
    for value in raw_values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            category_list.append(part)
    if not category_list:
        parser.error("category list is empty")
    if "all" in category_list:
        if len(category_list) > 1:
            parser.error("category 'all' cannot be combined with other categories")
        return None
    return category_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idx",
        nargs="+",
        default=None,
        help=(
            "kernel index (support multiple, e.g. --idx 1 2 or --idx 1,2). "
            "If omitted, run all indices in selected categories."
        ),
    )
    parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        help=(
            "KernelBench category (level1 subdir, support multiple, "
            "e.g. --category activation_functions reduction)"
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="max parallel cases (default: 1)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run profile after verification",
    )
    parser.add_argument(
        "--llm",
        nargs="?",
        const="default",
        default=None,
        choices=["default", "kernelgenonly", "evolve", "adaptive_search"],
        help="use LLM pipeline: default, kernelgenonly, evolve, or adaptive_search (codeonly removed)",
    )
    parser.add_argument(
        "--pass",
        dest="pass_times",
        type=int,
        default=1,
        help="repeat each case independently n times (default: 1)",
    )
    parser.add_argument(
        "--worker-mode",
        type=str,
        choices=["local", "remote"],
        default="local",
        help="worker mode: local or remote",
    )
    parser.add_argument(
        "--pypto-run-mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="AIKG_PYPTO_RUN_MODE for this task (default: 0, 0: NPU, 1: SIM)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="comma-separated device IDs for local worker (e.g. --devices 0,1,2,3)",
    )
    parser.add_argument(
        "--kernelbench-pypto-root",
        type=str,
        default="",
        help=(
            "path to KernelBench-pypto repo root "
            "(contains KernelBench/level1/...)"
        ),
    )
    parser.add_argument(
        "--evolve-rounds",
        type=int,
        default=1,
        help="evolve: number of evolution rounds (default: 1)",
    )
    parser.add_argument(
        "--evolve-parallel",
        type=int,
        default=2,
        help="evolve: number of parallel tasks per round (default: 2)",
    )
    parser.add_argument(
        "--evolve-islands",
        type=int,
        default=1,
        help="evolve: number of islands (default: 1)",
    )
    parser.add_argument(
        "--as-max-tasks",
        type=int,
        default=2,
        help="adaptive_search: max total tasks (default: 2)",
    )
    parser.add_argument(
        "--as-concurrent",
        type=int,
        default=2,
        help="adaptive_search: max concurrent tasks (default: 2)",
    )
    args = parser.parse_args()
    if args.parallel <= 0:
        parser.error("--parallel must be greater than 0")
    if args.pass_times <= 0:
        parser.error("--pass must be greater than 0")
    if args.llm == "evolve":
        if args.evolve_rounds <= 0:
            parser.error("--evolve-rounds must be greater than 0")
        if args.evolve_parallel <= 0:
            parser.error("--evolve-parallel must be greater than 0")
        if args.evolve_islands <= 0:
            parser.error("--evolve-islands must be greater than 0")
    if args.llm == "adaptive_search":
        if args.as_max_tasks <= 0:
            parser.error("--as-max-tasks must be greater than 0")
        if args.as_concurrent <= 0:
            parser.error("--as-concurrent must be greater than 0")
    if args.idx is None:
        idx_list = None if args.category is not None else [19]
    else:
        idx_list = _parse_idx_list(args.idx, parser)
    category_list = _parse_category_list(args.category, parser)
    run_profile = args.profile

    cli_devices = None
    if args.devices:
        cli_devices = []
        for part in args.devices.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                cli_devices.append(int(part))
            except ValueError:
                parser.error(f"--devices: invalid device id '{part}', must be integers")
        if not cli_devices:
            parser.error("--devices: device list is empty")

    asyncio.run(
        benchmark_pypto_ascend910b4_torch(
            idx_list,
            category_list,
            parallel=args.parallel,
            run_profile=run_profile,
            use_llm=args.llm is not None,
            llm_workflow=args.llm or "default",
            pass_times=args.pass_times,
            worker_mode=args.worker_mode,
            pypto_run_mode=args.pypto_run_mode,
            devices=cli_devices,
            kernelbench_pypto_root=args.kernelbench_pypto_root or None,
            evolve_rounds=args.evolve_rounds,
            evolve_parallel=args.evolve_parallel,
            evolve_islands=args.evolve_islands,
            as_max_tasks=args.as_max_tasks,
            as_concurrent=args.as_concurrent,
        )
    )
