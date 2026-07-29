"""Detect and stop the isolated vLLM server used for evaluation."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import contextmanager

# Match however the server was launched: `python -m vllm.entrypoints.openai.api_server`
# (used by run_pipeline.sh), the `vllm serve` CLI, and the spawned engine workers.
VLLM_PGREP_PATTERNS = (
    r"vllm\.entrypoints\.openai\.api_server",
    r"vllm serve",
    r"VLLM::EngineCore",
)


def find_vllm_pids() -> list[int]:
    """Return PIDs for vLLM serve and EngineCore worker processes."""
    pids: set[int] = set()
    for pattern in VLLM_PGREP_PATTERNS:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
    return sorted(pids)


def is_vllm_running() -> bool:
    return bool(find_vllm_pids())


def get_gpu_memory_gb(device_index: int = 0) -> dict[str, float] | None:
    """Return GPU memory stats in GB, or None if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            used_bytes = total_bytes - free_bytes
            return {
                "total_gb": total_bytes / (1024**3),
                "used_gb": used_bytes / (1024**3),
                "free_gb": free_bytes / (1024**3),
            }
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        total_mb, used_mb, free_mb = (float(x.strip()) for x in result.stdout.strip().split(","))
        return {
            "total_gb": total_mb / 1024,
            "used_gb": used_mb / 1024,
            "free_gb": free_mb / 1024,
        }
    except Exception:
        return None


def print_gpu_memory(prefix: str = "GPU memory") -> None:
    mem = get_gpu_memory_gb()
    if mem is None:
        print(f"{prefix}: unavailable")
        return
    print(
        f"{prefix}: {mem['free_gb']:.1f} GB free / "
        f"{mem['used_gb']:.1f} GB used / {mem['total_gb']:.1f} GB total"
    )


def stop_vllm_server(grace_seconds: int = 30, verbose: bool = True) -> bool:
    """
    Stop local vLLM serve / OpenAI API server processes.

    Returns True if no vLLM processes remain after the stop attempt.
    """
    pids = find_vllm_pids()
    if not pids:
        if verbose:
            print("No vLLM process found.")
            print_gpu_memory()
        return True

    if verbose:
        print(f"Stopping vLLM (PIDs: {', '.join(map(str, pids))})...")

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    for _ in range(grace_seconds):
        if not find_vllm_pids():
            if verbose:
                print("vLLM stopped.")
                print_gpu_memory()
            return True
        time.sleep(1)

    for pid in find_vllm_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    remaining = find_vllm_pids()
    if remaining:
        if verbose:
            print(f"Warning: vLLM may still be running (PIDs: {', '.join(map(str, remaining))})")
        return False

    if verbose:
        print("vLLM stopped (forced).")
        print_gpu_memory()
    return True


@contextmanager
def stop_vllm_on_exit_if(enabled: bool):
    """Stop vLLM when the wrapped block exits (success, error, or KeyboardInterrupt)."""
    try:
        yield
    finally:
        if enabled:
            print("\n--- Stopping vLLM (--stop-vllm-on-exit) ---")
            stop_vllm_server()
