from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

from common import ARTIFACT_ROOT, PROJECT_ROOT, configure_environment, ensure_project_dirs, now_utc, write_json


def package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=15).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    configure_environment()
    ensure_project_dirs()
    report = {
        "timestamp_utc": now_utc(),
        "project_root": str(PROJECT_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: package_version(name)
            for name in [
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "datasets",
                "sentence_transformers",
                "streamlit",
                "sacrebleu",
                "yaml",
            ]
        },
        "torch_cuda": None,
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
    }

    try:
        import torch

        report["torch_cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        report["torch_cuda"] = {"error": str(exc)}

    output = ARTIFACT_ROOT / "system_report.json"
    write_json(output, report)
    print(f"Wrote {output}")
    print(report)


if __name__ == "__main__":
    main()
