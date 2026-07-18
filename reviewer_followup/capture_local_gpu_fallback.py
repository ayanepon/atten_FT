#!/usr/bin/env python3
"""Capture a fresh local nvidia-smi snapshot after the lab API is unavailable.

This is an explicit fallback artifact, not a replacement for the lab API.
The caller must provide the observed API error so provenance cannot silently
misrepresent an nvidia-smi snapshot as an API response.
"""

from __future__ import annotations

import argparse
import datetime as dt
import socket
import subprocess
import sys
from pathlib import Path

from reviewer_followup.common import atomic_write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--api-url",
        default="https://gpu-status.example.invalid/api/gpu/status",
        help="Site-local GPU-status endpoint attempted before this fallback.",
    )
    parser.add_argument("--api-error", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    query = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True, text=True, capture_output=True,
    )
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        check=False, text=True, capture_output=True,
    )
    gpus = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        index, name, total, used, free, utilization = [field.strip() for field in line.split(",", 5)]
        gpus.append(
            {
                "gpu_index": int(index), "name": name,
                "memory_total_mib": int(total), "memory_used_mib": int(used), "memory_free_mib": int(free),
                "utilization_percent": int(utilization), "processes": [],
            }
        )
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "explicit_nvidia_smi_fallback_after_api_failure",
        "api_attempt": {"url": args.api_url, "error": args.api_error},
        "servers": [
            {
                "id": socket.gethostname(), "status": "ok", "age_seconds": 0,
                "gpus": gpus, "raw_compute_processes": process_query.stdout.splitlines(),
            }
        ],
    }
    atomic_write_json(Path(args.output_json), payload)
    print(f"Saved explicit API-failure fallback snapshot for {socket.gethostname()}")


if __name__ == "__main__":
    main(sys.argv[1:])
