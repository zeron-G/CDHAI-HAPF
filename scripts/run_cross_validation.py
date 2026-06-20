from __future__ import annotations

import argparse
import json
from pathlib import Path

from hapf.config import load_config
from hapf.training.cross_validation import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory leave-one-subject-out HAPF evaluation.")
    parser.add_argument("--data", required=True, type=Path, help="Path to record_CGM5Min.parquet.")
    parser.add_argument("--output", default=Path("runs/cross_validation"), type=Path)
    parser.add_argument("--config", default=Path("configs/sample_cgm.yaml"), type=Path)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()
    result = run_cross_validation(
        data_path=args.data,
        output_dir=args.output,
        config=load_config(args.config),
        device_name=args.device,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

