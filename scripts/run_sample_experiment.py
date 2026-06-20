from __future__ import annotations

import argparse
import json
from pathlib import Path

from hapf.config import load_config
from hapf.training.experiment import run_sample_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exploratory HAPF CGM experiment.")
    parser.add_argument("--data", required=True, type=Path, help="Path to record_CGM5Min.parquet.")
    parser.add_argument("--output", default=Path("runs/sample"), type=Path)
    parser.add_argument("--config", default=Path("configs/sample_cgm.yaml"), type=Path)
    parser.add_argument("--heldout-subject", default=None, help="Local subject_key override; never written to output.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()
    result = run_sample_experiment(
        data_path=args.data,
        output_dir=args.output,
        config=load_config(args.config),
        heldout_subject=args.heldout_subject,
        device_name=args.device,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

