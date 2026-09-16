"""Generate independent-seed configs for the every-five-round Q audit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        default="configs/lemma456/cifar10_fcpc_grad_cumulative_seed42.yaml",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--output-dir", default="configs/lemma456/generated_cumulative_q"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    template = load_config(args.template)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        config = copy.deepcopy(template)
        config["seed"] = seed
        config["evaluation"]["validation_seed"] = seed + 10_000
        config["audit"]["ldp_seed"] = seed + 20_000
        run_dir = f"outputs/fcpc_grad_cumulative_seed{seed}"
        config["audit"]["output_dir"] = run_dir
        config["audit"]["checkpoint_dir"] = f"{run_dir}/unused_stream_checkpoints"
        path = output_dir / f"cifar10_fcpc_grad_cumulative_seed{seed}.json"
        path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
