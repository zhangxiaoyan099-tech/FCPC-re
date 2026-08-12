from __future__ import annotations

from src.federated.trainer import Trainer
from src.utils.config import load_config, parse_args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    if args.dry_run:
        result = trainer.dry_run()
        for key, value in result.items():
            print(f"{key}: {value}")
        return
    result = trainer.train()
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
