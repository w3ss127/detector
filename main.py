# main.py

import wandb
import torch
from utils.config import load_config
from train.trainer import run_training


def main():
    # Load config from YAML + CLI
    config = load_config()

    # Initialize Weights & Biases
    wandb.init(
        project=config['project_name'],
        name=config['run_name'],
        config=config,
        resume="allow" if config['resume'] else None
    )

    # Run training
    run_training(config)

    wandb.finish()


if __name__ == "__main__":
    main()
