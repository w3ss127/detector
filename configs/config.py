# configs/config.py

import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def load_config():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    parser.add_argument('--run_name', type=str, help='Optional override of run name')
    parser.add_argument('--gpus', type=int, help='Override number of GPUs')
    args, _ = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    if args.run_name:
        config_dict['run_name'] = args.run_name
    if args.gpus is not None:
        config_dict['gpus'] = args.gpus

    config = dict_to_namespace(config_dict)
    return config


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
