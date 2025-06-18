#!/usr/bin/env python
"""Load hyperparameters from a YAML file and emit shell-compatible exports."""
import sys
import yaml
import shlex


def main(path: str):
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    for key, value in data.items():
        if isinstance(value, bool):
            value = int(value)
        value_str = str(value)
        quoted = shlex.quote(value_str)
        print(f"{key}={quoted}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: load_hparams.py <path_to_yaml>")
    main(sys.argv[1])
