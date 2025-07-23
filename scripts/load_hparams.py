#!/usr/bin/env python
# scripts/load_hparams.py
"""Load hyperparameters from a YAML file and emit shell-compatible exports."""
import sys
import yaml
import shlex
import warnings


def main(path: str):
    warnings.warn(
        "load_hparams.py is deprecated; pass config fragments directly to Hydra",
        DeprecationWarning,
    )
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    for key, value in data.items():
        key = key.lower()
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value)
        elif isinstance(value, bool):
            value = int(value)
        value_str = str(value)
        quoted = shlex.quote(value_str)
        print(f"{key}={quoted}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: load_hparams.py <path_to_yaml>")
    main(sys.argv[1])
