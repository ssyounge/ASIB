# scripts/generate_config.py

import argparse
import glob
import os
import yaml

parser = argparse.ArgumentParser(description="Merge base YAML with overrides")
parser.add_argument(
    "--base",
    required=True,
    nargs="+",
    help="One or more YAML files or a directory of YAML fragments",
)
parser.add_argument('--out', required=True, help='Output YAML file')
parser.add_argument('--hparams', help='YAML file with numeric hyperparameters')
parser.add_argument('--method', help='Distillation method name')
parser.add_argument('--verbose', action='store_true', help='Print overridden keys')
parser.add_argument('overrides', nargs='*', help='KEY=VAL pairs to override')
args = parser.parse_args()

cfg = {}
for base in args.base:
    paths = []
    if os.path.isdir(base):
        paths.extend(sorted(glob.glob(os.path.join(base, "*.yml"))))
        paths.extend(sorted(glob.glob(os.path.join(base, "*.yaml"))))
    else:
        paths.append(base)
    for p in paths:
        with open(p) as f:
            data = yaml.safe_load(f)
            if data:
                cfg.update(data)

if args.hparams:
    with open(args.hparams) as f:
        hparams = yaml.safe_load(f)
        if hparams:
            cfg.update(hparams)

if args.method:
    cfg['method'] = args.method

for ov in args.overrides:
    if '=' not in ov:
        continue
    key, val = ov.split('=', 1)
    try:
        cfg[key] = yaml.safe_load(val)
    except yaml.YAMLError:
        lower = val.lower()
        if lower in ['true', 'false']:
            cfg[key] = lower == 'true'
        else:
            try:
                cfg[key] = int(val)
            except ValueError:
                cfg[key] = val
    if args.verbose:
        print(f"{key} = {cfg[key]}")

with open(args.out, 'w') as f:
    yaml.safe_dump(cfg, f)
