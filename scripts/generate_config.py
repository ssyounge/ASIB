# scripts/generate_config.py

import argparse, yaml

parser = argparse.ArgumentParser(description="Merge base YAML with overrides")
parser.add_argument('--base', required=True, help='Base YAML file')
parser.add_argument('--out', required=True, help='Output YAML file')
parser.add_argument('overrides', nargs='*', help='KEY=VAL pairs to override')
args = parser.parse_args()

with open(args.base) as f:
    cfg = yaml.safe_load(f)

for ov in args.overrides:
    if '=' not in ov:
        continue
    key, val = ov.split('=', 1)
    try:
        cfg[key] = yaml.safe_load(val)
    except yaml.YAMLError:
        cfg[key] = val

with open(args.out, 'w') as f:
    yaml.safe_dump(cfg, f)
