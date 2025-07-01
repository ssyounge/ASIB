import ast
import sys

# Extract parse_args function without importing full dependencies
with open('main.py', 'r') as f:
    src = f.read()
mod = ast.parse(src)
parse_src = None
for node in mod.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'parse_args':
        parse_src = ast.get_source_segment(src, node)
        break
import argparse

namespace = {'argparse': argparse}
exec(parse_src, namespace)
parse_args = namespace['parse_args']

def test_adam_beta_parse(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', '--adam_beta1', '0.8', '--adam_beta2', '0.95'])
    args = parse_args()
    assert args.adam_beta1 == 0.8
    assert args.adam_beta2 == 0.95

def test_adam_beta_default(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog'])
    args = parse_args()
    assert args.adam_beta1 is None
    assert args.adam_beta2 is None

def test_teacher_weight_decay_parse(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog', '--teacher_weight_decay', '0.005'])
    args = parse_args()
    assert args.teacher_weight_decay == 0.005

def test_teacher_weight_decay_default(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['prog'])
    args = parse_args()
    assert args.teacher_weight_decay is None
