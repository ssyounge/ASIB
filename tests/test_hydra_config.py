import ast


def test_no_parse_args():
    with open('main.py') as f:
        src = f.read()
    mod = ast.parse(src)
    assert not any(isinstance(node, ast.FunctionDef) and node.name == 'parse_args' for node in mod.body)


def test_uses_hydra():
    with open('main.py') as f:
        src = f.read()
    assert '@hydra.main' in src
