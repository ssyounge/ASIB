# utils/print_cfg.py

def print_hparams(cfg, title="Hyper-parameters"):
    """Pretty-print flattened hyperparameters in a table.

    Parameters
    ----------
    cfg : dict | Namespace | OmegaConf | yacs CfgNode
        Configuration object. Nested structures are flattened using
        dot notation.
    title : str, optional
        Table title displayed at the top. Defaults to ``"Hyper-parameters"``.
    """
    # 1) cfg to python dict -------------------------------------------------
    try:
        from omegaconf import OmegaConf  # ΩConf?
        if isinstance(cfg, (OmegaConf,)):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ModuleNotFoundError:
        pass
    try:
        from yacs.config import CfgNode  # yacs?
        if isinstance(cfg, CfgNode):
            cfg = cfg.clone().freeze(False)
            cfg = cfg.dump() if isinstance(cfg, str) else dict(cfg)
    except ModuleNotFoundError:
        pass
    if not isinstance(cfg, dict):
        cfg = vars(cfg)  # Namespace -> dict

    # 2) flatten ------------------------------------------------------------
    flat = {}

    def _walk(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                _walk(v, key + ".")
            else:
                flat[key] = v

    _walk(cfg)

    # 3) pretty print -------------------------------------------------------
    keys, vals = zip(*sorted(flat.items())) if flat else ([], [])
    k_width = max((len(k) for k in keys), default=0) + 2
    v_width = max((len(str(v)) for v in vals), default=0) + 2
    bar = "┌" + "─" * (k_width + v_width + 1) + "┐"
    print(bar)
    print(f"│ {title} ({len(keys)})".ljust(k_width + v_width + 2) + "│")
    print("├" + "─" * k_width + "┬" + "─" * v_width + "┤")
    for k in keys:
        v = flat[k]
        print(f"│ {k.ljust(k_width-1)}│ {str(v).ljust(v_width-1)}│")
    print("└" + "─" * k_width + "┴" + "─" * v_width + "┘")
