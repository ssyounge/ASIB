import utils.config_utils as cu


def test_log_options_propagated():
    cfg = {"log": {"level": "WARNING", "filename": "out.log"}}
    out = cu.flatten_hydra_config(dict(cfg))
    assert out["log_level"] == "WARNING"
    assert out["log_filename"] == "out.log"


def test_existing_top_level_not_overridden():
    cfg = {
        "log": {"level": "DEBUG", "filename": "nested.log"},
        "log_level": "INFO",
    }
    out = cu.flatten_hydra_config(dict(cfg))
    assert out["log_level"] == "INFO"
    assert out["log_filename"] == "nested.log"

