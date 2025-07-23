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


def test_experiment_nested_dataset():
    cfg = {
        "experiment": {
            "dataset": {
                "dataset": {"name": "cifar100", "root": "./data"},
            }
        }
    }
    out = cu.flatten_hydra_config(dict(cfg))
    assert out["dataset_name"] == "cifar100"
    assert out["data_root"] == "./data"


def test_nested_teacher_model():
    cfg = {"model": {"teacher": {"model": {"teacher": {"lr": 0.001}}}}}
    out = cu.flatten_hydra_config(dict(cfg))
    assert out["teacher_lr"] == 0.001


def test_method_params_propagated():
    cfg = {"method": {"method": {"ce_alpha": 0.3, "kd_alpha": 0.7}}}
    out = cu.flatten_hydra_config(dict(cfg))
    assert out["ce_alpha"] == 0.3
    assert out["kd_alpha"] == 0.7
