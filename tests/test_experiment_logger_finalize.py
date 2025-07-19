import os
import sys
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.logger import ExperimentLogger


def test_finalize_creates_missing_dir(tmp_path):
    results_dir = tmp_path / "nested" / "results"
    cfg = {"results_dir": str(results_dir), "eval_mode": "test"}
    logger = ExperimentLogger(cfg)
    logger.update_metric("train_acc", 0.0)
    exp_id = logger.exp_id
    logger.finalize()

    json_path = results_dir / f"{exp_id}.json"
    csv_path = results_dir / "summary.csv"
    latest_path = results_dir / "latest.json"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert latest_path.exists()

    # sanity check that files are readable
    with open(json_path) as f:
        data = json.load(f)
        assert data["exp_id"]

    with open(csv_path) as f:
        rows = list(csv.reader(f))
        assert len(rows) >= 2
