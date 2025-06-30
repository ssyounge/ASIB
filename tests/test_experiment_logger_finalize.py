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
    logger.finalize()

    json_path = results_dir / "summary.json"
    csv_path = results_dir / "summary.csv"

    assert json_path.is_file()
    assert csv_path.is_file()

    # sanity check that files are readable
    with open(json_path) as f:
        data = json.load(f)
        assert data["exp_id"]

    with open(csv_path) as f:
        rows = list(csv.reader(f))
        assert len(rows) >= 2
