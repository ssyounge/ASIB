# tests/test_snapshot_teacher.py

import pytest; pytest.importorskip("torch")

from models.ensemble import SnapshotTeacher


def test_snapshot_teacher_missing_ckpt(tmp_path):
    missing = tmp_path / "foo.pt"
    with pytest.raises(FileNotFoundError) as exc:
        SnapshotTeacher([str(missing)])
    # check message mentions path and env variable
    msg = str(exc.value)
    assert str(missing) in msg
    assert "ASMB_KD_ROOT" in msg
