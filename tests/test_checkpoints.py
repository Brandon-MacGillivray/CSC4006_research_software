import pytest

pytest.importorskip("torch")

from handpose.checkpoints import CHECKPOINT_VERSION, get_training_config, validate_checkpoint


def make_checkpoint(**overrides):
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "stage": "2",
        "epoch": "3",
        "model_state": {},
        "num_keypoints": "21",
        "keypoint_indices": list(range(21)),
        "training_config": {"job_id": "demo"},
    }
    checkpoint.update(overrides)
    return checkpoint


def test_validate_checkpoint_normalizes_numeric_fields():
    ckpt = validate_checkpoint(make_checkpoint())
    assert ckpt["checkpoint_version"] == CHECKPOINT_VERSION
    assert ckpt["stage"] == 2
    assert ckpt["epoch"] == 3
    assert ckpt["num_keypoints"] == 21
    assert ckpt["keypoint_indices"] == list(range(21))


def test_validate_checkpoint_rejects_missing_required_field():
    ckpt = make_checkpoint()
    del ckpt["stage"]
    with pytest.raises(ValueError, match="Missing required checkpoint field: stage"):
        validate_checkpoint(ckpt)


def test_validate_checkpoint_rejects_inconsistent_num_keypoints():
    with pytest.raises(ValueError, match="inconsistent"):
        validate_checkpoint(make_checkpoint(num_keypoints=10))


def test_validate_checkpoint_rejects_duplicate_keypoint_indices():
    with pytest.raises(ValueError, match="must contain unique indices"):
        validate_checkpoint(make_checkpoint(keypoint_indices=[0, 1, 1]))


def test_get_training_config_returns_copy():
    cfg = get_training_config(make_checkpoint(training_config={"job_id": "copy-check"}))
    cfg["job_id"] = "mutated"
    original = get_training_config(make_checkpoint(training_config={"job_id": "copy-check"}))
    assert original["job_id"] == "copy-check"
