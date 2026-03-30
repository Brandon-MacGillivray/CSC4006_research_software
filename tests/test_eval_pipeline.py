import pytest

pytest.importorskip("torch")

from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES
from handpose.evaluation.eval_pipeline import resolve_eval_indices, resolve_root_keypoint_local_index


def test_resolve_eval_indices_full_layout_without_shared10():
    positions, eval_keypoints = resolve_eval_indices(list(range(21)), shared_10_eval=False)
    assert positions == list(range(21))
    assert eval_keypoints == list(range(21))


def test_resolve_eval_indices_shared10_from_full_layout():
    positions, eval_keypoints = resolve_eval_indices(list(range(21)), shared_10_eval=True)
    assert positions == list(TIP_BASE_KEYPOINT_INDICES)
    assert eval_keypoints == list(TIP_BASE_KEYPOINT_INDICES)


def test_resolve_eval_indices_shared10_rejects_missing_keypoint():
    model_keypoints = [kp for kp in range(21) if kp != 20]
    with pytest.raises(ValueError, match="Missing: \\[20\\]"):
        resolve_eval_indices(model_keypoints, shared_10_eval=True)


def test_resolve_root_keypoint_prefers_wrist_when_present():
    root_index = resolve_root_keypoint_local_index(
        eval_keypoint_indices=[0, 4, 8],
        model_keypoint_indices=list(range(21)),
    )
    assert root_index == 0


def test_resolve_root_keypoint_uses_keypoint_one_for_tip_base_model(capsys):
    root_index = resolve_root_keypoint_local_index(
        eval_keypoint_indices=list(TIP_BASE_KEYPOINT_INDICES),
        model_keypoint_indices=list(TIP_BASE_KEYPOINT_INDICES),
    )
    captured = capsys.readouterr()
    assert root_index == 0
    assert "using keypoint 1 as EPE root for tip/base 10-joint model" in captured.err


def test_resolve_root_keypoint_returns_none_without_supported_root(capsys):
    root_index = resolve_root_keypoint_local_index(
        eval_keypoint_indices=[2, 3, 4],
        model_keypoint_indices=list(range(21)),
    )
    captured = capsys.readouterr()
    assert root_index is None
    assert "no supported EPE root in eval set" in captured.err
