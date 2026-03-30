import pytest

torch = pytest.importorskip("torch")

from handpose.models.hand_pose_model import HandPoseNet


@pytest.mark.parametrize("num_keypoints", [21, 10])
def test_hand_pose_model_output_shapes(num_keypoints):
    model = HandPoseNet(num_keypoints=num_keypoints)
    model.eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        heatmaps, coords = model(x)
        heatmaps_only = model.forward_heatmap(x)

    assert heatmaps.shape == (1, num_keypoints, 64, 64)
    assert coords.shape == (1, num_keypoints, 2)
    assert heatmaps_only.shape == (1, num_keypoints, 64, 64)
