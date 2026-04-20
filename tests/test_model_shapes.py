import pytest
import torch

from handpose.models.hand_pose_model import HandPoseNet


@pytest.mark.parametrize("num_keypoints", [21, 10])
def test_hand_pose_net_output_shapes(num_keypoints):
    model = HandPoseNet(num_keypoints=num_keypoints).eval()
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        heatmaps, coords = model(x)

    assert heatmaps.shape == (2, num_keypoints, 64, 64)
    assert coords.shape == (2, num_keypoints, 2)
    assert torch.isfinite(heatmaps).all()
    assert torch.isfinite(coords).all()
