import pytest

torch = pytest.importorskip("torch")

from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES
from handpose.inference.fusion import (
    fuse_coords,
    heatmaps_to_coords_argmax,
    resolve_fusion_bone_edges,
)


def test_heatmaps_to_coords_argmax_decodes_peak_location():
    heatmaps = torch.zeros((1, 1, 4, 5), dtype=torch.float32)
    heatmaps[0, 0, 2, 3] = 1.0
    coords = heatmaps_to_coords_argmax(heatmaps)
    assert torch.allclose(coords[0, 0], torch.tensor([3.0 / 4.0, 2.0 / 3.0]))


def test_resolve_fusion_bone_edges_for_full_layout():
    edges = resolve_fusion_bone_edges(list(range(21)))
    assert len(edges) == 20
    assert edges[0] == (0, 1)


def test_resolve_fusion_bone_edges_for_tip_base_layout():
    edges = resolve_fusion_bone_edges(list(TIP_BASE_KEYPOINT_INDICES))
    assert len(edges) == 5
    assert edges[0] == (0, 1)


def test_resolve_fusion_bone_edges_rejects_unsupported_layout():
    with pytest.raises(ValueError, match="Fusion supports only full 21-joint checkpoints"):
        resolve_fusion_bone_edges([0, 1, 2])


def test_fuse_coords_prefers_heatmap_when_close_and_coord_when_far():
    heatmaps = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    heatmaps[0, 0, 1, 1] = 1.0
    heatmaps[0, 1, 2, 2] = 1.0

    pred_coords = torch.tensor(
        [[[0.34, 0.34], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    fused, hm_coords, alpha, use_heatmap = fuse_coords(
        pred_heatmaps=heatmaps,
        pred_coords=pred_coords,
        bone_edges_local=[(0, 1)],
    )

    assert alpha.shape == (1,)
    assert use_heatmap.shape == (1, 2)
    assert bool(use_heatmap[0, 0].item()) is True
    assert bool(use_heatmap[0, 1].item()) is False
    assert torch.allclose(fused[0, 0], hm_coords[0, 0])
    assert torch.allclose(fused[0, 1], pred_coords[0, 1])
