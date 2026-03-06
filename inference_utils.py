import torch

from model import TIP_BASE_KEYPOINT_INDICES


def load_checkpoint(path: str, device: torch.device):
    obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path!r}")
    if "model_state" in obj:
        return obj, obj["model_state"]
    return {}, obj


def infer_num_keypoints_from_state(state_dict: dict):
    candidate_keys = [
        "heatmapHead.convM.net.0.weight",  # old Conv2D output head
        "heatmapHead.convM.net.weight",    # current Conv2DOut output head
    ]
    for key in candidate_keys:
        if key in state_dict:
            return int(state_dict[key].shape[0])

    for k, v in state_dict.items():
        if k.endswith("heatmapHead.convM.net.0.weight") or k.endswith("heatmapHead.convM.net.weight"):
            return int(v.shape[0])

    raise KeyError("Could not infer num_keypoints from checkpoint state_dict")


def infer_checkpoint_keypoint_indices(ckpt_meta: dict, state_dict: dict):
    if "keypoint_indices" in ckpt_meta and ckpt_meta["keypoint_indices"] is not None:
        return [int(x) for x in ckpt_meta["keypoint_indices"]]

    k = infer_num_keypoints_from_state(state_dict)
    if k == 21:
        return list(range(21))
    if k == len(TIP_BASE_KEYPOINT_INDICES):
        return list(TIP_BASE_KEYPOINT_INDICES)

    raise ValueError(
        "Checkpoint does not contain keypoint_indices metadata and num_keypoints "
        f"={k} is not one of the supported defaults (21 or {len(TIP_BASE_KEYPOINT_INDICES)})."
    )
