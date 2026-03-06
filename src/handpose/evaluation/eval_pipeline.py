import sys

import torch
from torch.utils.data import DataLoader

from handpose.data.dataset import RHDDatasetCoords
from handpose.models.hand_pose_model import HandPoseNet
from handpose.data.keypoints import TIP_BASE_KEYPOINT_INDICES


def resolve_device(device_arg: str):
    """Resolve requested device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def resolve_eval_indices(model_keypoint_indices, shared_10_eval: bool):
    """Pick eval keypoints and map to model output positions."""
    model_keypoint_indices = [int(x) for x in model_keypoint_indices]

    if not shared_10_eval:
        # Default: evaluate all model keypoints.
        return list(range(len(model_keypoint_indices))), list(model_keypoint_indices)

    # Fair comparison mode: evaluate only the shared 10.
    eval_keypoints = list(TIP_BASE_KEYPOINT_INDICES)
    eval_positions = []
    missing = []
    for kp in eval_keypoints:
        try:
            eval_positions.append(model_keypoint_indices.index(kp))
        except ValueError:
            missing.append(kp)

    if missing:
        raise ValueError(
            "Checkpoint keypoint set does not contain all shared-10 keypoints. "
            f"Missing: {missing}. Checkpoint keypoints: {model_keypoint_indices}"
        )

    return eval_positions, eval_keypoints


def resolve_root_keypoint_local_index(eval_keypoint_indices, model_keypoint_indices):
    """Choose the root keypoint index used for root-relative EPE."""
    tip_base_set = set(int(x) for x in TIP_BASE_KEYPOINT_INDICES)
    model_keypoint_set = set(int(x) for x in model_keypoint_indices)
    is_tip_base_10_model = (
        len(model_keypoint_indices) == len(TIP_BASE_KEYPOINT_INDICES)
        and model_keypoint_set == tip_base_set
    )

    if 0 in eval_keypoint_indices:
        return eval_keypoint_indices.index(0)
    if is_tip_base_10_model and 1 in eval_keypoint_indices:
        print(
            "[eval] using keypoint 1 as EPE root for tip/base 10-joint model.",
            file=sys.stderr,
        )
        return eval_keypoint_indices.index(1)

    print(
        "[eval] no supported EPE root in eval set; epe_norm will be null.",
        file=sys.stderr,
    )
    return None


def build_model(num_keypoints: int, state_dict: dict, device: torch.device):
    """Build model and load checkpoint weights."""
    model = HandPoseNet(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(state_dict)
    return model


def build_loader(args, device: torch.device, model_keypoint_indices):
    """Build evaluation DataLoader with checkpoint keypoint order."""
    ds = RHDDatasetCoords(
        args.root,
        split=args.split,
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
        keypoint_indices=model_keypoint_indices,
        return_visibility=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return loader
