def validate_keypoint_indices(keypoint_indices, total_keypoints=21):
    """Validate and normalize the requested keypoint index list."""
    if keypoint_indices is None:
        return list(range(total_keypoints))
    if len(keypoint_indices) == 0:
        raise ValueError("keypoint_indices must contain at least one index")
    if len(set(keypoint_indices)) != len(keypoint_indices):
        raise ValueError("keypoint_indices must be unique")
    if min(keypoint_indices) < 0 or max(keypoint_indices) >= total_keypoints:
        raise ValueError(f"keypoint_indices must be in [0, {total_keypoints - 1}]")
    return list(keypoint_indices)


def select_hand(uv_data, hand: str):
    """Select left, right, or most-visible hand landmarks from RHD annotations."""
    if hand == "left":
        return uv_data[0:21]
    if hand == "right":
        return uv_data[21:42]
    if hand == "auto":
        left = uv_data[0:21]
        right = uv_data[21:42]
        left_score = left[:, 2].sum()
        right_score = right[:, 2].sum()
        return right if right_score >= left_score else left
    raise ValueError("hand must be one of: 'left', 'right', 'auto'")


def find_uv_key(anno: dict):
    """Find the UV-coordinate annotation key in a sample dict."""
    return next(k for k in anno.keys() if str(k).lower().startswith("uv"))
