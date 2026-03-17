import sys
import time

import torch

from handpose.inference.predict import build_fusion_context, predict_all_modes, predict_coords


@torch.no_grad()
def evaluate_checkpoint(
    model,
    loader,
    device,
    model_keypoint_indices,
    eval_positions,
    root_keypoint_local_index,
    pck_threshold: float,
    prediction_mode: str = "fusion",
    debug_coords: bool = False,
    with_fusion_diagnostics: bool = False,
):
    """Evaluate fused-coordinate predictions with SSE/EPE/PCK metrics."""
    model.eval()
    index_tensor = None
    if eval_positions is not None:
        # Prebuild index tensor for shared-10 slicing.
        index_tensor = torch.tensor(eval_positions, dtype=torch.long, device=device)

    fusion_context = build_fusion_context(model_keypoint_indices=model_keypoint_indices)

    # Running totals.
    total_samples = 0
    total_points = 0
    total_visible_points = 0
    total_sse = 0.0
    total_epe = 0.0
    total_epe_points = 0
    total_pck_hits = 0
    total_pck_points = 0

    prediction_seconds = 0.0
    num_batches = 0
    debug_printed = False

    total_fusion_points = 0
    total_heatmap_selected = 0.0
    total_coord_selected = 0.0
    total_alpha_sum = 0.0
    total_alpha_count = 0
    all_alpha_values = []
    total_disagreement_sum = 0.0
    total_disagreement_count = 0
    all_disagreement_values = []
    total_fusion_matches_oracle = 0
    total_fusion_oracle_points = 0
    per_joint_heatmap_selected = None
    per_joint_total = None

    for imgs, coords, vis in loader:
        # Move batch to device.
        imgs = imgs.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)

        # Time canonical inference path (model forward + fusion).
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pred_coords = predict_coords(
            model=model,
            x=imgs,
            fusion_context=fusion_context,
            prediction_mode=prediction_mode,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_seconds += time.perf_counter() - t0
        num_batches += 1

        diag_outputs = None
        if with_fusion_diagnostics:
            diag_outputs = predict_all_modes(
                model=model,
                x=imgs,
                fusion_context=fusion_context,
            )

        if index_tensor is not None:
            # Restrict to eval subset.
            coords = coords.index_select(1, index_tensor)
            pred_coords = pred_coords.index_select(1, index_tensor)
            vis = vis.index_select(1, index_tensor)

        if diag_outputs is not None:
            hm_coords = diag_outputs["heatmap"]
            coord_coords = diag_outputs["coord"]
            use_heatmap = diag_outputs["use_heatmap"]
            disagreement = diag_outputs["d"]
            alpha = diag_outputs["alpha"]
            if index_tensor is not None:
                hm_coords = hm_coords.index_select(1, index_tensor)
                coord_coords = coord_coords.index_select(1, index_tensor)
                use_heatmap = use_heatmap.index_select(1, index_tensor)
                disagreement = disagreement.index_select(1, index_tensor)
            if per_joint_heatmap_selected is None:
                per_joint_heatmap_selected = torch.zeros(
                    hm_coords.shape[1], dtype=torch.float64
                )
                per_joint_total = torch.zeros(
                    hm_coords.shape[1], dtype=torch.float64
                )

        # Visible-joint masked SSE/PCK.
        diff = pred_coords - coords
        sq_per_joint = (diff * diff).sum(dim=-1)
        l2_per_joint = torch.sqrt(sq_per_joint)
        visible_mask = (vis > 0).float()
        visible_mask_bool = vis > 0
        sse_per_sample = (sq_per_joint * visible_mask).sum(dim=-1)

        # PCK@sigma on visible joints.
        pck_hits = ((l2_per_joint <= pck_threshold).float() * visible_mask).sum()
        pck_points = visible_mask.sum()

        # Root-relative EPE (normalized coords).
        if root_keypoint_local_index is not None:
            root_gt = coords[:, root_keypoint_local_index : root_keypoint_local_index + 1, :]
            root_pred = pred_coords[:, root_keypoint_local_index : root_keypoint_local_index + 1, :]
            rel_diff = (pred_coords - root_pred) - (coords - root_gt)
            rel_l2 = torch.sqrt((rel_diff * rel_diff).sum(dim=-1))
            total_epe += float((rel_l2 * visible_mask).sum().item())
            total_epe_points += int(visible_mask.sum().item())

        if diag_outputs is not None:
            hm_selected = use_heatmap.float()
            coord_selected = 1.0 - hm_selected
            total_heatmap_selected += float(hm_selected.sum().item())
            total_coord_selected += float(coord_selected.sum().item())
            total_fusion_points += int(use_heatmap.numel())
            per_joint_heatmap_selected += hm_selected.sum(dim=0).detach().cpu().to(torch.float64)
            per_joint_total += torch.full(
                (use_heatmap.shape[1],),
                float(use_heatmap.shape[0]),
                dtype=torch.float64,
            )

            total_alpha_sum += float(alpha.sum().item())
            total_alpha_count += int(alpha.numel())
            all_alpha_values.extend(alpha.detach().cpu().view(-1).tolist())

            total_disagreement_sum += float(disagreement.sum().item())
            total_disagreement_count += int(disagreement.numel())
            all_disagreement_values.extend(disagreement.detach().cpu().view(-1).tolist())

            hm_err = torch.sqrt(((hm_coords - coords) * (hm_coords - coords)).sum(dim=-1))
            coord_err = torch.sqrt(((coord_coords - coords) * (coord_coords - coords)).sum(dim=-1))
            oracle_prefers_heatmap = hm_err <= coord_err
            oracle_match = (use_heatmap == oracle_prefers_heatmap) & visible_mask_bool
            total_fusion_matches_oracle += int(oracle_match.sum().item())
            total_fusion_oracle_points += int(visible_mask_bool.sum().item())

        if debug_coords and not debug_printed:
            # One-time sample dump for sanity-checking.
            n_show = min(5, int(coords.shape[1]))
            gt0 = coords[0, :n_show].detach().cpu()
            pred0 = pred_coords[0, :n_show].detach().cpu()
            diff0 = diff[0, :n_show].detach().cpu()
            l20 = l2_per_joint[0, :n_show].detach().cpu()
            vis0 = vis[0, :n_show].detach().cpu()
            print("[debug-coords] showing first sample, first", n_show, "joints", file=sys.stderr)
            print("[debug-coords] gt[0,:n]   =", gt0, file=sys.stderr)
            print("[debug-coords] pred[0,:n] =", pred0, file=sys.stderr)
            print("[debug-coords] diff[0,:n] =", diff0, file=sys.stderr)
            print("[debug-coords] l2[0,:n]   =", l20, file=sys.stderr)
            print("[debug-coords] vis[0,:n]  =", vis0, file=sys.stderr)
            print(
                "[debug-coords] gt range / pred range =",
                (float(coords.min().item()), float(coords.max().item())),
                (float(pred_coords.min().item()), float(pred_coords.max().item())),
                file=sys.stderr,
            )
            print(
                "[debug-coords] mean l2 / max l2 =",
                float(l2_per_joint.mean().item()),
                float(l2_per_joint.max().item()),
                file=sys.stderr,
            )
            debug_printed = True

        # Update totals.
        batch_size = int(imgs.shape[0])
        total_samples += batch_size
        total_points += int(coords.shape[0] * coords.shape[1])
        total_visible_points += int(visible_mask.sum().item())
        total_sse += float(sse_per_sample.sum().item())
        total_pck_hits += int(pck_hits.item())
        total_pck_points += int(pck_points.item())

    if total_samples == 0 or total_points == 0:
        raise RuntimeError("No samples were evaluated")

    # Final normalized metrics.
    results = {
        "num_samples": total_samples,
        "num_points": total_points,
        "num_visible_points": total_visible_points,
        "num_eval_keypoints": total_points // total_samples,
        "sse_norm": (total_sse / total_samples),
        "epe_norm": (total_epe / total_epe_points) if total_epe_points > 0 else None,
        "epe_root_keypoint_index_in_eval": root_keypoint_local_index,
        "pck_threshold": float(pck_threshold),
        "pck": (float(total_pck_hits) / float(total_pck_points)) if total_pck_points > 0 else None,
        "timing": {
            "prediction_seconds": prediction_seconds,
            "num_batches": num_batches,
            "ms_per_image": (prediction_seconds * 1000.0 / total_samples),
            "images_per_second": (total_samples / prediction_seconds) if prediction_seconds > 0 else None,
        },
    }
    if with_fusion_diagnostics:
        per_joint_heatmap_rate = []
        per_joint_coord_rate = []
        if per_joint_heatmap_selected is not None and per_joint_total is not None:
            for hm_count, total_count in zip(per_joint_heatmap_selected.tolist(), per_joint_total.tolist()):
                if total_count > 0:
                    hm_rate = float(hm_count) / float(total_count)
                    coord_rate = 1.0 - hm_rate
                else:
                    hm_rate = None
                    coord_rate = None
                per_joint_heatmap_rate.append(hm_rate)
                per_joint_coord_rate.append(coord_rate)

        alpha_sorted = sorted(float(x) for x in all_alpha_values)
        disagreement_sorted = sorted(float(x) for x in all_disagreement_values)
        alpha_median = None
        disagreement_median = None
        if alpha_sorted:
            alpha_median = alpha_sorted[len(alpha_sorted) // 2]
        if disagreement_sorted:
            disagreement_median = disagreement_sorted[len(disagreement_sorted) // 2]

        results["fusion_diagnostics"] = {
            "enabled": True,
            "heatmap_selection_rate": (
                float(total_heatmap_selected) / float(total_fusion_points)
            ) if total_fusion_points > 0 else None,
            "coord_selection_rate": (
                float(total_coord_selected) / float(total_fusion_points)
            ) if total_fusion_points > 0 else None,
            "per_joint_heatmap_selection_rate": per_joint_heatmap_rate,
            "per_joint_coord_selection_rate": per_joint_coord_rate,
            "alpha_mean": (
                float(total_alpha_sum) / float(total_alpha_count)
            ) if total_alpha_count > 0 else None,
            "alpha_median": alpha_median,
            "disagreement_mean": (
                float(total_disagreement_sum) / float(total_disagreement_count)
            ) if total_disagreement_count > 0 else None,
            "disagreement_median": disagreement_median,
            "fusion_matches_lower_error_branch_rate": (
                float(total_fusion_matches_oracle) / float(total_fusion_oracle_points)
            ) if total_fusion_oracle_points > 0 else None,
        }
    return results
