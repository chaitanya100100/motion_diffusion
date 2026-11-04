"""Long-sequence conditional motion generation from a head trajectory.

Companion to ``infer_motion_from_traj_sp.py``, which is limited to the model's
fixed ``cfg.DATA.WINDOW``. This one drives ``modiff.module.long_sample`` and can
run the joint windowed denoising as well as the two baselines.

Self-tests (geometry, no checkpoint needed beyond AMASS for the second one)::

    python -m modiff.run.infer_long_motion_from_traj --self-test

Generate from a trajectory file (``T x 4 x 4`` .npy / .pt) and render a video::

    python -m modiff.run.infer_long_motion_from_traj \
        --exp-path ./exp/amass_v1_ait_unet/ --traj /path/to/aria_traj_T.npy --visualize

Compare all three modes against a ground-truth AMASS validation sequence::

    python -m modiff.run.infer_long_motion_from_traj \
        --exp-path ./exp/amass_v1_ait_unet/ --amass-eval --n-frames 300 --mode all
"""

import argparse
import os

import numpy as np
import torch
from loguru import logger

from modiff.dataset.canonicalization import get_a_canonicalized_segment, rot_trans_to_matrix
from modiff.dataset.motion_transforms import (
    CONTACT_END,
    HEAD_END,
    align_se3_trajectories,
    canonicalizing_transform,
    head_T_from_v1_ait,
    transform_v1_ait_motion,
)
from modiff.dataset.representation_utils import repre_to_sequence, sequence_to_repre
from modiff.dataset.smpl_utils import evaluate_smpl, get_smpl
from modiff.module.uem_module import UEM_Module

MODES = ["multidiffusion", "autoregressive", "naive"]
AMASS_VAL_DATASETS = ["Transitions_mocap", "Transitions", "HumanEva", "MPI_HDM05", "HDM05", "SFU", "MPI_mosh", "MoSh"]


# ------------------------------------------------------------------------------------------------
# Self tests
# ------------------------------------------------------------------------------------------------
def _random_motion(n_win, n_frames, seed=0):
    """A synthetic v1_ait tensor whose 6D channels hold valid rotations."""
    import modiff.utils.rotation_conversions as rc

    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n_win, n_frames, 23, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    rot6d = rc.matrix_to_rotation_6d(rc.quaternion_to_matrix(q))
    pos = torch.randn(n_win, n_frames, 23, 3, generator=g)
    joints = torch.cat([rot6d[:, :, :22], pos[:, :, :22]], -1).reshape(n_win, n_frames, 198)
    head = torch.cat([rot6d[:, :, 22], pos[:, :, 22]], -1)
    contacts = torch.rand(n_win, n_frames, 2, generator=g)
    return torch.cat([joints, head, contacts], -1)


def _random_se2(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, generator=g) * (2 * np.pi)
    R = torch.zeros(n, 3, 3)
    R[:, 0, 0], R[:, 0, 1] = theta.cos(), -theta.sin()
    R[:, 1, 0], R[:, 1, 1] = theta.sin(), theta.cos()
    R[:, 2, 2] = 1.0
    t = torch.cat([torch.randn(n, 2, generator=g), torch.zeros(n, 1)], -1)
    return rot_trans_to_matrix(R, t)


def self_test_transforms(tol=1e-5):
    """Round-trip, composition, and agreement with canonicalize_trajectory."""
    from modiff.dataset.canonicalization import canonicalize_trajectory

    x = _random_motion(3, 7)
    G, G2 = _random_se2(3, seed=1), _random_se2(3, seed=2)

    err = (transform_v1_ait_motion(transform_v1_ait_motion(x, G[:, None]), torch.inverse(G)[:, None]) - x).abs().max()
    assert err < tol, f"round-trip error {err}"
    logger.info(f"[transforms] round-trip                     max err {err:.2e}")

    a = transform_v1_ait_motion(x, (G @ G2)[:, None])
    b = transform_v1_ait_motion(transform_v1_ait_motion(x, G2[:, None]), G[:, None])
    err = (a - b).abs().max()
    assert err < tol, f"composition error {err}"
    logger.info(f"[transforms] composition                    max err {err:.2e}")

    head = head_T_from_v1_ait(x[0])
    err = (canonicalizing_transform(head[0])[None] @ head - canonicalize_trajectory(head, None, None)).abs().max()
    assert err < tol, f"canonicalization mismatch {err}"
    logger.info(f"[transforms] matches canonicalize_trajectory max err {err:.2e}")

    err = (canonicalizing_transform(canonicalize_trajectory(head, None, None)[0]) - torch.eye(4)).abs().max()
    assert err < tol, f"frame 0 transform is not identity: {err}"
    logger.info(f"[transforms] frame-0 transform is identity   max err {err:.2e}")


def self_test_window_frames(motion_data=None, n_frames=300, offset=60, window=80, tol=1e-3):
    """The load-bearing check: canonicalizing at frame k equals transforming frame 0's window.

    Builds the same underlying AMASS motion twice -- once canonicalized at frame 0,
    once at frame ``offset`` -- and asserts that the analytic window transform maps
    one onto the other.
    """
    smpl = get_smpl()
    if motion_data is None:
        motion_data = load_amass_val()
    seq_name = pick_amass_sequence(motion_data, n_frames, require_floor=True)
    seq = motion_data[seq_name]
    logger.info(f"[frames] using {seq_name} ({seq['num_frames']} frames, floor {seq['floor_height']:.4f})")

    def repre_from(start, length):
        seg = get_a_canonicalized_segment(
            seq["smpl_params"], seq["aria_traj"], seq["kp3d"], smpl, start, start + length - 1
        )
        motion, _ = sequence_to_repre(
            "v1_ait", seg["can_aria_traj"], seg["can_smpl_params"], seg["can_kp3d"], seq["floor_height"], smpl
        )
        return motion

    x_a = repre_from(0, n_frames)  # canonicalized at frame 0
    x_b = repre_from(offset, window)  # canonicalized at frame `offset`

    G = canonicalizing_transform(head_T_from_v1_ait(x_a)[offset])
    x_a_win = transform_v1_ait_motion(x_a[offset : offset + window], G)

    err = (x_a_win[:, :HEAD_END] - x_b[:, :HEAD_END]).abs().max()
    logger.info(f"[frames] joint + head channels              max err {err:.2e}")
    assert err < tol, f"window frame mismatch {err}"

    # foot_detect duplicates the first frame's velocity, so frame 0's contact flags
    # are computed from different data in the two segments; compare from frame 1.
    err_c = (x_a_win[1:, HEAD_END:CONTACT_END] - x_b[1:, HEAD_END:CONTACT_END]).abs().max()
    logger.info(f"[frames] contact channels (frames 1..)      max err {err_c:.2e}")
    assert err_c < tol, f"contact mismatch {err_c}"

    G0 = canonicalizing_transform(head_T_from_v1_ait(x_a)[0])
    assert (G0 - torch.eye(4)).abs().max() < tol
    logger.info("[frames] window 0 transform is identity     ok")
    return motion_data


# ------------------------------------------------------------------------------------------------
# AMASS helpers
# ------------------------------------------------------------------------------------------------
def load_amass_val(path=None):
    from modiff.config.defaults import get_cfg_defaults

    path = path or get_cfg_defaults().DATA.AMASS_DATA_PATH
    logger.info(f"Loading AMASS from {path} (this takes a while)")
    data = torch.load(path, weights_only=False)
    return {k: v for k, v in data.items() if k.split("___")[0] in AMASS_VAL_DATASETS}


def pick_amass_sequence(motion_data, n_frames, require_floor=False):
    for name in sorted(motion_data):
        seq = motion_data[name]
        if seq["num_frames"] < n_frames:
            continue
        if require_floor and abs(float(seq["floor_height"])) < 1e-8:
            continue
        return name
    raise RuntimeError(f"no validation sequence with >= {n_frames} frames")


def amass_ground_truth(motion_data, seq_name, n_frames, smpl):
    """GT ``v1_ait`` motion and head trajectory for the first ``n_frames`` of a sequence."""
    seq = motion_data[seq_name]
    seg = get_a_canonicalized_segment(seq["smpl_params"], seq["aria_traj"], seq["kp3d"], smpl, 0, n_frames - 1)
    motion, _ = sequence_to_repre(
        "v1_ait", seg["can_aria_traj"], seg["can_smpl_params"], seg["can_kp3d"], seq["floor_height"], smpl
    )
    return motion, seg["can_aria_traj"]


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
def head_traj_error(motion, traj_T):
    """Position / rotation error between the generated head and the input trajectory, post-alignment."""
    pred = head_T_from_v1_ait(motion).cpu().float()
    traj_T = traj_T.cpu().float()
    A, _, _, _ = align_se3_trajectories(pred, traj_T)
    aligned = A[None] @ traj_T
    pos = (aligned[:, :3, 3] - pred[:, :3, 3]).norm(dim=-1).mean()
    rel = aligned[:, :3, :3].transpose(1, 2) @ pred[:, :3, :3]
    cos = ((rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]) - 1) / 2
    rot = cos.clamp(-1, 1).arccos().mean() * 180 / np.pi
    return pos.item(), rot.item()


def foot_skate(j3d, contact, height_thresh=0.10):
    """Mean horizontal displacement of a foot joint while it is flagged in contact."""
    total, count = 0.0, 0
    for joint, ch in ((10, 0), (11, 1)):
        on = (contact[:-1, ch] > 0.5) & (j3d[:-1, joint, 2] < height_thresh)
        if on.any():
            disp = (j3d[1:, joint, :2] - j3d[:-1, joint, :2]).norm(dim=-1)
            total += disp[on].sum().item()
            count += int(on.sum())
    return total / max(count, 1)


def seam_stats(j3d, starts, window):
    """Joint-velocity magnitude at window boundaries vs everywhere else."""
    vel = (j3d[1:] - j3d[:-1]).norm(dim=-1).mean(-1)  # T-1
    boundary = torch.zeros_like(vel, dtype=torch.bool)
    for s in starts[1:]:
        for i in (s - 1, s):
            if 0 <= i < vel.shape[0]:
                boundary[i] = True
    if not boundary.any():
        return vel.mean().item(), float("nan")
    return vel[~boundary].mean().item(), vel[boundary].mean().item()


def evaluate(motion, traj_T, info, smpl, window, label, gt_motion=None):
    _, smpl_params, j3d = repre_to_sequence("v1_ait", motion.cpu().float(), None, smpl, None, None)
    contact = motion[:, HEAD_END:CONTACT_END].cpu().float()
    pos_err, rot_err = head_traj_error(motion, traj_T)
    interior, boundary = seam_stats(j3d, info.get("starts", [0]), window)

    row = {
        "mode": label,
        "head_pos_err_m": pos_err,
        "head_rot_err_deg": rot_err,
        "foot_skate_m": foot_skate(j3d, contact),
        "vel_interior": interior,
        "vel_at_seams": boundary,
    }
    if gt_motion is not None:
        _, _, gt_j3d = repre_to_sequence("v1_ait", gt_motion.cpu().float(), None, smpl, None, None)
        gt_j3d = gt_j3d - gt_j3d[:, :1] + j3d[:, :1]  # root-relative pose comparison
        row["mpjpe_root_rel_m"] = (gt_j3d - j3d).norm(dim=-1).mean().item()
        row["gt_foot_skate_m"] = foot_skate(gt_j3d, gt_motion[:, HEAD_END:CONTACT_END].cpu().float())
    return row, smpl_params, j3d


def print_table(rows):
    keys = [k for k in rows[0] if k != "mode"]
    header = f"{'mode':<16}" + "".join(f"{k:>20}" for k in keys)
    logger.info(header)
    logger.info("-" * len(header))
    for r in rows:
        logger.info(f"{r['mode']:<16}" + "".join(f"{r.get(k, float('nan')):>20.4f}" for k in keys))


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
def load_trajectory(path, n_frames=None):
    if path.endswith(".npy"):
        traj = torch.from_numpy(np.load(path))
    else:
        traj = torch.load(path, weights_only=False)
        if isinstance(traj, dict):
            traj = traj["aria_traj_T"]
    traj = torch.as_tensor(traj).float()
    return traj[:n_frames] if n_frames else traj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="run geometry checks and exit")
    parser.add_argument("--exp-path", default="./exp/amass_v1_ait_unet/")
    parser.add_argument("--traj", default=None, help="T x 4 x 4 head poses (.npy or .pt)")
    parser.add_argument("--amass-eval", action="store_true", help="use an AMASS val sequence and score against GT")
    parser.add_argument("--seq-name", default=None)
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument("--mode", default="multidiffusion", choices=MODES + ["all"])
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--taper", type=int, default=None)
    parser.add_argument("--n-passes", type=int, default=2)
    parser.add_argument("--frame-init", default="window0", choices=["window0", "autoregressive"])
    parser.add_argument("--cond-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.self_test:
        self_test_transforms()
        self_test_window_frames(n_frames=args.n_frames)
        logger.info("all self-tests passed")
        return

    smpl = get_smpl()
    model, cfg = UEM_Module.from_pretrained(args.exp_path)
    model = model.to(args.device).eval()
    window = cfg.DATA.WINDOW

    gt_motion, motion_data = None, None
    if args.amass_eval:
        motion_data = load_amass_val()
        seq_name = args.seq_name or pick_amass_sequence(motion_data, args.n_frames, require_floor=True)
        gt_motion, traj_T = amass_ground_truth(motion_data, seq_name, args.n_frames, smpl)
        logger.info(f"evaluating on {seq_name}, {args.n_frames} frames")
    else:
        assert args.traj is not None, "pass --traj or --amass-eval"
        traj_T = load_trajectory(args.traj, args.n_frames)
        seq_name = os.path.splitext(os.path.basename(args.traj))[0]
    logger.info(f"trajectory: {tuple(traj_T.shape)}, window {window}, stride {args.stride}")

    modes = MODES if args.mode == "all" else [args.mode]
    kwargs = {"multidiffusion": dict(stride=args.stride, taper=args.taper, n_passes=args.n_passes,
                                     frame_init=args.frame_init),
              "autoregressive": dict(stride=window // 2),
              "naive": dict()}

    rows, results = [], {}
    for mode in modes:
        logger.info(f"sampling with mode={mode}")
        motion, info = model.sample_long(
            traj_T, mode=mode, seed=args.seed, cond_scale=args.cond_scale, progress=True, **kwargs[mode]
        )
        motion = torch.nan_to_num(motion)
        row, smpl_params, j3d = evaluate(motion, traj_T, info, smpl, window, mode, gt_motion)
        rows.append(row)
        results[mode] = {"motion": motion.cpu(), "smpl_params": {k: v.cpu() for k, v in smpl_params.items()},
                         "starts": info.get("starts", [0])}

    print_table(rows)

    out_dir = args.out_dir or os.path.join(args.exp_path, f"long_{seq_name.replace('/', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"preds_T{args.n_frames}.pt")
    torch.save({"seq_name": seq_name, "traj_T": traj_T, "metrics": rows, "results": results}, out_path)
    logger.info(f"saved predictions to {out_path}")

    if args.visualize:
        from modiff.utils.vis_utils import save_video, visualize_sequence

        for mode, res in results.items():
            _, verts, _ = evaluate_smpl(smpl, res["smpl_params"])
            # Same convention as infer_motion_from_traj_sp.py: bring the input
            # trajectory into the prediction's frame rather than the other way round.
            A, _, _, _ = align_se3_trajectories(head_T_from_v1_ait(res["motion"]), traj_T)
            imgs = visualize_sequence(aria_traj=A[None] @ traj_T, verts=verts, faces=smpl.faces)
            save_video(imgs[..., ::-1], f"{mode}_T{args.n_frames}", out_dir, fps=10)
            logger.info(f"rendered {mode}")


if __name__ == "__main__":
    main()
