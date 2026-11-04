import copy
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from tqdm.auto import tqdm
import glob

from modiff.config.defaults import get_cfg
from modiff.module.uem_module import UEM_Module, UEM_Module_TwoStage
from modiff.utils.torch_utils import to_device, to_tensor
from modiff.utils.vis_utils import save_video, visualize_sequence_blender, visualize_sequence
from modiff.dataset.canonicalization import canonicalize_trajectory
from modiff.dataset.smpl_utils import evaluate_smpl, get_smpl
from modiff.dataset.representation_utils import (
    repre_to_sequence,
    sequence_to_repre,
    center_aria_traj,
    get_aria_traj_T,
    smpl_params_to_global_T,
)
from modiff.utils.torch_utils import careful_collate_fn
from modiff.utils import rotation_conversions as rc


def get_traj_repre_here(aria_traj_T, repre_type):
    if repre_type == "v1":
        aria_traj_repre = torch.cat(
            [rc.matrix_to_rotation_6d(aria_traj_T[:, :3, :3]), aria_traj_T[:, :3, 3]], dim=1
        )  # T x 9
        return aria_traj_repre
    elif repre_type == "v1f":
        aria_traj_repre = torch.cat(
            [rc.matrix_to_rotation_6d(aria_traj_T[:, :3, :3]), aria_traj_T[:, :3, 3]], dim=1
        )  # T x 9

        # for v1f
        fh = aria_traj_repre[0, -1]
        aria_traj_repre[:, -1] = aria_traj_repre[:, -1] - fh
        return aria_traj_repre
    elif repre_type == "v1_ait":
        aria_traj_repre = center_aria_traj(aria_traj_T)
        aria_traj_repre = torch.cat(
            [rc.matrix_to_rotation_6d(aria_traj_repre[:, :3, :3]), aria_traj_repre[:, :3, 3]], dim=1
        )  # T x 9
        return aria_traj_repre
    else:
        raise ValueError(f"Representation type {repre_type} not supported")


def frame_switch_traj(A):
    Rz = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
    Rz = Rz.to(A.dtype).to(A.device)
    A = Rz @ A @ Rz
    return A


"""
python run/infer_uem_sp.py CONFIG ./config/uem_notraj_recon.yaml TRAIN.EXP_PATH ./exp/uem_v4b_dinov2_notraj_recon/ MODEL.CKPT_PATH last_ckpt

"""


def main():
    visualize = False
    sp_dataset_name = "egoexo4d"
    subset_type = "naive_selection"
    traj_from = "egoego_T_hn"
    traj_from = "egoego_T_gn_gt_scaled"

    device = torch.device("cuda")
    use_blender = False

    # Get config
    pl.seed_everything(62, workers=True)
    sys.argv = sys.argv + [
        "TRAIN.ONLY_VALIDATE",
        "True",
    ]
    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None
    assert os.path.exists(cfg.TRAIN.EXP_PATH)

    # Setup output
    if visualize:
        out_dir = (
            f"{cfg.TRAIN.EXP_PATH}/sp_{sp_dataset_name}_{subset_type}_trajfrom_{traj_from}_vis{cfg.TRAIN.EVAL_SUFFIX}"
        )
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_path = f"{cfg.TRAIN.EXP_PATH}/sp_{sp_dataset_name}_{subset_type}_trajfrom_{traj_from}_preds{cfg.TRAIN.EVAL_SUFFIX}.pt"

    if os.path.exists(out_path):
        logger.error(f"Predictions already exist at {out_path}. Skipping.")
        return
    logger.info(f"Predictions will be saved to {out_path}")

    # Model with checkpoint
    is_twostage = cfg.MODEL.TRAJ_CKPT_PATH is not None
    if not is_twostage:
        ckpt_path = cfg.MODEL.CKPT_PATH
        if cfg.MODEL.CKPT_PATH == "last_ckpt":
            ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")
        assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
        logger.info(f"Loading model from {ckpt_path}")
        model = UEM_Module.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu").to(device).eval()
    else:
        model = UEM_Module_TwoStage(cfg=cfg).to(device).eval()

    # ------------------------------------------------------------
    # We want to cater to these repre types
    repre_type = cfg.DATA.REPRE_TYPE
    assert repre_type in ["v1", "v1f", "v1_ait"]
    logger.warning(f"Using representation type {repre_type} on trajectory")

    if traj_from.startswith("egoego_T"):
        egoego_preds = torch.load(
            f"/vision/u/chpatel/UniEgoMotion/exp/myegoego/{sp_dataset_name}_{subset_type}_preds.pt"
        )
    # ------------------------------------------------------------

    # Dataset
    smpl = get_smpl()

    # Load sample names
    sp_data_dir = "/vision/u/chpatel/data/spatialmotion"
    fname = f"sample_names_{sp_dataset_name}_val_{subset_type}.txt"
    with open(f"{sp_data_dir}/{fname}", "r") as f:
        sp_seq_names = f.read().splitlines()

    # Process samples
    all_preds = {}
    for idx in tqdm(range(len(sp_seq_names))):
        seq_name = sp_seq_names[idx]
        T = 80
        traj = None
        # ------------------------------------------------------------
        # Load Trajectory
        # ------------------------------------------------------------
        if traj_from == "aria":
            aria_traj_T = np.load(f"{sp_data_dir}/{sp_dataset_name}/{seq_name}/aria_traj_T.npy")  # T x 4 x 4
            aria_traj_T = to_tensor(aria_traj_T)
            with open(f"{sp_data_dir}/{sp_dataset_name}/{seq_name}/floor_height.txt", "r") as f:
                floor_height = float(f.read())
            offset = torch.tensor([0, 0, floor_height]).float().to(aria_traj_T.device)
            aria_traj_T[:, :3, 3] = aria_traj_T[:, :3, 3] - offset
            aria_traj_T = canonicalize_trajectory(aria_traj_T, None, None)

        elif traj_from == "vipe":
            fpaths = glob.glob(f"{sp_data_dir}/{sp_dataset_name}/{seq_name}/vipe/pose/*_undistorted.npz")
            assert len(fpaths) == 1
            pose_data = np.load(fpaths[0])
            extrinsics_list = torch.from_numpy(pose_data["data"]).float()  # f x 4 x 4
            aria_traj_T = frame_switch_traj(extrinsics_list)
        elif traj_from in ["egoego_T_hn", "egoego_T_gn_gt_scaled"]:
            aria_traj_T = egoego_preds[seq_name][traj_from.replace("egoego_", "")]
            # nan to num
            aria_traj_T = torch.nan_to_num(aria_traj_T)
        else:
            raise ValueError(f"Trajectory from {traj_from} not supported")

        aria_traj_T = aria_traj_T.cpu()
        traj = get_traj_repre_here(aria_traj_T, repre_type)

        # ------------------------------------------------------------
        assert T == aria_traj_T.shape[0]
        sample = {
            "y": {
                "traj": traj,  # T x D
                "valid_frames": torch.ones(T, dtype=torch.long),  # T
            },
            "misc": {
                "seq_name": seq_name,
                "start_idx": 0,
                "end_idx": T - 1,
            },
            "pred": {},
        }
        pred_batch = careful_collate_fn([sample])
        pred_batch["pred"] = {}

        with torch.inference_mode():
            y = to_device(pred_batch["y"], device)

            x = model.sample(y, 1, cond_scale=cfg.TRAIN.COND_SCALE)
            if not is_twostage:
                if cfg.MODEL.LEARN_TRAJ:
                    pred_batch["pred"]["traj"] = to_device(x, "cpu")
                else:
                    pred_batch["pred"]["motion"] = to_device(x, "cpu")
            else:
                pred_batch["pred"]["traj"] = to_device(x[0], "cpu")
                pred_batch["pred"]["motion"] = to_device(x[1], "cpu")

        motion = pred_batch["pred"]["motion"][0]
        pred_aria_traj_T, pred_smpl_params, _ = repre_to_sequence(cfg.DATA.REPRE_TYPE, motion, None, smpl, None, None)

        if pred_aria_traj_T is None:
            pred_aria_traj_T = get_aria_traj_T(smpl_params_to_global_T(pred_smpl_params, smpl))

        # align input aria traj T to predictions so that we can compute camera pose metrics also.
        A, _, _, _ = align_trajectories_translation_first(pred_aria_traj_T, aria_traj_T)
        pred_aria_traj_T = A @ aria_traj_T

        if visualize:
            _, pred_verts, _ = evaluate_smpl(smpl, pred_smpl_params)
            vis_fn = visualize_sequence_blender if use_blender else visualize_sequence
            imgs = vis_fn(aria_traj=pred_aria_traj_T, verts=pred_verts, faces=smpl.faces)
            save_video(imgs[..., ::-1], f"{seq_name}", out_dir, fps=10)
        else:
            all_preds[seq_name] = {
                "aria_traj_T": pred_aria_traj_T,
                "smpl_params": pred_smpl_params,
            }

    if not visualize:
        torch.save(all_preds, out_path)
        logger.info(f"Saved predictions to {out_path}")


def align_trajectories_translation_first(X, Y, alpha=0.1, with_scale=False):
    """
    Estimate of A (4x4) such that A @ Y_t ≈ X_t for t=0..T-1,.

    X, Y: (T, 4, 4)
    alpha: Weight for rotation consistency. alpha = 0 means only use translation.
    """
    T = X.shape[0]
    device, dtype = X.device, X.dtype

    # 1. Extract Components
    t_x = X[:, :3, 3]  # (T, 3)
    t_y = Y[:, :3, 3]  # (T, 3)
    R_x = X[:, :3, :3]  # (T, 3, 3)
    R_y = Y[:, :3, :3]  # (T, 3, 3)

    mu_x = t_x.mean(dim=0)
    mu_y = t_y.mean(dim=0)
    Xc = t_x - mu_x
    Yc = t_y - mu_y

    S_pos = Yc.T @ Xc
    S_rot = torch.bmm(R_y, R_x.transpose(1, 2)).mean(dim=0)
    position_energy = (Yc**2).sum()
    if position_energy < 1e-6:
        position_energy = torch.tensor(1.0, device=device, dtype=dtype)
    S_rot = S_rot * position_energy
    S = (1 - alpha) * S_pos + alpha * S_rot

    U, _, Vh = torch.linalg.svd(S)
    R = Vh.T @ U.T

    # Reflection Correction
    if torch.det(R) < 0:
        mask = torch.ones(3, device=device, dtype=dtype)
        mask[-1] = -1
        R = Vh.T @ torch.diag(mask) @ U.T

    # 5. Optional Scale (usually 1.0 for rigid SE(3) trajectories)
    if with_scale:
        var_y = (Yc**2).sum()
        s = torch.trace(S_pos @ R) / var_y
    else:
        s = torch.tensor(1.0, device=device, dtype=dtype)

    t = mu_x - s * (R @ mu_y)

    A = torch.eye(4, device=device, dtype=dtype)
    A[:3, :3] = s * R
    A[:3, 3] = t

    return A, R, t, s


if __name__ == "__main__":
    main()
