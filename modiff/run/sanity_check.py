"""Sanity check for EgoMo3R's motion prior: generate body motion from an example head trajectory.

    python modiff/run/sanity_check.py [--video ./vis]

Prints how closely the head of the generated body follows the input trajectory. Inputs are camera-to-world head
poses (T x 4 x 4). With --video, also renders the generated motion and the input trajectory.
"""

import argparse
import os

import numpy as np
import torch

from modiff.dataset.motion_transforms import align_se3_trajectories
from modiff.dataset.representation_utils import center_aria_traj, repre_to_sequence
from modiff.dataset.smpl_utils import evaluate_smpl, get_smpl
from modiff.module.uem_module import UEM_Module
from modiff.utils import rotation_conversions as rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="data/checkpoints/amass_v1_ait_unet")
    parser.add_argument("--traj", default="assets/example_head_traj.npy", help="T x 4 x 4 head poses")
    parser.add_argument("--video", default=None, help="output directory for a rendered video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = UEM_Module.from_pretrained(args.ckpt)
    model = model.to(device)
    smpl = get_smpl().to(device)

    head_T = torch.from_numpy(np.load(args.traj)).float().to(device)  # T x 4 x 4
    T = head_T.shape[0]

    # EgoMo3R's prior is conditioned on the head trajectory relative to its first frame.
    head_T_c = center_aria_traj(head_T[None])
    traj = torch.cat([rc.matrix_to_rotation_6d(head_T_c[..., :3, :3]), head_T_c[..., :3, 3]], dim=-1)  # 1 x T x 9

    torch.manual_seed(0)
    with torch.no_grad():
        if T == cfg.DATA.WINDOW:
            motion = model.sample({"traj": traj}, 1)[0]  # T x 209
        else:
            motion, _ = model.sample_long(traj[0], mode="multidiffusion")  # any length
        pred_head_T, smpl_params, _ = repre_to_sequence(cfg.DATA.REPRE_TYPE, motion, None, smpl, None, None)
        _, verts, _ = evaluate_smpl(smpl, smpl_params)

    # Generated motion is in its own canonical frame. Rigidly align the input trajectory to it and compare.
    A, _, _, _ = align_se3_trajectories(pred_head_T, head_T)
    head_T_aligned = A @ head_T
    err = (head_T_aligned[:, :3, 3] - pred_head_T[:, :3, 3]).norm(dim=-1)
    print(f"Generated {T} frames. Head position error vs. input trajectory: mean {err.mean() * 100:.1f} cm")

    if args.video is not None:
        from modiff.utils.vis_utils import save_video, visualize_sequence

        os.makedirs(args.video, exist_ok=True)

        imgs = visualize_sequence(aria_traj=head_T_aligned.cpu(), verts=verts.cpu(), faces=smpl.faces)
        save_video(imgs[..., ::-1], "sanity_check", args.video, fps=10)
        print(f"Saved {args.video}/sanity_check.mp4")


if __name__ == "__main__":
    main()
