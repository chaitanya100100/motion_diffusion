import os
import numpy as np
import torch
import argparse
from tqdm.auto import tqdm
from loguru import logger
from modiff.dataset.amass_utils import amass_loop

# FROM
# https://github.com/mkocabas/PARE/blob/master/pare/core/constants.py
# Permutation of SMPL pose parameters when flipping the shape
# fmt: off
# SMPL_JOINTS_FLIP_PERM = [
#     0, 2, 1, 3, 5, 4, 6, 8, 7, 9,
#     11, 10, 12, 14, 13, 15, 17, 16, 19, 18,
#     21, 20,
# ]  # Hands are removed: # , 23, 22]
# # fmt: on

# SMPL_POSE_FLIP_PERM = []
# for i in SMPL_JOINTS_FLIP_PERM:
#     SMPL_POSE_FLIP_PERM.append(3 * i)
#     SMPL_POSE_FLIP_PERM.append(3 * i + 1)
#     SMPL_POSE_FLIP_PERM.append(3 * i + 2)

SMPL_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9,
    11, 10, 12, 14, 13, 15, 17, 16, 19, 18,
    21, 20, 23, 22
]

SMPLH_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9,
    11, 10, 12, 14, 13, 15, 17, 16, 19, 18,
    21, 20,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
]

SMPLX_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9,
    11, 10, 12, 14, 13, 15, 17, 16, 19, 18,
    21, 20,
    22, 24, 23,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
]


# FROM
# https://github.com/mkocabas/PARE/blob/master/pare/utils/image_utils.py#L220


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl_type", type=str, required=True, choices=["smplx_n", "smplx_g", "smplh_g"])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def flip_pose(_pose, smpl_type):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    pose = _pose.clone()

    smpl_type = smpl_type.split("_")[0].lower()
    if smpl_type == "smplx":
        smpl_joints_flip_perm = SMPLX_PERM
    elif smpl_type == "smplh":
        smpl_joints_flip_perm = SMPLH_PERM
    elif smpl_type == "smpl":
        smpl_joints_flip_perm = SMPL_PERM
    else:
        raise ValueError(f"Invalid smpl_type: {smpl_type}")

    smpl_pose_flip_perm = []
    for i in smpl_joints_flip_perm:
        smpl_pose_flip_perm.append(3 * i)
        smpl_pose_flip_perm.append(3 * i + 1)
        smpl_pose_flip_perm.append(3 * i + 2)
    
    assert len(smpl_pose_flip_perm) == pose.shape[-1]

    pose = pose[..., smpl_pose_flip_perm]
    # we also negate the second and the third dimension of the axis-angle
    pose[..., 1::3] = -pose[..., 1::3]
    pose[..., 2::3] = -pose[..., 2::3]
    return pose


def main():

    args = get_args()
    assert args.smpl_type in ["smplx_n", "smplx_g", "smplh_g"]
    src_string = f"/{args.smpl_type}/"  # e.g. "/smplx_n/"
    dst_string = f"/M_{args.smpl_type}/"  # e.g. "/M_smplx_n/"


    for motion_path in tqdm(amass_loop(
        [args.smpl_type],
        return_what="seq_path",
        fix_fps=False,
        ignore_non_floor_seqs=False,
        verbose=True,
    )):
        if "humanact12" in motion_path:
            continue

        new_motion_path = motion_path.replace(src_string, dst_string)

        if os.path.exists(new_motion_path) and not args.overwrite:
            logger.info(f"{new_motion_path} exists. Skipping.")
            continue
        parent_dir = os.path.dirname(new_motion_path)
        os.makedirs(parent_dir, exist_ok=True)

        data = {x: y for x, y in np.load(motion_path, allow_pickle=True).items()}

        # process sequences
        poses = torch.from_numpy(data["poses"])
        trans = torch.from_numpy(data["trans"])

        # flip poses
        # assert poses.shape[-1] == 66
        mirror_poses = flip_pose(poses, smpl_type=args.smpl_type)

        # flip trans
        x, y, z = torch.unbind(trans, dim=-1)
        mirrored_trans = torch.stack((-x, y, z), axis=-1)

        data["poses"] = mirror_poses.numpy()
        data["trans"] = mirrored_trans.numpy()

        np.savez(new_motion_path, **data)


if __name__ == "__main__":
    main()
