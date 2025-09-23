import argparse
import os

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

import modiff.utils.rotation_conversions as rc
from modiff.dataset.canonicalization import rot_trans_to_matrix
from modiff.dataset.egoego_utils import determine_floor_height_and_contacts, local2global_pose
from modiff.dataset.smpl_utils import evaluate_smpl, get_smpl
from modiff.utils.pca_conversions import matrix_to_pca
from modiff.utils.torch_utils import to_tensor

ALL_DATASETS = [
    "ACCAD",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "GRAB",
    "HumanEva",
    "KIT",
    # "MOYO",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    "WEIZMANN",
]
TEST_DATASETS = ["Transitions_mocap", "HumanEva"]
VAL_DATASETS = ["MPI_HDM05", "SFU", "MPI_mosh"]
OUT_FPS = 10
MIN_SEG_LEN = 10


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_dir", type=str, default="/vision/u/chpatel/data/amass/amass_smplx")
    parser.add_argument("--out_dir", type=str, default="/vision/u/chpatel/data/ego2motion")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    return parser.parse_args()


def make_things_small(smpl_params, aria_traj, smpl):
    # Make Things Small
    smpl_params["global_orient"] = rc.matrix_to_rotation_6d(smpl_params["global_orient"])  # T x 6
    smpl_params["body_pose"] = rc.matrix_to_rotation_6d(smpl_params["body_pose"])  # T x (21*6)
    assert torch.allclose(smpl_params["betas"], smpl_params["betas"][0:1])
    smpl_params["betas"] = smpl_params["betas"][:1]  # 1 x 10 because betas are same
    smpl_params["left_hand_pose"] = matrix_to_pca(smpl_params["left_hand_pose"], smpl.left_hand_components)
    smpl_params["right_hand_pose"] = matrix_to_pca(smpl_params["right_hand_pose"], smpl.right_hand_components)

    aria_traj_rot6d = rc.matrix_to_rotation_6d(aria_traj[:, :3, :3])  # T x 6
    aria_traj = torch.cat([aria_traj_rot6d, aria_traj[:, :3, 3]], dim=1)  # T x (6 + 3)
    return smpl_params, aria_traj


def main(seq_path, smpl):

    bdata = np.load(seq_path, allow_pickle=True)
    # bdata = {k: v for k, v in bdata.items()}
    assert bdata["gender"].item() == "neutral"
    fps = bdata["mocap_frame_rate"]
    # correct mislabeled data
    if seq_path.find("BMLhandball") >= 0:
        fps = 240
    if seq_path.find("20160930_50032") >= 0 or seq_path.find("20161014_50033") >= 0:
        fps = 59

    num_frames = bdata["poses"].shape[0]
    fps_ratio = float(OUT_FPS) / fps
    new_num_frames = int(fps_ratio * num_frames)
    didxs = np.linspace(0, num_frames - 1, num=new_num_frames, dtype=int)
    T = len(didxs)

    if T < MIN_SEG_LEN:
        return None

    # Load SMPL parameters
    global_orient = rc.axis_angle_to_matrix(to_tensor(bdata["poses"][didxs, :3]))  # T x 3 x 3
    body_pose = rc.axis_angle_to_matrix(to_tensor(bdata["poses"][didxs, 3:66]).view(T, 21, 3))  # T x 21 x 3 x 3
    # ignoring jaw pose
    left_hand_pose = rc.axis_angle_to_matrix(to_tensor(bdata["poses"][didxs, 75:120]).view(T, 15, 3))  # T x 15 x 3 x 3
    right_hand_pose = rc.axis_angle_to_matrix(to_tensor(bdata["poses"][didxs, 120:]).view(T, 15, 3))  # T x 15 x 3 x 3
    betas = to_tensor(bdata["betas"])[None, :10].repeat(T, 1)  # T x 10
    transl = to_tensor(bdata["trans"][didxs])  # T x 3

    # # neutral body
    # betas = torch.zeros_like(betas)

    smpl_params = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "betas": betas,
        "transl": transl,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
    }

    # Evaluate SMPL
    kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(kp3d.numpy(), OUT_FPS)

    # Load ARIA trajectory
    # leye_idx = 57
    leye_idx = 23
    head_idx = 15
    global_pose = local2global_pose(full_pose, list(smpl.parents))
    aria_rot = global_pose[:, head_idx]  # T x 3 x 3 head rotation
    aria_trans = kp3d[:, leye_idx]  # + 1.5 * (kp3d[:, leye_idx] - kp3d[:, head_idx])  # T x 3 supposed aria translation
    aria_traj = rot_trans_to_matrix(aria_rot, aria_trans)  # T x 4 x 4

    smpl_params, aria_traj = make_things_small(smpl_params, aria_traj, smpl)

    # body root is not at origin and depends on beta
    body_root_offset = kp3d[0, 0] - smpl_params["transl"][0]  # 3

    data_dict = {
        "aria_traj": aria_traj,
        "smpl_params": smpl_params,
        "kp3d": kp3d[:, :76],  # only body and hands
        "body_root_offset": body_root_offset,
        "floor_height": floor_height,
        "discard_seq": discard_seq,
        "contacts": contacts,
        "num_frames": aria_traj.shape[0],
    }
    return data_dict


def process_data():
    args = get_args()

    out_dict = {}
    out_file = f"{args.out_dir}/amass_smplx.pt"
    # out_file = f"{args.out_dir}/amass_smplx_zerobeta.pt"

    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    smpl = get_smpl()

    for ds_name in ALL_DATASETS:
        # if "CMU" not in ds_name:
        #     continue
        for sub_name in os.listdir(f"{args.amass_dir}/{ds_name}"):

            if not os.path.isdir(f"{args.amass_dir}/{ds_name}/{sub_name}"):
                continue

            for seq_name in os.listdir(f"{args.amass_dir}/{ds_name}/{sub_name}"):

                seq_path = f"{args.amass_dir}/{ds_name}/{sub_name}/{seq_name}"

                if "BMLmovi/Subject_49_F_MoSh/Subject_49_F_19_stageii.npz" in seq_path:
                    continue
                if "treadmill" in seq_path:
                    continue

                if not seq_path.endswith(".npz"):
                    logger.info(f"Skipping {seq_path}")
                    continue

                if "stageii" not in seq_name:
                    logger.info(f"Skipping {seq_path}")
                    continue

                logger.info(f"Processing {seq_path}")

                # if "CircleCCW" not in seq_name:
                #     continue
                # if len(out_dict) > 0:
                #     break

                ret = main(seq_path, smpl)

                if ret is None:
                    logger.info(f"Got None in {seq_path}")
                    continue

                if ret["discard_seq"]:
                    logger.info(f"Discarding {seq_path}")
                    continue

                assert "stageii" in seq_name
                seq_name = seq_name.split("_stageii")[0]
                take_name = f"{ds_name}___{sub_name}___{seq_name}"
                assert take_name not in out_dict
                out_dict[take_name] = ret

    torch.save(out_dict, out_file)


if __name__ == "__main__":
    process_data()
    # save_segmented()
