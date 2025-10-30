import argparse
import os

import numpy as np
from pandas.core.api import bdate_range
import torch
from loguru import logger
from tqdm.auto import tqdm
import glob

from modiff.dataset.amass_utils import amass_root, amass_loop
import modiff.utils.rotation_conversions as rc
from modiff.dataset.canonicalization import rot_trans_to_matrix
from modiff.dataset.egoego_utils import determine_floor_height_and_contacts, local2global_pose
from modiff.dataset.smpl_utils import (
    evaluate_smpl,
    get_smpl,
    full_sequence_to_saved_sequence,
    pca_to_matrix,
    matrix_to_pca,
)
from modiff.utils.torch_utils import to_tensor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="/vision/u/chpatel/data/amass/")
    parser.add_argument("--smpl_type", type=str, default="smplx_n", choices=["smplx_n"])
    parser.add_argument(
        "--source_amass_ds_type", type=str, default="smplx_n", choices=["smplx_n", "smplx_g", "smplh_g"]
    )
    parser.add_argument("--zero_beta", action="store_true", help="Zero beta.")
    parser.add_argument("--out_dir", type=str, default="/vision/u/chpatel/data/modiff")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--out_fps", type=int, default=10, help="Output fps.")
    parser.add_argument("--min_seg_len", type=int, default=1, help="Minimum segment length in seconds.")

    return parser.parse_args()


def main(bdata, smpl, out_fps, min_seg_len, zero_beta):

    fps = bdata["fps"]

    num_frames = bdata["poses"].shape[0]
    fps_ratio = float(out_fps) / fps
    new_num_frames = int(fps_ratio * num_frames)
    didxs = np.linspace(0, num_frames - 1, num=new_num_frames, dtype=int)
    T = len(didxs)

    if T < min_seg_len * out_fps:
        return None

    # Load SMPL parameters
    global_orient = rc.axis_angle_to_matrix(to_tensor(bdata["poses"][didxs, :3]))  # T x 3 x 3
    body_pose = rc.axis_angle_to_matrix(to_tensor(bdata["poses"][didxs, 3:66]).view(T, 21, 3))  # T x 21 x 3 x 3
    betas = to_tensor(bdata["betas"])[None, :10].repeat(T, 1)  # T x 10
    transl = to_tensor(bdata["trans"][didxs])  # T x 3
    if zero_beta:
        betas = torch.zeros_like(betas)

    smpl_params = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "betas": betas,
        "transl": transl,
    }

    # Evaluate SMPL
    kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(kp3d.numpy(), out_fps)

    # Load ARIA trajectory
    head_idx, leye_idx, reye_idx = 15, 23, 24
    global_pose = local2global_pose(full_pose, list(smpl.parents))
    aria_rot = global_pose[:, head_idx]  # T x 3 x 3 head rotation
    # aria_trans = kp3d[:, [leye_idx, reye_idx]].mean(1)  # T x 3 supposed aria translation
    aria_trans = kp3d[:, [head_idx]].mean(1)  # T x 3 supposed aria translation
    aria_traj = rot_trans_to_matrix(aria_rot, aria_trans)  # T x 4 x 4

    smpl_params, aria_traj = full_sequence_to_saved_sequence(smpl_params, aria_traj, smpl)

    # body root is not at origin and depends on beta
    body_root_offset = kp3d[0, 0] - smpl_params["transl"][0]  # 3

    data_dict = {
        "aria_traj": aria_traj,
        "smpl_params": smpl_params,
        "kp3d": kp3d[:, :25],  # only body, no hands
        "body_root_offset": body_root_offset,
        "floor_height": floor_height,
        "discard_seq": discard_seq,
        # "contacts": contacts,
        "num_frames": aria_traj.shape[0],
    }
    return data_dict


def process_data():
    args = get_args()
    print(vars(args))

    out_dict = {}
    out_file = f"{args.out_dir}/amass_{args.smpl_type}{'_zerobeta' if args.zero_beta else ''}_from_{args.source_amass_ds_type}.pt"

    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"{out_file} exists. Skipping.")
        return

    smpl = get_smpl(args.smpl_type)

    # done = {}

    for seq_path, fps, bdata_orig in tqdm(
        amass_loop(
            [args.source_amass_ds_type],
            return_what="bdata",
            fix_fps=True,
            ignore_non_floor_seqs=True,
            verbose=False,
        )
    ):
        ds_name, sub_name, seq_name = seq_path.split("/")[-3:]
        # if (ds_name, sub_name) not in done:
        #     done[(ds_name, sub_name)] = 0
        # if done[(ds_name, sub_name)] > 0:
        #     continue

        bdata = {
            "fps": fps,
            "poses": bdata_orig["poses"][:],
            "betas": bdata_orig["betas"][:],
            "trans": bdata_orig["trans"][:],
        }

        ret = main(bdata, smpl, args.out_fps, args.min_seg_len, args.zero_beta)

        if ret is None:
            logger.warning(f"Got None in {seq_path}")
            continue

        # if not ret["discard_seq"]:
        #     continue
        # if ret["discard_seq"]:
        #     logger.warning(f"Discarding {seq_path}")
        #     continue

        if seq_name.endswith("_stageii.npz"):
            seq_name = seq_name.split("_stageii.npz")[0]
        if seq_name.endswith("_poses.npz"):
            seq_name = seq_name.split("_poses.npz")[0]
        assert not seq_name.endswith(".npz")

        take_name = f"{ds_name}___{sub_name}___{seq_name}"
        assert take_name not in out_dict
        out_dict[take_name] = ret
        # done[(ds_name, sub_name)] += 1

    torch.save(out_dict, out_file)


def visualize_data():
    from modiff.utils.vis_utils import visualize_sequence, save_video
    from modiff.dataset.representation_utils import saved_sequence_to_full_sequence

    args = get_args()
    out_file = f"{args.out_dir}/amass_{args.smpl_type}{'_zerobeta' if args.zero_beta else ''}_from_{args.source_amass_ds_type}.pt"
    out_dict = torch.load(out_file)
    viz_dir = "/vision/u/chpatel/data/amass/viz_discarded"
    os.makedirs(viz_dir, exist_ok=True)

    smpl = get_smpl(args.smpl_type)
    for take_name, data_dict in out_dict.items():
        ds_name, sub_name, seq_name = take_name.split("___")
        aria_traj, smpl_params = saved_sequence_to_full_sequence(data_dict["aria_traj"], data_dict["smpl_params"], smpl)
        kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)
        imgs = visualize_sequence(
            aria_traj=aria_traj, verts=verts, faces=smpl.faces, floor_height=data_dict["floor_height"]
        )
        save_video(imgs[..., ::-1], f"{ds_name}___{sub_name}___{seq_name}", viz_dir, fps=10)


def merge_datasets():
    ds_paths = [
        "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_n.pt",
        "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_g.pt",
    ]
    out_path = "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_n_and_smplx_g.pt"
    assert not os.path.exists(out_path)

    ds = [torch.load(path, weights_only=False) for path in ds_paths]
    print([len(d) for d in ds])

    # make sure that no two keys are the same
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            assert len(set(ds[i].keys()) & set(ds[j].keys())) == 0

    final_ds = {}
    for d in ds:
        final_ds.update(d)
    torch.save(final_ds, out_path)


if __name__ == "__main__":
    # process_data()
    # visualize_data()
    merge_datasets()
    pass  # # fmt: skip
