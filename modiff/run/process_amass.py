import argparse
import os

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm
import glob
import json

import modiff.dataset.amass_utils as amass_utils
from modiff.dataset.amass_utils import amass_root, amass_loop, smplh_key_to_smplx_key, process_text_anns
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
    # TODO: paths in this file are on our cluster. If you want to preprocess AMASS, set them to your own paths.
    parser.add_argument("--amass_root", type=str, default="/vision/u/chpatel/data/amass/")
    parser.add_argument("--smpl_type", type=str, default="smplx_n", choices=["smplx_n"])
    parser.add_argument("--mirrored", action="store_true", help="Work with mirrored sequences.")
    parser.add_argument("--source_amass_ds_type", type=str, choices=["smplx_n", "smplx_g", "smplh_g"], required=True)
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
    amass_utils.amass_root = args.amass_root  # amass_loop reads the AMASS root from amass_utils

    out_dict = {}
    mirror_prefix = "M_" if args.mirrored else ""
    out_file = (
        f"{args.out_dir}/amass_"
        f"{args.out_fps}fps_"
        f"{args.smpl_type}"
        f"{'_zerobeta' if args.zero_beta else ''}"
        f"_from_"
        f"{mirror_prefix}{args.source_amass_ds_type}.pt"
    )
    print(out_file)

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
            mirrored=args.mirrored,
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


def merge_datasets(ds_paths=None, out_path=None):
    # ds_paths = [
    #     "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_n.pt",
    #     "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_g.pt",
    # ]
    # out_path = "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_ng.pt"

    if ds_paths is None:
        ds_paths = [
            "/vision/u/chpatel/data/modiff/amass_10fps_smplx_n_from_M_smplx_n.pt",
            "/vision/u/chpatel/data/modiff/amass_10fps_smplx_n_from_smplx_n.pt",
        ]
        out_path = "/vision/u/chpatel/data/modiff/amass_10fps_smplx_n_from_mM_smplx_n.pt"
    assert not os.path.exists(out_path)

    def modify_key_for_mirrored(key):
        try:
            ds_name, sub_name, seq_name = key.split("___", 2)
        except:
            logger.error(f"Error splitting key: {key}")
            raise
        return f"{ds_name}___{sub_name}___M_{seq_name}"

    ds = []
    for path in ds_paths:
        d = torch.load(path, weights_only=False)
        logger.info(f"Loading {path}")
        if "_M_" in path:
            logger.info(f"Modifying keys for this mirrored dataset")
            d = {modify_key_for_mirrored(key): value for key, value in d.items()}
        ds.append(d)
    print([len(d) for d in ds])

    # make sure that no two keys are the same
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            assert len(set(ds[i].keys()) & set(ds[j].keys())) == 0

    final_ds = {}
    for d in ds:
        final_ds.update(d)
    torch.save(final_ds, out_path)


def text_anns_smplh_to_smplx():
    text_anns_file = "/vision/u/chpatel/data/modiff/babel_humanml3d_kitml_with_mirror.json"
    out_file = "/vision/u/chpatel/data/modiff/babel_humanml3d_kitml_with_mirror_smplx.json"
    assert not os.path.exists(out_file)

    with open(text_anns_file, "r") as f:
        text_anns = json.load(f)

    new_text_anns = {}
    not_available_in_smplx = []
    for key, ann in tqdm(text_anns.items()):
        if "humanact12" in key:
            continue
        new_key, gender = smplh_key_to_smplx_key(key)
        if new_key is None:  # not available in smplx
            not_available_in_smplx.append(key)
            continue

        fpath = new_key
        if fpath.startswith("M/"):
            fpath = fpath[2:]
            mirror_prefix = "M_"
        else:
            mirror_prefix = ""

        in_smplx_n = os.path.exists(f"{amass_root}/{mirror_prefix}smplx_n/{fpath}.npz")
        in_smplx_g = os.path.exists(f"{amass_root}/{mirror_prefix}smplx_g/{fpath}.npz")

        if not in_smplx_n and not in_smplx_g:
            logger.error(f"Sequence {key} with new key {new_key} is not available")
            raise

        new_text_anns[new_key] = ann

    logger.info(f"Total: {len(text_anns)}")
    logger.info(f"Not available in smplx: {len(not_available_in_smplx)}")
    logger.info(f"Available in smplx: {len(new_text_anns)}")

    with open(out_file, "w") as f:
        json.dump(new_text_anns, f, indent=4)


def visualize_with_text_anns():
    import json
    from modiff.utils.vis_utils import visualize_sequence, save_video, put_text_on_img_pil
    from modiff.dataset.representation_utils import saved_sequence_to_full_sequence

    out_file = "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplh_g_mM.pt"
    text_anns_file = "/vision/u/chpatel/data/modiff/babel_humanml3d_kitml_with_mirror.json"
    viz_dir = "/vision/u/chpatel/data/modiff/viz_text_anns_smplh_on_smplx"

    out_file = "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_ng_mM.pt"
    text_anns_file = "/vision/u/chpatel/data/modiff/babel_humanml3d_kitml_with_mirror_smplx.json"
    viz_dir = "/vision/u/chpatel/data/modiff/viz_text_anns_smplx"

    os.makedirs(viz_dir, exist_ok=True)
    out_dict = torch.load(out_file, weights_only=False)
    with open(text_anns_file, "r") as f:
        text_anns = json.load(f)
    text_anns = process_text_anns(text_anns)

    smpl_type = "smplx_n"
    smpl = get_smpl(smpl_type)

    # Select random keys to visualize
    keys_to_viz = list(text_anns.keys())
    np.random.seed(42)
    np.random.shuffle(keys_to_viz)

    # Add corresponding mirrored/unmirrored keys to the list
    new_keys_to_viz = []
    for key in keys_to_viz:
        print(key)
        new_keys_to_viz.append(key)

        # Check if mirrored/unmirrored key is available in text_anns
        ds_name, sub_name, seq_name = key.split("___", 2)
        if seq_name.startswith("M_"):
            seq_name = seq_name[2:]
        else:
            seq_name = f"M_{seq_name}"

        new_key = f"{ds_name}___{sub_name}___{seq_name}"
        if new_key in text_anns:
            print(new_key)
            new_keys_to_viz.append(new_key)

        if len(new_keys_to_viz) > 100:
            break
    keys_to_viz = sorted(new_keys_to_viz)

    # Visualize the selected keys
    for key in keys_to_viz:
        logger.info(f"Visualizing {key}")
        if key not in out_dict:
            logger.error(f"Key {key} not found in out_dict")
            logger.error(f"Annotations: {text_anns[key]}")
            continue
        data_dict = out_dict[key]
        ds_name, sub_name, seq_name = key.split("___", 2)
        aria_traj, smpl_params = saved_sequence_to_full_sequence(data_dict["aria_traj"], data_dict["smpl_params"], smpl)
        kp3d, verts, full_pose = evaluate_smpl(smpl, smpl_params)
        imgs = visualize_sequence(
            aria_traj=aria_traj, verts=verts, faces=smpl.faces, floor_height=data_dict["floor_height"]
        )

        text = ""
        for ann in text_anns[key]["annotations"]:
            text += f"{ann['start']:.1f}-{ann['end']:.1f}: {ann['text']}\n"
        print(text)

        imgs = put_text_on_img_pil(imgs, text, font_size=20)
        save_video(imgs[..., ::-1], f"{ds_name}___{sub_name}___{seq_name}", viz_dir, fps=10)


if __name__ == "__main__":
    # process_data()
    # visualize_data()
    merge_datasets()
    # text_anns_smplh_to_smplx()
    # visualize_with_text_anns()
    pass  # # fmt: skip
