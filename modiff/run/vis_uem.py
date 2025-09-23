import copy
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger

from modiff.config.defaults import get_cfg
from modiff.dataset.ee4d_motion_dataset import careful_collate_fn
from modiff.dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from modiff.dataset.amass_motion_dataset import AMASS_Motion_Dataset
from modiff.module.uem_module import UEM_Module, UEM_Module_TwoStage
from modiff.utils.torch_utils import to_device
from modiff.utils.vis_utils import (
    save_video,
    visualize_sequence_blender,
    visualize_sequence,
    pad_filler,
    pad_filler_traj,
)
from modiff.dataset.smpl_utils import evaluate_smpl, get_smpl


def main():
    use_blender = False

    device = torch.device("cuda")

    pl.seed_everything(62, workers=True)
    sys.argv = sys.argv + [
        "TRAIN.ONLY_VALIDATE",
        "True",
    ]
    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None
    assert os.path.exists(cfg.TRAIN.EXP_PATH)

    ds_name = cfg.DATA.DATASET_NAME
    out_dir = f"{cfg.TRAIN.EXP_PATH}/{ds_name}_vis{cfg.TRAIN.EVAL_SUFFIX}"
    os.makedirs(out_dir, exist_ok=False)

    is_twostage = cfg.MODEL.TRAJ_CKPT_PATH is not None

    # Model with checkpoint
    if not is_twostage:
        ckpt_path = cfg.MODEL.CKPT_PATH
        if cfg.MODEL.CKPT_PATH == "last_ckpt":
            ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")
        assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
        logger.info(f"Loading model from {ckpt_path}")
        model = UEM_Module.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu").to(device).eval()
    else:
        model = UEM_Module_TwoStage(cfg=cfg).to(device).eval()

    smpl = get_smpl()

    for split in ["val"]:
        # Dataset
        ds = {"ee4d": EE4D_Motion_Dataset, "amass": AMASS_Motion_Dataset}[cfg.DATA.DATASET_NAME](split=split, cfg=cfg)

        seq_names = np.random.choice(ds.seq_names, size=10, replace=False)
        examples = {("recon", seq_name, 0) for seq_name in seq_names}

        for task, seq_name, start_idx in examples:
            for jdx in range(1):  # number of generations
                if True:
                    if is_twostage and task == "recon":
                        continue
                    with torch.inference_mode():
                        idx = 0  # Used as bookkeeping during training. Not important here.
                        sample = ds.get_from_seq_and_st(seq_name, start_idx, idx)
                        # Mask appropriately for task
                        pred_sample = ds.process_sample_for_task(sample, task)

                        batch = careful_collate_fn([sample])
                        pred_batch = careful_collate_fn([pred_sample])
                        y = to_device(pred_batch["y"], device)

                        # Sample motion
                        x = model.sample(y, 1, cond_scale=cfg.TRAIN.COND_SCALE)
                        if not is_twostage:
                            if cfg.MODEL.LEARN_TRAJ:
                                pred_batch["pred"]["traj"] = to_device(x, "cpu")
                            else:
                                pred_batch["pred"]["motion"] = to_device(x, "cpu")
                        else:
                            pred_batch["pred"]["traj"] = to_device(x[0], "cpu")
                            pred_batch["pred"]["motion"] = to_device(x[1], "cpu")

                    # Process prediction to get full motion sequence
                    gt_mdata = ds.ret_to_full_sequence(batch)
                    pred_mdata = ds.ret_to_full_sequence(pred_batch)

                    gt_aria_traj_T = gt_mdata["aria_traj_T"][0]
                    gt_smpl_params = gt_mdata["smpl_params_full"][0]
                    nf_gt = len(gt_smpl_params["global_orient"])

                    pred_smpl_params = pred_mdata["smpl_params_full"][0]
                    nf_pred = len(pred_smpl_params["global_orient"])

                    _, pred_verts, _ = evaluate_smpl(smpl, pred_smpl_params)
                    _, gt_verts, _ = evaluate_smpl(smpl, gt_smpl_params)

                    if task == "recon":
                        pass
                    elif task == "gen":
                        gt_verts = None
                        gt_aria_traj_T = None
                    elif task in ["fore"]:
                        avail = min(nf_gt, 20)
                        gt_aria_traj_T = pad_filler_traj(gt_aria_traj_T[:avail], nf_pred)
                        gt_verts = pad_filler(gt_verts[:avail], nf_pred)
                    else:
                        raise NotImplementedError

                    vis_fn = visualize_sequence_blender if use_blender else visualize_sequence
                    imgs = vis_fn(
                        aria_traj=gt_aria_traj_T,
                        verts=gt_verts,
                        faces=smpl.faces,
                        pred_verts=pred_verts,
                    )

                    seq_name = batch["misc"]["seq_name"][0]
                    start_idx, end_idx = batch["misc"]["start_idx"][0], batch["misc"]["end_idx"][0]
                    save_video(
                        imgs[..., ::-1],
                        f"{split}_idx{idx}_{seq_name}_{start_idx}_{end_idx}_{task}_{jdx}",
                        out_dir,
                        fps=10,
                    )


if __name__ == "__main__":
    main()
