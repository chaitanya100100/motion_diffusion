import copy
import os
import sys
import copy

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger

from modiff.config.defaults import get_cfg
from modiff.utils.torch_utils import careful_collate_fn
from modiff.dataset.motion_dataset import Motion_Dataset
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
from modiff.utils.pc_viewer_general import PointCloudViewer


class T2MViewer(PointCloudViewer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hack
        self.add_mesh("prediction", np.zeros((1, 1, 3)), np.zeros((1, 3)), static=False)

        # get floor
        from modiff.utils.meshviewer import get_checkerboard_floor

        # floor = get_checkerboard_floor(cam_up=[0.0, 0.0, 1.0])
        # self.add_mesh("floor", floor.vertices, floor.faces, color=[0.7, 0.7, 0.7], static=True, wireframe=True)
        grid = self.server.scene.add_grid(
            "grid",
            width=20.0,
            height=20.0,
            position=np.array([0.0, 0.0, 0.0]),
        )

    def add_model_interface(self, predict_fn):
        with self.server.add_gui_folder("Model Inference"):
            gui_prompt = self.server.add_gui_text("Prompt", "walking")
            gui_btn = self.server.add_gui_button("Run")

        @gui_btn.on_click
        def _(_):
            result = predict_fn(gui_prompt.value)

            self.num_frames = len(result["vertices"])
            self.gui_t.max = self.num_frames - 1

            self.add_mesh(name="prediction", vertices=result["vertices"], faces=result["faces"], static=False)

            self._render_all(names_to_render=["prediction"])


def main():

    device = torch.device("cuda")

    pl.seed_everything(62, workers=True)
    sys.argv = sys.argv + [
        "TRAIN.ONLY_VALIDATE",
        "True",
    ]
    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None
    assert os.path.exists(cfg.TRAIN.EXP_PATH)

    # ds_name = cfg.DATA.DATASET_NAME
    # out_dir = f"{cfg.TRAIN.EXP_PATH}/{ds_name}_vis{cfg.TRAIN.EVAL_SUFFIX}"
    # os.makedirs(out_dir, exist_ok=False)

    ckpt_path = cfg.MODEL.CKPT_PATH
    if cfg.MODEL.CKPT_PATH == "last_ckpt":
        ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
    logger.info(f"Loading model from {ckpt_path}")
    model = UEM_Module.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu").to(device).eval()

    smpl = get_smpl()

    from modiff.dataset.feats import TextFeats
    from modiff.dataset.representation_utils import repre_to_sequence

    text_feats = TextFeats(cfg.DATA.TEXT_ENCODER_NAME)
    text_model = text_feats.load_text_model(device)

    def predict_fn(prompt):
        with torch.inference_mode():
            text_emb = text_feats.encode_text(text_model, prompt, device).float()
            text_emb = to_device(text_emb, device)
            x = model.sample({"text_emb": text_emb[None]}, 1, cond_scale=cfg.TRAIN.COND_SCALE)
            x = to_device(x, "cpu")[0]

            _, smpl_params, _ = repre_to_sequence(cfg.DATA.REPRE_TYPE, x, None, smpl, None, None)
            _, verts, _ = evaluate_smpl(smpl, smpl_params)
            return {
                "vertices": verts,
                "faces": smpl.faces,
            }

    viewer = T2MViewer()
    viewer.add_model_interface(predict_fn)
    viewer.run()


if __name__ == "__main__":
    main()
