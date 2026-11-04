import os
import IPython
import numpy as np
import torch
import copy
import pytorch_lightning as pl
from tqdm.auto import tqdm


from loguru import logger
from torch.utils.data import Dataset, DataLoader

from modiff.dataset.canonicalization import get_a_canonicalized_segment
from modiff.dataset.representation_utils import repre_to_sequence, sequence_to_repre
from modiff.dataset.smpl_utils import get_smpl, evaluate_smpl
from modiff.utils.vis_utils import save_video, visualize_sequence, visualize_sequence_blender, put_text_on_img_pil
from modiff.utils.torch_utils import careful_collate_fn, to_device, to_numpy, to_tensor
from modiff.dataset.feats import ImageFeats, TextFeats


class Motion_Dataset(Dataset):
    def __init__(self, *, cfg, split):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.cond_img_feat = cfg.DATA.COND_IMG_FEAT
        self.cond_traj = cfg.DATA.COND_TRAJ
        self.repre_type = cfg.DATA.REPRE_TYPE
        self.img_feat_type = cfg.DATA.IMG_FEAT_TYPE
        self.cond_betas = cfg.DATA.COND_BETAS
        self.dataset_name = cfg.DATA.DATASET_NAME
        self.cond_text_feat = cfg.DATA.COND_TEXT_FEAT
        self.text_encoder_name = cfg.DATA.TEXT_ENCODER_NAME
        if self.cond_betas:
            logger.warning("Conditioning on betas.")
        if self.cond_text_feat:
            logger.warning("Conditioning on text features.")
        assert self.split in ["train", "val"]
        assert self.img_feat_type in ["clip_all", "egovideo", "dinov2", "dinov2_reg"]

        if not self.cond_traj:
            logger.warning("NOT USING TRAJ AS CONDITION.")

        self.window = cfg.DATA.WINDOW
        self.segment_stride = getattr(cfg.DATA, "SEGMENT_STRIDE", 4)

        # Load SMPL
        self.smpl = get_smpl()

        self.load_motion_data()

    def load_amass_data(self):
        processed_path = self.cfg.DATA.AMASS_DATA_PATH
        logger.info(f"Loading {processed_path}.")
        self.motion_data = torch.load(processed_path, weights_only=False)
        val_datasets = ["Transitions_mocap", "Transitions", "HumanEva", "MPI_HDM05", "HDM05", "SFU", "MPI_mosh", "MoSh"]
        self.seq_names = []
        for seq_name in list(self.motion_data.keys()):
            db_name = seq_name.split("___")[0]
            if self.split == "train":  # We train on all the sequences
                self.seq_names.append(seq_name)
            elif self.split == "val" and db_name in val_datasets:
                self.seq_names.append(seq_name)

        if self.cond_text_feat:
            self.text_feats = TextFeats(self.text_encoder_name)
            self.text_feats.prepare_for_training(self.seq_names, self.window)

    def load_ee4d_data(self):

        # TODO: path on our cluster. If you want to use this, set it to your own path.
        ee4d_dir = "/vision/group/egoexo4d/"
        processed_path = f"{ee4d_dir}/uniegomotion/ee_{self.split}.pt"
        logger.info(f"Loading {processed_path}.")
        self.motion_data = torch.load(processed_path, weights_only=False)
        self.seq_names = list(self.motion_data.keys())

        if self.cond_img_feat:
            self.img_feats = ImageFeats(ee4d_dir, self.split, self.img_feat_type)

    def load_motion_data(self):

        if self.dataset_name == "ee4d":
            self.load_ee4d_data()
        elif self.dataset_name == "amass":
            self.load_amass_data()
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        if self.cond_text_feat:
            return

        self.idx_to_sidx_fidx = []
        for seq_idx, seq_name in enumerate(self.seq_names):
            T = self.motion_data[seq_name]["num_frames"]
            for fidx in range(0, T - self.segment_stride, self.segment_stride):
                self.idx_to_sidx_fidx.append((seq_idx, fidx))
        self.idx_to_sidx_fidx = np.array(self.idx_to_sidx_fidx).astype(int)

        logger.info(f"Using {len(self.seq_names)} sequences.")
        logger.info(f"Using {len(self.idx_to_sidx_fidx)} segments.")

    def __len__(self):
        if self.cond_text_feat:
            return len(self.text_feats)
        return len(self.idx_to_sidx_fidx)

    def pad_to_window(self, ret):
        valid_frames = torch.ones(self.window).long()
        T = ret["misc"]["motion"].shape[0]
        if T != self.window:
            valid_frames[T:] = 0  # mark invalid frames
            pad_fn = lambda x: torch.cat([x, torch.zeros(self.window - T, *x.shape[1:], dtype=x.dtype)], dim=0)

            if "traj" in ret["y"]:
                ret["y"]["traj"] = pad_fn(ret["y"]["traj"])
            if "img_embs" in ret["y"]:
                ret["y"]["img_embs"] = pad_fn(ret["y"]["img_embs"])
                ret["y"]["valid_img_embs"] = pad_fn(ret["y"]["valid_img_embs"])
            if "traj" in ret["misc"]:
                ret["misc"]["traj"] = pad_fn(ret["misc"]["traj"])
            if "motion" in ret["misc"]:
                ret["misc"]["motion"] = pad_fn(ret["misc"]["motion"])

        ret["y"]["valid_frames"] = valid_frames  # determines the number of frames
        return ret

    def __getitem__(self, idx):
        if self.cond_text_feat:
            seq_name, st, en, text = self.text_feats.idx_to_info[idx]
            ret = self.get_from_seq_and_st(seq_name, st, idx, en)
            ret["y"]["text_emb"] = self.text_feats.get_text_feat(text)
            ret["misc"]["text"] = text
            return ret
        sidx, st = self.idx_to_sidx_fidx[idx]
        seq_name = self.seq_names[sidx]
        return self.get_from_seq_and_st(seq_name, st, idx)

    def get_from_seq_and_st(self, seq_name, st, idx, en=None):
        if en is None:
            en = min(st + self.window - 1, self.motion_data[seq_name]["num_frames"] - 1)
        else:
            # assert en <= self.motion_data[seq_name]["num_frames"] - 1
            if en > self.motion_data[seq_name]["num_frames"] - 1:
                logger.warning(f"{idx} {seq_name} {st} {en} {self.motion_data[seq_name]['num_frames']}")
                en = self.motion_data[seq_name]["num_frames"] - 1
        floor_height = self.motion_data[seq_name]["floor_height"]
        # logger.warning(f"Sequence: {seq_name}, start index: {st}, end index: {en}")

        segment_data = get_a_canonicalized_segment(
            self.motion_data[seq_name]["smpl_params"],
            self.motion_data[seq_name]["aria_traj"],
            self.motion_data[seq_name]["kp3d"],
            self.smpl,
            st,
            en,
        )

        motion, traj = sequence_to_repre(
            self.repre_type,
            segment_data["can_aria_traj"],
            segment_data["can_smpl_params"],
            segment_data["can_kp3d"],
            floor_height,
            self.smpl,
        )

        if self.cond_img_feat:
            img_embs, valid_img_embs = self.img_feats.get_img_feats(seq_name, st * 3, en * 3)  # T x 768

        if self.cond_betas:
            betas = segment_data["can_smpl_params"]["betas"][:1].view(10)

        ret = {}
        ret["y"] = {}  # holds conditioning inputs
        ret["pred"] = {}  # holds predicted outputs
        if self.cond_traj:
            ret["y"]["traj"] = traj
        if self.cond_img_feat:
            ret["y"]["img_embs"] = img_embs
            ret["y"]["valid_img_embs"] = valid_img_embs

        if self.cond_betas:
            ret["y"]["betas"] = betas
        ret["misc"] = {  # holds ground truth and other metadata
            "seq_name": seq_name,
            "start_idx": st * 3,
            "end_idx": en * 3,
            "idx": idx,
            "dataset": self.dataset_name,
            "traj": traj,
            "motion": motion,
        }
        ret = self.pad_to_window(ret)
        return ret

    def repre_to_sequence(self, motion, traj, vf=None):
        is_batch = True
        if motion.ndim == 2:
            motion = motion[None]
            traj = traj[None] if traj is not None else None
            vf = vf[None] if vf is not None else None
            is_batch = False

        mdata = {
            "aria_traj_T": [],
            "smpl_params": [],
            "kp3d": [],
            "verts": [],
            "full_pose": [],
        }

        for i in range(motion.shape[0]):
            motion_i = motion[i]
            traj_i = traj[i] if traj is not None else None
            if vf is not None:
                vf_i = vf[i].bool()
                motion_i = motion_i[vf_i]
                traj_i = traj_i[vf_i] if traj_i is not None else None

            aria_traj_T, smpl_params, _ = repre_to_sequence(self.repre_type, motion_i, traj_i, self.smpl, None, None)
            kp3d, verts, full_pose = evaluate_smpl(self.smpl, smpl_params)

            mdata["aria_traj_T"].append(aria_traj_T)
            mdata["smpl_params"].append(smpl_params)
            mdata["kp3d"].append(kp3d)
            mdata["verts"].append(verts)
            mdata["full_pose"].append(full_pose)

        return mdata if is_batch else {k: v[0] for k, v in mdata.items()}


if __name__ == "__main__":
    cfg = lambda: None
    cfg.DATA = lambda: None
    cfg.DATA.DATASET_NAME = "amass"
    cfg.DATA.REPRE_TYPE = "v1"
    cfg.DATA.COND_TEXT_FEAT = True
    cfg.DATA.COND_IMG_FEAT = False
    cfg.DATA.IMG_FEAT_TYPE = "dinov2"
    cfg.DATA.COND_BETAS = False
    cfg.DATA.COND_TRAJ = False
    cfg.DATA.AMASS_DATA_PATH = "/vision/u/chpatel/data/modiff/amass_smplx_n_zerobeta_from_smplx_ng_mM.pt"
    cfg.DATA.WINDOW = 128
    dataset = Motion_Dataset(split="train", cfg=cfg)
    print(len(dataset))

    while True:
        i = np.random.randint(0, len(dataset) - 1)
        sample = dataset[i]

        mdata = dataset.repre_to_sequence(sample["misc"]["motion"], sample["misc"]["traj"], sample["y"]["valid_frames"])
        text = sample["misc"]["text"]
        aria_traj_T = mdata["aria_traj_T"]
        verts = mdata["verts"]
        kp3d = mdata["kp3d"]
        faces = dataset.smpl.faces
        imgs = visualize_sequence(aria_traj=aria_traj_T, global_jpos=kp3d, verts=verts, faces=faces)
        imgs = put_text_on_img_pil(imgs, text)
        for i in range(imgs.shape[0]):
            imgs[i] = put_text_on_img_pil(imgs[i], f"{i}", pos="top_right")

        seq_name = sample["misc"]["seq_name"]
        st, en = sample["misc"]["start_idx"], sample["misc"]["end_idx"]
        save_video(imgs[..., ::-1], f"{seq_name}___{st}___{en}", "/vision/u/chpatel/test/motion_diffusion", fps=10)
