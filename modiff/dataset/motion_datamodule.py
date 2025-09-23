import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import os
from loguru import logger
from modiff.dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from modiff.dataset.amass_motion_dataset import AMASS_Motion_Dataset
from modiff.utils.torch_utils import careful_collate_fn
from modiff.utils.vis_utils import save_video


class Motion_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        assert stage in ["fit", "validate", "test", "predict", "train", "val"]

        dataset_name = self.cfg.DATA.DATASET_NAME
        assert dataset_name in ["ee4d", "amass"]
        ds_class = {"ee4d": EE4D_Motion_Dataset, "amass": AMASS_Motion_Dataset}[dataset_name]

        if stage in ["fit", "train"]:
            self.train_dataset = ds_class(split="train", cfg=self.cfg)
            logger.info(f"Train dataset: {len(self.train_dataset)}")

        self.val_dataset = ds_class(split="val", cfg=self.cfg)
        logger.info(f"Val dataset: {len(self.val_dataset)}")

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            shuffle=shuffle,
            collate_fn=careful_collate_fn,
            # pin_memory=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            shuffle=True,
            collate_fn=careful_collate_fn,
            # pin_memory=False,
        )


def save_statistics():
    ds_name = "ee4d"
    repre_type = "v5_beta"
    window = 80  # This is the window size used in the paper.
    data_dir = "/vision/group/egoexo4d/"
    out_file = f"{data_dir}/uniegomotion/{repre_type}_{ds_name}_train_stats.pt"

    if os.path.exists(out_file):
        logger.info(f"{out_file} exists. Skipping.")
        return

    cfg = lambda: None
    cfg.DATA = lambda: None
    cfg.DATA.DATA_DIR = data_dir
    cfg.DATA.REPRE_TYPE = repre_type
    cfg.DATA.COND_IMG_FEAT = False
    cfg.DATA.COND_TRAJ = True
    cfg.DATA.WINDOW = window
    cfg.DATA.IMG_FEAT_TYPE = "clip_all"
    cfg.DATA.COND_BETAS = False
    dataset = {"ee4d": EE4D_Motion_Dataset, "amass": AMASS_Motion_Dataset}[ds_name](
        split="train", cfg=cfg, do_normalization=False
    )

    all_motion = []
    all_traj = []
    for sample in tqdm(dataset):
        vf = sample["y"]["valid_frames"].bool()
        motion = sample["misc"]["motion"][vf]
        traj = sample["misc"]["traj"][vf]
        all_motion.append(motion)
        all_traj.append(traj)
    all_motion = torch.cat(all_motion, dim=0)  # T_dataset x F
    all_traj = torch.cat(all_traj, dim=0)  # T_dataset x F

    stats = {}
    stats["motion_mean"] = all_motion.mean(dim=0)
    stats["motion_std"] = all_motion.std(dim=0)
    stats["motion_min"] = all_motion.min(dim=0).values
    stats["motion_max"] = all_motion.max(dim=0).values

    stats["traj_mean"] = all_traj.mean(dim=0)
    stats["traj_std"] = all_traj.std(dim=0)
    stats["traj_min"] = all_traj.min(dim=0).values
    stats["traj_max"] = all_traj.max(dim=0).values

    torch.save(stats, out_file)


def vis_a_sample():
    data_dir = "/vision/group/egoexo4d/"
    out_dir = "/vision/u/chpatel/test"
    window = 80
    use_blender = False  # use_blender=True for blendify blender visualization if available.

    os.makedirs(out_dir, exist_ok=True)

    cfg = lambda: None
    cfg.DATA = lambda: None
    cfg.DATA.DATA_DIR = data_dir
    cfg.DATA.REPRE_TYPE = "v1"
    cfg.DATA.COND_TRAJ = True
    # The following args are not important for visualization.
    cfg.DATA.COND_IMG_FEAT = False
    cfg.DATA.COND_BETAS = False
    cfg.DATA.IMG_FEAT_TYPE = "dinov2"
    cfg.DATA.WINDOW = window
    dataset = EE4D_Motion_Dataset(split="val", cfg=cfg, do_normalization=False)

    # After processing, each take is broken down into several 'good' segments, discarding segments with bad optim results.
    # Thus each sequence name is '<take_name>___<start_frame_index>___<end_frame_index>' where frame indices are according
    # to 30fps numbering and inclusive. Note that EE4D Motion dataset is at 10 fps. For example,
    # uniandes_basketball_003_42___585___825 sequence will have smpl motion sequence of length (825 - 585) / 3 + 1 = 81 at 10fps.

    # You can print all sequence names for that particular split
    # print(dataset.seq_names)

    # Choose a sequence to visualize
    seq = "iiith_cooking_58_2___2478___3498"
    seq = "uniandes_basketball_003_42___585___825"

    # Choose start index within that segment. It will visualize the segment [st:st+window] within `seq`
    st = 0

    sample = dataset.get_from_seq_and_st(seq, st, idx=0)  # idx=0 is for bookkeeping during training. Ignore it here.
    imgs = dataset.visualize_sample(sample, use_blender=use_blender)
    save_video(imgs[..., ::-1], f"{seq}___starting_{st}", out_dir, fps=10)

    import IPython

    IPython.embed()


if __name__ == "__main__":

    save_statistics()
    # vis_a_sample()

    import IPython

    IPython.embed()
