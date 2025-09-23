import numpy as np
import torch

from loguru import logger

from modiff.dataset.ee4d_motion_dataset import EE4D_Motion_Dataset


class AMASS_Motion_Dataset(EE4D_Motion_Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = "amass"

    def load_statistics(self):
        if not self.do_normalization:
            return
        logger.warning("Loading AMASS stats for the window of 80 frames.")
        # self.stats = torch.load(f"{self.data_dir}/uniegomotion/{self.repre_type}_ee_train_stats.pt", weights_only=False)
        self.stats = torch.load(f"/vision/u/chpatel/data/ego2motion/v4_beta_amass_stats.pt", weights_only=False)
        clean_it = lambda x: torch.where(x.abs() < 1e-8, torch.ones_like(x), x)
        self.stats["traj_std"] = clean_it(self.stats["traj_std"])
        self.stats["motion_std"] = clean_it(self.stats["motion_std"])

    def load_motion_data(self):
        processed_path = f"/vision/u/chpatel/data/ego2motion/amass_smplx.pt"
        logger.info(f"Loading {processed_path}.")
        self.motion_data = torch.load(processed_path, weights_only=False)

        TEST_DATASETS = ["Transitions_mocap", "HumanEva"]
        VAL_DATASETS = ["MPI_HDM05", "SFU", "MPI_mosh"]
        val_datasets = VAL_DATASETS + TEST_DATASETS

        self.seq_names = []
        for seq_name in list(self.motion_data.keys()):
            db_name = seq_name.split("___")[0]
            if self.split == "train" and db_name not in val_datasets:
                self.seq_names.append(seq_name)
            elif self.split == "val" and db_name in val_datasets:
                self.seq_names.append(seq_name)
        logger.info(f"Using {len(self.seq_names)} sequences.")

        self.idx_to_sidx_fidx = []
        for seq_idx, seq_name in enumerate(self.seq_names):
            if "treadmill" in seq_name:
                continue
            # if seq_name.startswith("WEIZMANN") or seq_name.startswith("GRAB"):
            #     continue
            T = self.motion_data[seq_name]["num_frames"]
            for fidx in range(0, T - self.segment_stride, self.segment_stride):
                self.idx_to_sidx_fidx.append((seq_idx, fidx))
        self.idx_to_sidx_fidx = np.array(self.idx_to_sidx_fidx).astype(int)


if __name__ == "__main__":

    from modiff.utils.vis_utils import save_video

    dataset = AMASS_Motion_Dataset(
        data_dir="",  # We hardcode for the amass dataset
        split="val",
        repre_type="v4_beta",
        cond_traj=True,
        cond_img_feat=False,
        cond_betas=False,
        window=80,
        img_feat_type="dinov2",
    )
    idx = 1000
    # sidx = dataset.seq_names.index("GRAB___s1___hand_inspect_1")
    # for i, a in enumerate(dataset.idx_to_sidx_fidx):
    #     if a[0] == sidx:
    #         idx = i

    sample = dataset[idx]
    imgs = dataset.visualize_sample(sample)
    save_video(imgs[..., ::-1], f"amass_sample", "/vision/u/chpatel/test", fps=10)

    import IPython

    IPython.embed()
