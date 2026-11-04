import os
import torch
from loguru import logger


class BufferDict(torch.nn.Module):
    def __init__(self, input_dict):
        super().__init__()
        for k, v in input_dict.items():
            self.register_buffer(k, v)

    def __getitem__(self, key):
        return getattr(self, key)


class Normalizer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.repre_type = cfg.DATA.REPRE_TYPE
        self.dataset_name = cfg.DATA.DATASET_NAME
        self.window = cfg.DATA.WINDOW
        self.default_k = "traj" if cfg.MODEL.LEARN_TRAJ else "motion"

        stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/processed")
        stats_path = f"{stats_dir}/{self.repre_type}_{self.dataset_name}_w{self.window}_stats.pt"
        logger.info(f"Loading stats from {stats_path}")
        self.stats = torch.load(stats_path, weights_only=False)

        clean_it = lambda x: torch.where(x.abs() < 1e-8, torch.ones_like(x), x)
        self.stats["traj_std"] = clean_it(self.stats["traj_std"])
        self.stats["motion_std"] = clean_it(self.stats["motion_std"])
        self.stats = BufferDict(self.stats)

    def normalize(self, a, k):
        # if not self.do_normalization:
        #     return a
        k = k if k is not None else self.default_k
        na = (a - self.stats[k + "_mean"]) / (self.stats[k + "_std"] + 1e-6)
        return na

    def denormalize(self, a, k):
        # if not self.do_normalization:
        #     return a
        k = k if k is not None else self.default_k
        na = a * (self.stats[k + "_std"] + 1e-6) + self.stats[k + "_mean"]
        return na
