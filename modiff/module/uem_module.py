import os
import copy
import pytorch_lightning as pl
import torch
import torch.distributed
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from modiff.model.uniegomotion import UniEgoMotion
from modiff.model.uniegomotion_basic import UniEgoMotionBasic
from modiff.model.uniegomotion_dec import UniEgoMotionDec
from modiff.model.uniegomotion_basic_latentcond import UniEgoMotionBasicLatentCond, MotionEncoder

from modiff.module.utils import cfg_to_dict, create_gaussian_diffusion
from modiff.mydiffusion.resample import create_named_schedule_sampler
from modiff.mydiffusion.gaussian_diffusion import sum_flat
from modiff.dataset.normalizer import Normalizer


class UEM_Module(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        if not cfg.TRAIN.ONLY_VALIDATE:
            self.save_hyperparameters(cfg_to_dict(cfg))
        self.cfg = cfg
        self.window = self.cfg.DATA.WINDOW
        self.model_name = cfg.MODEL.MODEL_NAME
        self.learn_traj = cfg.MODEL.LEARN_TRAJ
        if self.learn_traj:
            assert not self.cfg.DATA.COND_TRAJ
        self.normalizer = Normalizer(cfg)

        if self.model_name == "uem":
            self.model = UniEgoMotion(self.cfg)
        elif self.model_name == "uem_basic" or self.model_name == "uem_unet":
            self.model = UniEgoMotionBasic(self.cfg)
        elif self.model_name == "uem_dec":
            self.model = UniEgoMotionDec(self.cfg)
        elif self.model_name == "uem_basic_latentcond":
            self.model = UniEgoMotionBasicLatentCond(self.cfg)
            self.motion_encoder = MotionEncoder(self.cfg)
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

        self.diffusion = create_gaussian_diffusion(cfg)

        self.schedule_sampler_type = "uniform"
        # self.schedule_sampler_type = "loss-second-moment"
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        self.last_iters = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            # fused=True,
        )
        # return optimizer

        # scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=15, verbose=True)
        scheduler = StepLR(optimizer, self.cfg.TRAIN.NUM_EPOCHS - 50, gamma=0.1)
        return [optimizer], [
            {
                "scheduler": scheduler,
                # "monitor": "val/loss_epoch",
                "interval": "epoch",
            }
        ]

    def on_train_start(self):
        if self.cfg.MODEL.CKPT_PATH is None:
            return
        if self.cfg.TRAIN.USE_CKPT_LR:
            logger.warning("Using LR from checkpoint.")
            return
        logger.warning("Discarding LR of optimizer dict and using config LR.")
        for g in self.optimizers().param_groups:
            g["lr"] = self.cfg.TRAIN.LR
        for g in self.optimizers().param_groups:
            g["weight_decay"] = self.cfg.TRAIN.WEIGHT_DECAY

    def training_step(self, batch, batch_idx, mode="train"):

        if "traj" in batch["y"]:
            batch["y"]["traj"] = self.normalizer.normalize(batch["y"]["traj"], "traj")
        x = batch["misc"]["traj"] if self.learn_traj else batch["misc"]["motion"]
        x = self.normalizer.normalize(x, None)

        if self.model_name == "uem_basic_latentcond":
            batch["y"]["semcode"] = self.motion_encoder(x, batch["y"]["valid_frames"])

        t, weights = self.schedule_sampler.sample(x.shape[0], self.device)
        losses = self.diffusion.training_losses(self.model, x, t, model_kwargs={"y": batch["y"]})
        if mode == "train" and self.schedule_sampler_type != "uniform":
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()

        # To monitor diffusion step wise loss
        assert losses["loss"].shape[0] == x.shape[0]
        diff_steps, bin_size = self.cfg.MODEL.DIFFUSION_STEPS, self.cfg.MODEL.DIFFUSION_STEPS // 10
        for idx, i in enumerate(range(0, diff_steps, bin_size)):
            mask = (t >= i) & (t < i + bin_size)
            if mask.any():
                range_loss = losses["loss"][mask].mean()
                self.log(
                    f"{mode}_loss/{idx}",
                    range_loss,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=mask.sum(),
                )

        for k, v in losses.items():
            self.log(f"{mode}/{k}", v.mean(), on_step=True, on_epoch=True, sync_dist=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.training_step(batch, batch_idx, mode="val")

    def sample(self, y, B=1, cond_scale=None, return_all_pred_xstart=False, cond_fn=None):

        y = copy.deepcopy(y)
        if "traj" in y:
            y["traj"] = self.normalizer.normalize(y["traj"], "traj")

        # B = y["traj"].shape[0]
        for k, v in y.items():
            assert len(v) == B, f"y[{k}] has batch size {len(v)} but expected {B}"
        x = self.diffusion.p_sample_loop(
            self.model,
            (B, self.window, self.model.input_feats),
            model_kwargs={"y": y, "cond_scale": cond_scale, "diffusion": self.diffusion},
            clip_denoised=False,
            noise=None,
            progress=False,
            cond_fn=cond_fn,
            # skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            # init_image=None,
            # dump_steps=None,
            # const_noise=False,
            return_all_pred_xstart=return_all_pred_xstart,
        )
        if return_all_pred_xstart:
            x[0] = self.normalizer.denormalize(x[0], None)
            x[1] = [self.normalizer.denormalize(s, None) for s in x[1]]
            return x[0], x[1]  # final sample, all predicted xstart
        else:
            x = self.normalizer.denormalize(x, None)
            return x

    def sample_long(self, aria_traj_T, **kwargs):
        """Generate a motion longer than cfg.DATA.WINDOW from a long head trajectory.

        Thin delegation to modiff.module.long_sample.sample_long -- see there for
        the modes and options. Imported lazily so this module's import graph is
        unchanged. Returns (motion: T x 209, info).
        """
        from modiff.module.long_sample import sample_long

        return sample_long(self, aria_traj_T, **kwargs)

    @classmethod
    def from_pretrained(cls, exp_path):
        from modiff.config.defaults import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg_fpath = os.path.join(exp_path, "hparams.yaml")
        assert os.path.exists(cfg_fpath), f"Config file {cfg_fpath} does not exist"
        cfg.merge_from_file(cfg_fpath)
        cfg.freeze()

        ckpt_path = os.path.join(exp_path, "last.ckpt")
        assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist"
        logger.info(f"Loading model from {ckpt_path}")
        model = cls.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu").eval()

        return model, cfg


class UEM_Module_TwoStage(pl.LightningModule):
    """For two stage baseline."""

    def __init__(self, cfg):
        super().__init__()

        traj_ckpt_path = cfg.MODEL.TRAJ_CKPT_PATH
        traj_exp_path = os.path.dirname(traj_ckpt_path)
        traj_cfg = cfg.clone()
        traj_cfg.defrost()
        traj_cfg.merge_from_file(f"{traj_exp_path}/hparams.yaml")
        traj_cfg.freeze()

        motion_ckpt_path = cfg.MODEL.MOTION_CKPT_PATH
        motion_exp_path = os.path.dirname(motion_ckpt_path)
        motion_cfg = cfg.clone()
        motion_cfg.defrost()
        motion_cfg.merge_from_file(f"{motion_exp_path}/hparams.yaml")
        motion_cfg.freeze()

        logger.warning(f"Loading from {traj_ckpt_path}")
        self.traj_module = UEM_Module.load_from_checkpoint(traj_ckpt_path, cfg=traj_cfg, map_location="cpu")

        logger.warning(f"Loading from {motion_ckpt_path}")
        self.motion_module = UEM_Module.load_from_checkpoint(motion_ckpt_path, cfg=motion_cfg, map_location="cpu")

        self.motion_cfg = motion_cfg
        self.traj_cfg = traj_cfg

    def sample_traj(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        return self.traj_module.sample(y, B, cond_scale, return_all_pred_xstart)

    def sample_motion(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        return self.motion_module.sample(y, B, cond_scale, return_all_pred_xstart)

    def sample(self, y, B=1, cond_scale=None, return_all_pred_xstart=False):
        assert self.motion_cfg.DATA.REPRE_TYPE == self.traj_cfg.DATA.REPRE_TYPE

        # prep for trajectory prediction
        traj = y.pop("traj", None)  # B x T x D
        traj_mask = y.pop("traj_mask", None)  # B x T

        assert traj is not None
        if traj_mask is None:
            traj_mask = torch.zeros_like(traj[:, :, 0])

        y["repaint_mask"] = 1 - traj_mask[..., None].expand_as(traj)
        y["repaint_value"] = traj
        traj_pred = self.sample_traj(y, B, cond_scale, return_all_pred_xstart=False)

        # prep for motion prediction
        y.pop("repaint_mask", None)
        y.pop("repaint_value", None)
        y["traj"] = traj_pred
        y["traj_mask"] = torch.zeros_like(traj_mask)

        motion = self.sample_motion(y, B, cond_scale, return_all_pred_xstart=False)
        return traj_pred, motion
