import copy
import numpy as np
import torch
import torch.nn as nn
import einops
from modiff.model.core import DecoderBlock
from modiff.model.core import EncoderBlock
from loguru import logger
from modiff.model.unet_1d import TemporalUnet

from modiff.dataset.representation_utils import TRAJ_DIM, MOTION_DIM


def mask_it(mask, cond, replace_token):
    # mask: B bool
    # cond: B x ... x D
    # replace_token: D
    expand_dims = [1] * (len(cond.shape) - len(mask.shape))
    mask = mask.view(*mask.shape, *expand_dims).float()  # B x ... x 1

    expand_dims = [1] * (len(cond.shape) - 1)
    mask_token = replace_token.view(*expand_dims, -1)  # 1 x ... x D
    cond = cond * (1.0 - mask) + mask * mask_token
    return cond


class UniEgoMotionBasic(nn.Module):
    def __init__(
        self,
        cfg,
        dropout=0.1,
    ):
        super().__init__()

        self.cfg = cfg

        latent_dim = 768
        ff_size = 768 * 2
        num_layers = 12
        num_heads = 12

        if cfg.MODEL.MODEL_SIZE == "small":
            latent_dim = 512
            ff_size = 512 * 2
            num_layers = 8
            num_heads = 8

        self.repre_type = cfg.DATA.REPRE_TYPE
        self.input_feats = MOTION_DIM[self.repre_type]
        traj_dim = TRAJ_DIM[self.repre_type]
        self.latent_dim = latent_dim

        if cfg.MODEL.LEARN_TRAJ:
            self.input_feats = traj_dim
            logger.warning("LEARNING TRAJ... USING SMALLER MODEL.")
            latent_dim = 512
            ff_size = 512 * 2
            num_layers = 8
            num_heads = 8

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # self.cond_mask_prob = cond_mask_prob
        self.cond_mask_prob = {
            "traj": 0.1,
            "text": 0.1,
        }

        self.input_process = nn.Linear(self.input_feats, self.latent_dim)
        self.pos_enc = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.pos_enc)
        self.embed_traj_cond = nn.Linear(traj_dim, self.latent_dim)

        self.mask_tokens = nn.ParameterDict(
            {
                "traj": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
            }
        )

        if self.cfg.DATA.COND_TEXT_FEAT:
            self.embed_text_cond = nn.Linear(512, self.latent_dim)
            self.mask_tokens["text"] = nn.Parameter(torch.randn(self.latent_dim) * 0.05)

        if self.cfg.MODEL.MODEL_NAME == "uem_basic":
            logger.warning("Using encoder.")
            self.tsfm = nn.ModuleList(
                [EncoderBlock(self.latent_dim, self.num_heads, self.dropout, 2) for _ in range(self.num_layers)]
            )
        elif self.cfg.MODEL.MODEL_NAME == "uem_unet":
            logger.warning("Using unet.")
            self.dim_mults = [1, 1, 1, 1]
            self.unet = TemporalUnet(
                self.latent_dim, self.latent_dim, self.latent_dim, self.dim_mults, adagn=True, zero=True
            )

        self.output_process = nn.Linear(self.latent_dim, self.input_feats)

    def mask_cond(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D

        # for inference of recon, fore and gen, masks are already provided.
        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        if not self.training:
            # technically, this should never happen because we always do inference with cond_mask.
            return cond

        # During training, mask whole condition randomly based on probability.
        # To simulate generation, we additionally mask subsequent frames (after first frame) with some probability.
        # Overall, this will simulate recon and gen tasks with some probability.

        inp_shape = cond.shape
        assert cond_type in ["traj", "text"]
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob[cond_type]  # B
        cond = mask_it(mask, cond, self.mask_tokens[cond_type])

        assert cond.shape == inp_shape
        return cond

    def forward(self, x, timesteps, y, cond_scale=None, diffusion=None):
        """
        x_t: B x T x F
        timesteps: B
        y: dict
        """
        if "valid_frames" not in y:
            y["valid_frames"] = torch.ones_like(x[:, :, 0])  # B x T

        if cond_scale is not None:
            assert diffusion is not None
            x_cond = self.forward(x, timesteps, y)  # conditional output
            x_uncond = self.forward(x, timesteps, {"valid_frames": y["valid_frames"]})  # unconditional output

            assert self.cfg.MODEL.PREDICT_XSTART
            x_scaled = x_uncond + (x_cond - x_uncond) * cond_scale
            return x_scaled

        B, T, F = x.shape
        x = self.input_process(x)  # B x T x D
        enc_time = self.embed_timestep(timesteps)  # B x D
        assert "betas" not in y

        # Traj
        if "traj" in y:  # B x T x F
            traj_mask = y["traj_mask"] if "traj_mask" in y else None
            enc_traj = self.embed_traj_cond(y["traj"])  # B x T x D
            enc_traj = self.mask_cond(enc_traj, "traj", traj_mask)
            x = x + enc_traj
        else:
            x = x + self.mask_tokens["traj"]

        # Text
        if self.cfg.DATA.COND_TEXT_FEAT:
            if "text_emb" in y:
                text_mask = y["text_mask"] if "text_mask" in y else None
                enc_text = self.embed_text_cond(y["text_emb"])  # B x D
                enc_text = self.mask_cond(enc_text, "text", text_mask)  # B x D
            else:
                enc_text = self.mask_tokens["text"].view(1, -1).repeat(B, 1)  # B x D
            enc_time = enc_time + enc_text

        if self.cfg.MODEL.MODEL_NAME == "uem_basic":
            x = self.pos_enc(x)  # B x T x D
            x = torch.cat((enc_time[:, None], x), axis=1)  # B x (T+1) x D
            mask = y["valid_frames"]  # B x T where valid are 1
            mask = torch.cat((torch.ones((B, 1), device=mask.device, dtype=mask.dtype), mask), dim=1)  # B x (T+1)
            mask = mask[:, None, None, :]  # B x 1 x 1 x (T+1)
            for enc in self.tsfm:
                x = enc(x=x, mask=mask)
            x = self.output_process(x[:, 1 : 1 + T])  # B x T x (J x F)

        elif self.cfg.MODEL.MODEL_NAME == "uem_unet":
            x = einops.rearrange(x, "b t d -> t b d")
            bT = int(np.ceil(T / 16)) * 16
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, bT - T), value=0)
            x = self.unet.forward(x, enc_time)
            x = x[:T]
            x = einops.rearrange(x, "t b d -> b t d")
            x = self.output_process(x)  # B x T x (J x F)
        else:
            raise ValueError(f"Unknown model name {self.cfg.MODEL.MODEL_NAME}")

        x = x.view(B, T, F)
        if "repaint_mask" in y:
            assert self.cfg.MODEL.PREDICT_XSTART
            repaint_mask = y["repaint_mask"]
            repaint_value = y["repaint_value"]
            x = x * (1 - repaint_mask) + repaint_mask * repaint_value

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe[None])  # 1 x Tmax x D

    def forward(self, x):
        # x is B x T x D
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, pos_enc):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_enc = pos_enc  # .pe is 1 x Tmax x D

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # timesteps is (B,)
        t = self.pos_enc.pe[0, timesteps]  # B x D
        return self.time_embed(t)  # B x D


if __name__ == "__main__":
    cfg = lambda: None
    cfg.DATA = lambda: None
    cfg.MODEL = lambda: None
    cfg.DATA.COND_IMG_FEAT = True
    cfg.DATA.COND_BETAS = True
    cfg.DATA.IMG_FEAT_TYPE = "clip_all"
    cfg.DATA.REPRE_TYPE = "v1"
    cfg.MODEL.LEARN_TRAJ = False

    model = UniEgoMotionBasic(cfg)
    x = torch.randn(2, 10, 224)
    timesteps = torch.tensor([342, 21])

    y = {
        "valid_frames": torch.ones(2, 10).long(),
        "traj": torch.randn(2, 10, 9),
    }
    out = model(x, timesteps, y)

    import IPython

    IPython.embed()
