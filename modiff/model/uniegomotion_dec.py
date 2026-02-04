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
    # mask: B bool denoting INVALID frames
    # cond: B x ... x D
    # replace_token: D
    expand_dims = [1] * (len(cond.shape) - len(mask.shape))
    mask = mask.view(*mask.shape, *expand_dims).float()  # B x ... x 1

    expand_dims = [1] * (len(cond.shape) - 1)
    mask_token = replace_token.view(*expand_dims, -1)  # 1 x ... x D
    cond = cond * (1.0 - mask) + mask * mask_token
    return cond


def pad_sequence(sequences, padding_value=0, pad_len=None):
    # sequences: B x [T x ...]
    # returns: B x T x ... and B x T int mask of valid frames

    if len(sequences) == 1:
        out_tensor = sequences[0][None]  # 1 x T x ...
        mask = torch.ones(1, out_tensor.shape[1], dtype=torch.long, device=out_tensor.device)  # 1 x T
        return out_tensor, mask

    trailing_dims = sequences[0].shape[1:]
    max_len = max(len(s) for s in sequences)
    if pad_len is not None:
        assert pad_len >= max_len
        max_len = max(max_len, pad_len)
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.long, device=sequences[0].device)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
        mask[i, :length] = 1

    return out_tensor, mask


class UniEgoMotionDec(nn.Module):
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
        self.text_emb_dim = {"distilbert": 768, "clip_vitb32": 512}[cfg.DATA.TEXT_ENCODER_NAME]

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
        self.embed_text_cond = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.mask_tokens = nn.ParameterDict(
            {
                "traj": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
                "text": nn.Parameter(torch.randn(self.latent_dim) * 0.05),
            }
        )

        self.tsfm = nn.ModuleList(
            [DecoderBlock(self.latent_dim, self.num_heads, self.dropout, 2) for _ in range(self.num_layers)]
        )
        self.output_process = nn.Linear(self.latent_dim, self.input_feats)

    def mask_cond(self, cond, cond_type, cond_mask=None):
        # cond: B x T x ... x D

        # for inference of recon, fore and gen, masks are already provided.
        if cond_mask is not None:
            cond = mask_it(cond_mask, cond, self.mask_tokens[cond_type])
            return cond

        if not self.training:
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
            y_uncond = copy.deepcopy(y)
            y_uncond["text_mask"] = torch.ones_like(x[:, 0, 0])  # B
            x_uncond = self.forward(x, timesteps, y_uncond)  # unconditional output

            assert self.cfg.MODEL.PREDICT_XSTART
            x_scaled = x_uncond + (x_cond - x_uncond) * cond_scale
            return x_scaled

        B, T, F = x.shape
        enc_time = self.embed_timestep(timesteps)  # B x D
        context = [enc_time[:, None]]  # B x 1 x D
        context_mask = [torch.ones(B, 1, dtype=torch.long, device=x.device)]  # B x 1

        assert "traj" not in y
        # # Traj
        # if "traj" in y:  # B x T x F
        #     traj_mask = y["traj_mask"] if "traj_mask" in y else None
        #     enc_traj = self.embed_traj_cond(y["traj"])  # B x T x D
        #     enc_traj = self.mask_cond(enc_traj, "traj", traj_mask)  # B x T x D
        # else:
        #     enc_traj = self.mask_tokens["traj"].view(1, 1, -1).repeat(B, T, 1)  # B x T x D

        assert "text_emb" in y
        if "text_emb" in y:
            text_emb = y["text_emb"]  # B x [T x D]
            text_emb, text_valid_tokens = pad_sequence(text_emb)  # B x T x D and B x T
            text_emb = self.embed_text_cond(text_emb)  # B x T x D
            text_mask = y["text_mask"] if "text_mask" in y else None
            text_emb = self.mask_cond(text_emb, "text", text_mask)  # B x T x D
            context.append(text_emb)
            context_mask.append(text_valid_tokens)

        context = torch.cat(context, dim=1)  # B x C x D
        context_mask = torch.cat(context_mask, dim=1)  # B x C
        context_mask = context_mask[:, None, None, :]  # B x 1 x 1 x C

        x = self.input_process(x)  # B x T x D
        mask = y["valid_frames"]  # B x T
        mask = mask[:, None, None, :]  # B x 1 x 1 x T

        x = self.pos_enc(x)  # B x T x D
        for dec in self.tsfm:
            x = dec(x=x, context=context, mask=mask, context_mask=context_mask)  # B x T x D
        x = self.output_process(x)  # B x T x F
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
    cfg.DATA.REPRE_TYPE = "v4"
    cfg.DATA.COND_TEXT_FEAT = True

    model = UniEgoMotionDec(cfg)
    x = torch.randn(2, 10, 224)
    timesteps = torch.tensor([342, 21])

    y = {
        "valid_frames": torch.ones(2, 10).long(),
        "traj": torch.randn(2, 10, 9),
    }
    out = model(x, timesteps, y)

    import IPython

    IPython.embed()
