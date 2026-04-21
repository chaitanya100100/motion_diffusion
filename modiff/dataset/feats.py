import os
import torch
from loguru import logger
import json
import clip
from tqdm.auto import tqdm

from modiff.utils.torch_utils import to_tensor, print_dict
from modiff.dataset.amass_utils import process_text_anns


class ImageFeats(object):
    def __init__(self, data_dir, split, img_feat_type):
        self.data_dir = data_dir
        self.split = split
        self.img_feat_type = img_feat_type
        self.load_img_feats()

    def load_img_feats(self):

        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            processed_path = f"{self.data_dir}/uniegomotion/egoview_dinov2_{self.split}.pt"
            logger.info(f"Loading dinov2 feats from {processed_path}.")
            self.ego_img_feats = torch.load(processed_path, weights_only=False)
            return

        if self.img_feat_type == "egovideo":
            processed_path = f"{self.data_dir}/uniegomotion/egoview_egovideo_{self.split}.pt"
            logger.info(f"Loading egovideo feats from {processed_path}.")
            self.ego_img_feats = torch.load(processed_path, weights_only=False)
            self.ego_img_feats = to_tensor(self.ego_img_feats)
            return

        if self.img_feat_type in ["clip_all"]:
            processed_path = f"{self.data_dir}/uniegomotion/egoview_clip_{self.split}.pt"
            logger.info(f"Loading clip feats from {processed_path}.")
            self.ego_img_feats = torch.load(processed_path, weights_only=False)
            return

        raise AttributeError

    def get_img_feats(self, seq_name, start_idx, end_idx):
        take_name, st, en = seq_name.split("___")
        start_idx = start_idx + int(st)
        end_idx = end_idx + int(st)
        assert end_idx <= int(en)
        del seq_name

        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            start_idx = start_idx // 3
            end_idx = end_idx // 3
            embs = (
                self.ego_img_feats["feats"][take_name][start_idx // 2 : end_idx // 2 + 1].clone().float()
            )  # T/2 x 5 x 1024 because dino feats are at 5 fps

            # duplicate the embs to make it 10 fps
            # assert start_idx % 2 == 0
            embs = embs[:, None].repeat(1, 2, 1, 1).flatten(0, 1)  # T x 5 x 1024
            if start_idx % 2 != 0:
                embs = embs[1:]

            if embs.shape[0] == (end_idx - start_idx + 2):  # handle partial segment
                embs = embs[:-1]

            assert embs.shape[0] == (end_idx - start_idx + 1)

            valid_img_embs = torch.ones(embs.shape[0], dtype=torch.long)
            if self.img_feat_type == "dinov2":  # remove register tokens
                embs = embs[:, 0]  # T x 1024
            return embs, valid_img_embs

        if self.img_feat_type == "egovideo":

            take_feats = self.ego_img_feats["feats"][take_name]
            feat_st = self.ego_img_feats["start_frame"][take_name]
            feat_en = self.ego_img_feats["end_frame"][take_name]
            feat_idx = ((feat_st + feat_en) // 2).long()

            q_idx = torch.arange(start_idx, end_idx + 1, 3, dtype=torch.long)

            # egovideo features are at short video segment level, not per frame
            # duplicate the features to make it per frame
            embs = batched_linear_interpolate(take_feats[None], feat_idx[None], q_idx[None])[0]
            assert embs.shape[0] == (end_idx // 3 - start_idx // 3 + 1)
            embs = torch.nn.functional.normalize(embs.float(), dim=-1)
            valid_img_embs = torch.ones(embs.shape[0], dtype=torch.long)

            return embs, valid_img_embs

        if self.img_feat_type in ["clip_all"]:
            start_idx = start_idx // 3
            end_idx = end_idx // 3
            embs = self.ego_img_feats["feats"][take_name][start_idx : end_idx + 1].clone().float()
            assert embs.shape[0] == (end_idx - start_idx + 1)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            valid_img_embs = torch.ones(embs.shape[0], dtype=torch.long)
            return embs, valid_img_embs
        raise AttributeError


class TextFeats(object):
    def __init__(self, text_encoder_name):
        self.text_encoder_name = text_encoder_name
        assert text_encoder_name in ["clip_vitb32", "distilbert"]
        self.text_anns_path = "/vision/u/chpatel/data/modiff/babel_humanml3d_kitml_with_mirror_smplx.json"
        self.text_anns = None
        self.preencoded_feats_path = f"/vision/u/chpatel/data/modiff/text_feats_{text_encoder_name}_bhk_with_mirror.pt"
        self.preencoded_feats = None

        self.idx_to_info = None

    # Setter function for annotations and preencoded features
    def prepare_for_training(self, seq_names, window):
        self.text_anns = self.load_text_anns()
        self.preencoded_feats = self.load_preencoded_text_feats()
        fps = 10

        self.idx_to_info = []
        for seq_name, anns in self.text_anns.items():
            # val dataset will have only val sequences
            if seq_name not in seq_names:
                continue
            nf = int(anns["duration"] * fps)
            for ann in anns["annotations"]:
                st = int(ann["start"] * fps)
                en = int(ann["end"] * fps) - 1
                if en - st + 1 > window:  # ignore long annotations
                    continue
                if en - st + 1 < 20:  # ignore short annotations
                    continue
                text = ann["text"]
                if en >= nf:  # find problematic annotations
                    logger.warning(f"en >= nf for sequence {seq_name}")
                    en = nf - 1
                self.idx_to_info.append((seq_name, st, en, text))

        logger.info(f"Total number of text annotations: {len(self.idx_to_info)}")

    def __len__(self):
        return len(self.idx_to_info)

    def load_text_anns(self):
        with open(self.text_anns_path, "r") as f:
            text_anns = json.load(f)
        text_anns = process_text_anns(text_anns)
        return text_anns

    def preencode_all_texts(self):
        assert not os.path.exists(self.preencoded_feats_path)
        text_anns = self.load_text_anns()

        texts = []
        for key, seq_ann in text_anns.items():
            for a in seq_ann["annotations"]:
                texts.append(a["text"])
        texts = list(set(texts))

        device = torch.device("cuda")
        model = self.load_text_model(device)
        bs = 1
        out_dict = {"idx_mapping": {}, "feats": []}
        idx = 0
        for i in tqdm(range(0, len(texts), bs)):
            texts_batch = texts[i : i + bs]
            feats = self.encode_text(model, texts_batch, device)
            feats = feats.detach().cpu()
            out_dict["feats"].append(feats)

            for t in texts_batch:
                out_dict["idx_mapping"][t] = idx
                idx += 1

        if self.text_encoder_name == "clip_vitb32":
            out_dict["feats"] = torch.cat(out_dict["feats"], dim=0)
        elif self.text_encoder_name == "distilbert":
            out_dict["feats"] = [x[0] for x in out_dict["feats"]]

        print_dict(out_dict)
        torch.save(out_dict, self.preencoded_feats_path)

    def load_preencoded_text_feats(self):
        assert os.path.exists(self.preencoded_feats_path)
        logger.info(f"Loading preencoded text feats from {self.preencoded_feats_path}.")
        return torch.load(self.preencoded_feats_path, weights_only=False)

    def get_text_feat(self, text):
        idx = self.preencoded_feats["idx_mapping"][text]
        x = self.preencoded_feats["feats"][idx]
        return x.float()

    def load_text_model(self, device):
        if self.text_encoder_name == "clip_vitb32":
            logger.info(f"Loading CLIP ViT-B/32 model for text encoding.")
            clip_model, clip_process = clip.load("ViT-B/32", device=device, jit=False)
            clip_model.eval()
            model = clip_model
        elif self.text_encoder_name == "distilbert":
            logger.info(f"Loading DistilBERT model for text encoding.")
            from transformers import AutoTokenizer, AutoModel
            from transformers import logging

            logging.set_verbosity_error()
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
            text_model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
            text_model.eval().to(device)
            model = (text_model, tokenizer)
        return model

    def encode_text(self, model, texts, device):
        batched = True
        if isinstance(texts, str):
            texts = [texts]
            batched = False
        if self.text_encoder_name == "clip_vitb32":
            tokens = clip.tokenize(texts, truncate=True).to(device)
            feats = model.encode_text(tokens)  # B x D
        elif self.text_encoder_name == "distilbert":
            text_model, tokenizer = model
            assert len(texts) == 1
            encoded_inputs = tokenizer(texts, return_tensors="pt", padding=True)
            feats = text_model(**encoded_inputs.to(device)).last_hidden_state  # B x T x D
            assert torch.all(encoded_inputs.attention_mask.to(dtype=bool))
        if not batched:
            feats = feats[0]
        return feats


def batched_linear_interpolate(A, t_i, t):
    """
    Efficient batched linear interpolation in PyTorch.

    Args:
        A: Tensor of shape (B, T, D) - batch of time series data.
        t_i: Tensor of shape (B, T) - time points corresponding to A.
        t: Tensor of shape (B, N) - query timestamps to interpolate.

    Returns:
        A_interp: Tensor of shape (B, N, D) - interpolated values.
    """
    B, T, D = A.shape  # Batch size, time steps, feature dimension
    _, N = t.shape  # Number of query points per batch

    # Ensure time steps are sorted
    assert torch.all(t_i[..., :-1] <= t_i[..., 1:]), "t_i must be sorted in increasing order"

    # Find indices where `t` would be inserted in `t_i`
    indices = torch.searchsorted(t_i, t, right=True)  # (B, N)

    # Get lower and upper bounds, clamping to valid indices
    lower_indices = torch.clamp(indices - 1, 0, T - 1)
    upper_indices = torch.clamp(indices, 0, T - 1)

    # Gather corresponding time values and A values
    t_lower = torch.gather(t_i, 1, lower_indices)  # (B, N)
    t_upper = torch.gather(t_i, 1, upper_indices)  # (B, N)
    A_lower = torch.gather(A, 1, lower_indices.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)
    A_upper = torch.gather(A, 1, upper_indices.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)

    # Compute interpolation weights
    t_range = (t_upper - t_lower).clamp(min=1e-6)  # Avoid division by zero
    alpha = (t - t_lower) / t_range  # (B, N)

    # Compute interpolated values
    A_interp = (1 - alpha.unsqueeze(-1)) * A_lower + alpha.unsqueeze(-1) * A_upper  # (B, N, D)

    return A_interp


if __name__ == "__main__":
    text_feats = TextFeats(text_encoder_name="distilbert")
    text_feats.preencode_all_texts()
