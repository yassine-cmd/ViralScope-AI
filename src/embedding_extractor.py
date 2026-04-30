"""EmbeddingExtractor — SigLIP image/text embeddings + probe features."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import SiglipModel, SiglipProcessor


class EmbeddingExtractor:
    """Extract SigLIP embeddings and probe features, with disk caching."""

    def __init__(self, config, thumbnail_manager):
        self.config = config
        self.tm = thumbnail_manager
        self.tensor_dir = config["data"]["tensor_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.img_dim = config["model"]["embedding"]["image_dim"]
        self.txt_dim = config["model"]["embedding"]["text_dim"]
        self.max_seq_length = config["model"]["embedding"].get("max_seq_length", 64)
        self.PROBE_PAIRS = [
            (p[0], p[1]) for p in config["model"].get("probe_pairs", [])
        ]
        self.n_probes = len(self.PROBE_PAIRS)

        self.img_path = f"{self.tensor_dir}/img_embeddings.npy"
        self.txt_path = f"{self.tensor_dir}/txt_embeddings.npy"
        self.probe_path = f"{self.tensor_dir}/probe_features.npy"
        self.ids_path = f"{self.tensor_dir}/video_ids.npy"

    def _to_tensor(self, output):
        """Safely extract tensor from model output."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        raise TypeError(f"Cannot extract embedding tensor from {type(output)}")

    def extract(self, df, force_recompute=False):
        """Extract embeddings. Load from cache if available unless force_recompute."""
        if (not force_recompute and
            os.path.exists(self.img_path) and
            os.path.exists(self.txt_path) and
            os.path.exists(self.probe_path) and
            os.path.exists(self.ids_path)):

            cached_img = np.load(self.img_path)
            cached_txt = np.load(self.txt_path)
            cached_probe = np.load(self.probe_path)
            cached_ids = np.load(self.ids_path, allow_pickle=True).astype(str)

            id_to_idx = {vid: idx for idx, vid in enumerate(cached_ids)}

            indices = []
            for vid in df["video_id"]:
                if vid in id_to_idx:
                    indices.append(id_to_idx[vid])

            if indices:
                print(f"[Embed] Loaded {len(indices)} aligned embeddings from cache.")
                return cached_img[indices], cached_txt[indices], cached_probe[indices]

        checkpoint = self.config["model"]["embedding"]["checkpoint"]
        print(f"[Embed] Loading SigLIP : {checkpoint}")
        print(f"[Embed] Device         : {self.device}")
        model = SiglipModel.from_pretrained(checkpoint).to(self.device)
        processor = SiglipProcessor.from_pretrained(checkpoint)
        model.eval()

        n = len(df)
        vids = df["video_id"].tolist()
        titles = df["title"].fillna("Untitled").astype(str).tolist()

        img_embs = []
        txt_embs = []
        probe_feats = []
        vids_out = []
        all_missing = []

        print("[Embed] Pre-encoding probe prompts...")
        pos_prompts = [p[0] for p in self.PROBE_PAIRS]
        neg_prompts = [p[1] for p in self.PROBE_PAIRS]
        max_length = min(self.max_seq_length, getattr(processor, "model_max_length", self.max_seq_length))

        with torch.no_grad():
            pos_inp = processor(text=pos_prompts, return_tensors="pt",
                              padding="max_length", truncation=True, max_length=max_length).to(self.device)
            probe_pos = F.normalize(self._to_tensor(model.get_text_features(**pos_inp)), dim=-1)

            neg_inp = processor(text=neg_prompts, return_tensors="pt",
                              padding="max_length", truncation=True, max_length=max_length).to(self.device)
            probe_neg = F.normalize(self._to_tensor(model.get_text_features(**neg_inp)), dim=-1)

        for s in tqdm(range(0, n, self.batch_size), desc="Extracting SigLIP embeddings"):
            e = min(s + self.batch_size, n)
            v_ids = vids[s:e]
            txts = titles[s:e]

            imgs = []
            valid_vids = []
            valid_txts = []

            for i, vid in enumerate(v_ids):
                try:
                    imgs.append(self.tm.load_image(vid))
                    valid_vids.append(vid)
                    valid_txts.append(txts[i])
                except Exception:
                    all_missing.append(vid)

            if not imgs:
                continue

            with torch.no_grad():
                img_inp = processor(images=imgs, return_tensors="pt").to(self.device)
                if_norm = F.normalize(
                    self._to_tensor(model.get_image_features(**img_inp)), dim=-1)

                txt_inp = processor(text=valid_txts, return_tensors="pt",
                                   padding="max_length", truncation=True, max_length=max_length).to(self.device)
                txt_emb = F.normalize(
                    self._to_tensor(model.get_text_features(**txt_inp)), dim=-1)

                probe = (if_norm @ probe_pos.T - if_norm @ probe_neg.T).cpu().numpy()

            img_embs.extend(if_norm.cpu().numpy())
            txt_embs.extend(txt_emb.cpu().numpy())
            probe_feats.extend(probe)
            vids_out.extend(valid_vids)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        img_embs = np.array(img_embs, dtype=np.float32)
        txt_embs = np.array(txt_embs, dtype=np.float32)
        probe_feats = np.array(probe_feats, dtype=np.float32)

        if all_missing:
            print(f"[Embed] Skipped {len(all_missing):,} rows with missing thumbnails")
        print(f"[Embed] Kept {len(img_embs):,} valid samples (of {n:,} attempted)")

        np.save(self.img_path, img_embs)
        np.save(self.txt_path, txt_embs)
        np.save(self.probe_path, probe_feats)
        np.save(self.ids_path, np.array(vids_out, dtype=object))

        print(f"[Embed] Saved img_embeddings.npy  : {img_embs.shape}")
        print(f"[Embed] Saved txt_embeddings.npy  : {txt_embs.shape}")
        print(f"[Embed] Saved probe_features.npy  : {probe_feats.shape}")
        print("[Embed] Done. SigLIP is no longer loaded — subsequent steps use CPU.")

        return img_embs, txt_embs, probe_feats
