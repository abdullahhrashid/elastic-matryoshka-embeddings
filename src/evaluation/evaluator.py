from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np


class MRLModelWrapper:
    def __init__(self, model, tokenizer, dim, device, batch_size=64, max_length=128):
        self.model, self.tokenizer = model, tokenizer
        self.dim, self.device = dim, device
        self.batch_size, self.max_length = batch_size, max_length

    def encode(self, sentences, batch_size=None, show_progress_bar=True, **kwargs):
        bs = batch_size or self.batch_size
        all_embs = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), bs), disable=not show_progress_bar,
                          desc=f'Encoding dim={self.dim}'):
                enc = self.tokenizer(
                    sentences[i:i+bs], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors='pt'
                ).to(self.device)
                emb = self.model(enc)
                emb = F.normalize(emb[:, :self.dim], p=2, dim=1)
                all_embs.append(emb.cpu().float().numpy())
        return np.vstack(all_embs)
