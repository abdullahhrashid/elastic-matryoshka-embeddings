import torch
import torch.nn as nn
import torch.nn.functional as F

class MatryoshkaInfoNCELoss(nn.Module):
    def __init__(self, temperature, dims):
        super().__init__()
        self.temperature = temperature
        self.dims = dims
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, anchor_embs, positive_embs, negative_embs):
        batch_size = anchor_embs.shape[0]
        device = anchor_embs.device

        total_loss = 0

        for dim in self.dims:
            #slicing to the first d dimensions and normalizing
            a = F.normalize(anchor_embs[:, :dim], p=2, dim=1)     
            p = F.normalize(positive_embs[:, :dim], p=2, dim=1)   
            n = F.normalize(negative_embs[:, :dim], p=2, dim=1)   

            sim_matrix = (a @ p.T) / self.temperature            

            #in batch negatives + hard negatives
            hard_neg_scores = (a * n).sum(dim=1, keepdim=True) / self.temperature 
            logits = torch.cat([sim_matrix, hard_neg_scores], dim=1)              

            labels = torch.arange(batch_size, device=device)

            total_loss += self.cross_entropy(logits, labels)

        return total_loss
