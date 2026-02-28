from transformers import AutoModel
import torch.nn as nn
import torch

class EmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    #a function for pooling our embeddings into one unified representaion
    def mean_pool(self, embeddings, attention_mask):
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return (embeddings * expanded_attention_mask).sum(1) / torch.clamp(expanded_attention_mask.sum(1), min=1e-9)
        
    def forward(self, encoded_input):
        model_output = self.model(**encoded_input)

        #getting the embedding of the entire sequence
        embeddings = self.mean_pool(model_output.last_hidden_state, encoded_input['attention_mask'])
        
        return embeddings
