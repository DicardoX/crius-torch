import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGPTModel(nn.Module):
    def __init__(self, config):
        super(SimpleGPTModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads, dim_feedforward=config.hidden_size, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        for layer in self.transformer_layers:
            embeddings = layer(embeddings, src_key_padding_mask=attention_mask)
        output = self.fc(embeddings)
        return output
    
class SimpleGPTConfig:
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout    

