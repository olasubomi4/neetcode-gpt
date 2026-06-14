import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        
        # Create three linear projections (Key, Query, Value) with bias=False
        self.key_model = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.query_model = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.value_model = nn.Linear(embedding_dim,attention_dim,bias=False)

        self.attention_dim= attention_dim
        self.embedding_dim= embedding_dim
        # Instantiation order matters for reproducible weights: key, query, value
        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layer
        key_output= self.key_model(embedded)
        query_output= self.query_model(embedded)
        value_output= self.value_model(embedded)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)

        dot_result= query_output @ torch.transpose(key_output,1,2)
        scores= dot_result/(self.attention_dim **0.5)

        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        tril = torch.tril(torch.ones(scores.shape[1], scores.shape[1]))
        scores = scores.masked_fill(tril == 0, float('-inf'))
        
        # 4. Apply softmax(dim=2) to masked scores
        scores= nn.functional.softmax(scores,dim=-1)
        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round(scores @ value_output,decimals=4)
