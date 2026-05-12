import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(vocabulary_size,16)
        self.hidden_layer = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        self.embeddings= self.embedding_layer(x)
        print( self.embeddings)
        self.averaged_embeddings= torch.mean(self.embeddings,axis=1)
        print("newwww")
        print( self.averaged_embeddings)
        self.hidden_layer_output=self.hidden_layer.forward( self.averaged_embeddings) 

        self.sigmoid_output= self.sigmoid.forward(self.hidden_layer_output)
        return torch.round(self.sigmoid_output,decimals=4)
        # Return a B, 1 tensor and round to 4 decimal places
        pass
