import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        input_num=784
        self.first_layer= nn.Linear(input_num,512)
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(p=0.2)
        self.final_layer = nn.Linear(512,10)
        self.sigmoid= nn.Sigmoid()
        # Define the architecture here
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        self.first_layer_output= self.dropout(self.relu(self.first_layer.forward(images)))
        self.final_layer_output= self.sigmoid(self.final_layer.forward(self.first_layer_output))
        return torch.round(self.final_layer_output,decimals=4)
        # Return the model's prediction to 4 decimal places
