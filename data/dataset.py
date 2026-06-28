import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        str_list= raw_dataset.split()
        max_starting_position= len(str_list)-context_length
        X=[]
        Y=[]
        torch.manual_seed(0)

        for i in range(0,batch_size):
            start= torch.randint(0,max_starting_position,(1,)).item()
            X.append(str_list[start:start+context_length])
            Y.append(str_list[start+1:start+1+context_length])

        return (X,Y)