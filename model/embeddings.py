import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        i=0
        table={}
        for embedding in embeddings:
            table[i]=embedding
            i=i+1
        result=[]
        for token_id in token_ids:
            result.append(table[token_id])
        
        result=np.array(np.round(result,5))
        return np.array(result)
