
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        # B - batch size or number of examples
        # T - time steps or sequence length or context length
        # C - number of channels or embedding dimension

        # logits is the predicted next character scores (size of vocab_size or C) for each index in idx, which is of shape (B, T)
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # focus only on the last time step (meaning only the last character in the context, 
            # since this is a bigram model that depends only on the last character in the sequence)
            # that's why the index for the "T" dimension or the time dimension is -1
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence
        return idx