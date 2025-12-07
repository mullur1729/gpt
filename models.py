
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import block_size, n_embd, device, nlayers, nheads, dropout

class Head(nn.Module):
    """ Single attention head """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple attention heads in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple Linear layer followed by RELU """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class AttentionBlock(nn.Module):
    """ A single attention block: communication followed by computation """

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.ma = MultiHeadAttention(n_heads, head_size) # 4 heads of attention, each of size n_embd // n_heads
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        # Departure from the original Transformer architecture: we add layernorms and use pre-norm instead of post-norm
        x = x + self.ma(self.ln1(x)) # Communication
        x = x + self.ffwd(self.ln2(x)) # Computation
        return x


class GPTLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[AttentionBlock(n_embd, n_heads=nheads) for _ in range(nlayers)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # self.sa_head = Head(n_embd) # Single attention head

        # self.ma_heads = MultiHeadAttention(4, n_embd // 4) # 4 heads of attention, each of size n_embd // 4
        # self.fwd = FeedForward(n_embd)
    
    def forward(self, idx, targets=None):

        B, T = idx.shape
        # B - batch size or number of examples
        # T - time steps or sequence length or context length
        # C - number of channels or embedding dimension

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)

        # x = self.sa_head(x) # (B, T, C) - apply one head of self attention

        # x = self.ma_heads(x) # (B, T, C) - apply multi-head self attention
        # x = self.fwd(x) # (B, T, C) - apply MLP layer

        x = self.blocks(x) # (B, T, C) - apply stack of attention blocks
        x = self.ln_f(x) # (B, T, C) - final layer norm

        logits = self.lm_head(x) # (B, T, vocab_size)

        # logits is the predicted next character scores (size of vocab_size or C) for each index in idx, which is of shape (B, T)
        # logits = self.token_embedding_table(idx) # (B, T, C)
        
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
            idx_cond = idx[:, -block_size:] # crop context to the last block_size tokens
            logits, loss = self(idx_cond)
            # focus only on the last time step (meaning only the last character in the context, 
            # since this is a bigram model that depends only on the last character in the sequence)
            # that's why the index for the "T" dimension or the time dimension is -1
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence
        return idx
    

