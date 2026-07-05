"""Lesson 04: multi-head attention.

Wire lesson 3's single Head into several heads running in parallel, and for
the first time plug attention into an actual trainable language model (with
position embeddings, since attention itself has no notion of order) so we
can compare its loss against lesson 2's bigram baseline.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import vocab_size, get_batch, estimate_loss, decode

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 8
batch_size = 32
n_embd = 32
n_head = 4
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Several independent attention heads in parallel, each looking for
    different kinds of relationships, then concatenated and linearly
    projected back to n_embd. Splitting one big head into several small ones
    (head_size = n_embd // n_head) costs nothing extra in parameters — it's
    the same total width, just able to represent several distinct attention
    patterns simultaneously instead of one averaged pattern."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class MultiHeadLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # attention has no built-in sense of position (unlike an RNN's
        # recurrence) — without this, "cat sat" and "sat cat" would produce
        # identical attention patterns. A learned vector per position, added
        # to the token embedding, is how the model recovers order.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, n_embd // n_head)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd), broadcast over batch
        x = self.sa_heads(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # attention needs a fixed-size window now
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = MultiHeadLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"device: {device}")
    print(f"params: {sum(p.numel() for p in model.parameters())}  (lesson 2 bigram had 4225)")

    for it in range(max_iters):
        if it % eval_interval == 0:
            losses = estimate_loss(model, block_size, batch_size, device)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n--- sample after training ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
    print("\n(compare val loss above to lesson 2's ~2.48-2.50 plateau)")
