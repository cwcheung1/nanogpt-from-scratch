"""Lesson 02: bigram baseline model.

The simplest possible neural language model: predict the next character
using only the current character, via a single learned lookup table. No
attention, no context beyond 1 token. This is the floor everything later
has to beat.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import vocab_size, get_batch, estimate_loss, decode

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each row i is directly the logits over "what comes after token i" —
        # no hidden layer, the embedding table *is* the whole model
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, C) — note: whole history recomputed,
            logits = logits[:, -1, :]  # but the model only ever looks at
            probs = F.softmax(logits, dim=-1)  # the last token anyway (bigram!)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"device: {device}")
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- before training ---")
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

    for it in range(max_iters):
        if it % eval_interval == 0:
            losses = estimate_loss(model, block_size, batch_size, device)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n--- after training ---")
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
