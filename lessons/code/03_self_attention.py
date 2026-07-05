"""Lesson 03: self-attention mechanics.

Part A: "the trick" — averaging past tokens efficiently via a masked matmul,
built up in three equivalent forms so you can see softmax attention is just
a generalization of an average.

Part B: a real single self-attention head (Q/K/V, scaled dot product,
causal mask) applied to random input, so we can inspect the actual attention
weight matrix before this is ever wired into a trainable model.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 32  # batch, time (sequence position), channels (embedding dim)


def trick_v1_loop(x):
    """Naive: for each position t, average the embeddings of all tokens <= t.
    Correct but O(T^2) python loop — this IS what we want conceptually."""
    out = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            out[b, t] = x[b, : t + 1].mean(0)
    return out


def trick_v2_matmul(x):
    """Same result, vectorized: a lower-triangular matrix of 1s, row-normalized,
    IS the 'average everything up to and including me' operator."""
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    return wei @ x  # (T, T) @ (B, T, C) broadcasts to (B, T, C)


def trick_v3_softmax(x):
    """Same result again, via softmax — this is the form that generalizes:
    instead of hard-coding 'equal weight to everyone before me', let the
    model LEARN the weights. Masking future positions with -inf before
    softmax is what forces those weights to become exactly 0."""
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    return wei @ x


class Head(nn.Module):
    """One self-attention head: every position emits a query ('what am I
    looking for'), a key ('what do I contain'), and a value ('what do I
    communicate if attended to'). Affinity = query . key; output = affinity-
    weighted sum of values."""

    def __init__(self, head_size, n_embd=C, block_size=T):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        head_size = k.shape[-1]

        wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)  # (B, T, head_size)
        return wei @ v, wei  # (B, T, head_size), and the weights for inspection


if __name__ == "__main__":
    x = torch.randn(B, T, C)

    out1, out2, out3 = trick_v1_loop(x), trick_v2_matmul(x), trick_v3_softmax(x)
    print("v1 (loop) vs v2 (matmul) match:", torch.allclose(out1, out2, atol=1e-6))
    print("v2 (matmul) vs v3 (softmax) match:", torch.allclose(out2, out3, atol=1e-6))

    head_size = 16
    head = Head(head_size)
    out, wei = head(x)
    print(f"\nHead output shape: {tuple(out.shape)}  (expect ({B}, {T}, {head_size}))")

    print("\nattention weights for batch element 0 (rows=query pos, cols=key pos):")
    torch.set_printoptions(precision=2, sci_mode=False)
    print(wei[0])

    is_causal = torch.allclose(torch.triu(wei[0], diagonal=1), torch.zeros(T, T))
    row_sums = wei[0].sum(dim=-1)
    print(f"\nupper triangle is all zero (causal): {is_causal}")
    print(f"each row sums to 1: {torch.allclose(row_sums, torch.ones(T))}")
