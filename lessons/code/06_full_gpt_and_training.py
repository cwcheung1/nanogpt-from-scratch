"""Lesson 06: the full GPT, scaled up, trained properly, and sampled from.

Same architecture as lesson 5's Block/GPTLanguageModel (unchanged
attention/MLP/residual/pre-LN pattern), but: bigger (n_embd=384, n_head=6,
n_layer=6, block_size=256), with dropout for regularization, gradient-
clipped AdamW with the config nanoGPT/GPT-2 use, and a real training run
long enough to produce recognizably Shakespeare-*shaped* (not coherent, but
structurally plausible) text. "Dropout", "gradient clipping", and "GPT-2-
style init" are defined in plain language in
lessons/06-full-gpt-and-training.md — read that first if any is new.
"""
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from common import vocab_size, get_batch, estimate_loss, decode

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- the "full" nanoGPT-tutorial config (Karpathy's char-level GPT) ---
block_size = 256
batch_size = 64
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_iters = int(os.environ.get("MAX_ITERS", 5000))
eval_interval = 500
eval_iters = 100
learning_rate = 3e-4
grad_clip = 1.0


class Head(nn.Module):
    """Same query/key/value/scaled-dot-product/causal-mask mechanism as
    lessons 3-5 (see 03_self_attention.py for the full explanation) — the
    only addition is a Dropout applied to the attention weights themselves."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # NEW vs. lesson 5: randomly zero some attention weights during
        # training (see lesson doc for what dropout is/why) — this fires
        # AFTER softmax, so the zeroed-out weights simply stop contributing
        # to the weighted average of values below, for this one forward pass.
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Same as lesson 4/5's MultiHeadAttention, plus a Dropout on the final
    projection's output."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Same Linear -> ReLU -> Linear as lesson 5 (see that file for the full
    explanation), plus a Dropout after the second Linear."""

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


class Block(nn.Module):
    """Identical to lesson 5's Block WITH residuals (the ablation's
    use_residual=False path is gone here — this lesson is about scaling up
    the version that works, not re-running the ablation)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Identical structure to lesson 5's GPTLanguageModel — token+position
    embeddings, n_layer stacked Blocks, final LayerNorm, output head — plus
    an explicit weight-initialization step (_init_weights, below) that
    lesson 5 left at PyTorch's defaults."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.apply(fn) calls fn once on every submodule in this model —
        # the mechanism that lets _init_weights below reach every Linear and
        # Embedding layer, no matter how deeply nested, without manually
        # listing them all out.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2's init scheme (see lesson doc): every Linear/Embedding's
        weights start as small random values from a mean-0, std-0.02 normal
        (bell-curve) distribution, and every Linear's bias starts at exactly
        0. Not load-bearing for a model this small — PyTorch's own default
        init would likely train fine too — but this is the actual scheme
        GPT-2/nanoGPT use, worth having as muscle memory."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()  # generation doesn't need gradients — no training happening here
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = GPTLanguageModel().to(device)
    print(f"device: {device}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"config: block_size={block_size} n_embd={n_embd} n_head={n_head} n_layer={n_layer} dropout={dropout}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    t0 = time.time()
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, block_size, batch_size, device, eval_iters=eval_iters)
            elapsed = time.time() - t0
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ({elapsed:.0f}s elapsed)")

        xb, yb = get_batch("train", block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # NEW vs. lesson 5: cap the overall size of this step's gradients
        # before applying them (see lesson doc for what/why) — cheap
        # insurance against one unusually extreme batch destabilizing
        # training at this scale.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    total_time = time.time() - t0
    print(f"\ntotal training time: {total_time:.0f}s for {max_iters} iters")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/nanogpt_shakespeare.pt")
    print("saved checkpoints/nanogpt_shakespeare.pt")

    print("\n--- 500 characters, sampled from scratch ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
