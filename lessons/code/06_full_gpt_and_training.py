"""Lesson 06: the full GPT, scaled up, trained properly, and sampled from.

Same architecture as lesson 5's Block/GPTLanguageModel, but: bigger
(n_embd=384, n_head=6, n_layer=6, block_size=256), with dropout for
regularization, gradient-clipped AdamW with the config nanoGPT/GPT-2 use,
and a real training run long enough to produce recognizably
Shakespeare-*shaped* (not coherent, but structurally plausible) text.
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
        wei = self.dropout(wei)  # randomly zero some attention weights during training,
        return wei @ v  # so the model can't over-rely on any single position's signal


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2's init scheme: small std normal, zero bias — not load-bearing
        # for this small a model, but standard practice worth knowing.
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

    @torch.no_grad()
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
