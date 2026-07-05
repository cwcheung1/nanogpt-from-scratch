"""Lesson 05: the transformer block.

Wrap multi-head attention with a feedforward MLP, residual ("skip")
connections, and pre-LayerNorm, then stack several of these blocks. Includes
a deliberate ablation: the same architecture trained WITH and WITHOUT
residual connections, to empirically show why they matter rather than just
asserting it.
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
n_layer = 4
max_iters = 3000
eval_interval = 500
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Per-position MLP: attention is how tokens exchange information with
    each other; the feedforward is where the model actually 'thinks' about
    what it gathered, independently at each position. The 4x expansion
    (n_embd -> 4*n_embd -> n_embd) matches the original Transformer paper and
    GPT-2 — more capacity to compute in, projected back down to stay
    residual-stream-compatible."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (MLP),
    each wrapped in a residual connection and preceded by LayerNorm
    (pre-norm, matching GPT-2 — the original 2017 Transformer paper used
    post-norm, but pre-norm trains more stably at depth, which is why every
    modern LLM uses it)."""

    def __init__(self, n_embd, n_head, use_residual=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.use_residual = use_residual

    def forward(self, x):
        if self.use_residual:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            # ablation: no skip connection — each block's output REPLACES
            # its input instead of adding to it. Gradients must flow back
            # through every block's transformation with nothing to skip
            # through, which gets harder as depth (n_layer) increases.
            x = self.sa(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, use_residual=True):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, use_residual=use_residual) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def train(use_residual, label):
    torch.manual_seed(1337)
    model = GPTLanguageModel(use_residual=use_residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"\n=== {label} (params: {sum(p.numel() for p in model.parameters())}) ===")
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, block_size, batch_size, device)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch("train", block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


if __name__ == "__main__":
    print(f"device: {device}, n_layer: {n_layer}, n_head: {n_head}, n_embd: {n_embd}")

    model_with = train(use_residual=True, label="WITH residual connections")
    model_without = train(use_residual=False, label="WITHOUT residual connections")

    print("\n--- sample from the WITH-residual model ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model_with.generate(context, max_new_tokens=200)[0].tolist()))

    print("\n(compare final val losses above — WITH should beat lesson 4's ~2.33-2.36,")
    print(" WITHOUT should train noticeably worse/slower at this depth)")
