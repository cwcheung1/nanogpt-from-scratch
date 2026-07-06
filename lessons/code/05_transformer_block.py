"""Lesson 05: the transformer block.

Wrap multi-head attention (unchanged from lesson 4) with a feedforward MLP,
residual ("skip") connections, and pre-LayerNorm, then stack several of
these blocks. Includes a deliberate ablation: the same architecture trained
WITH and WITHOUT residual connections, to empirically show why they matter
rather than just asserting it. "Residual" and "LayerNorm" are defined in
plain language in lessons/05-transformer-block.md — read that first if
either term is new.
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
    """Unchanged from lessons 3/4 — see 03_self_attention.py for the full
    query/key/value/scaled-dot-product/causal-mask explanation."""

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
    """Unchanged from lesson 4 — several Head instances in parallel,
    concatenated, then linearly projected back to n_embd width."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """NEW in this lesson. Per-position MLP: attention (above) is how
    positions exchange information WITH EACH OTHER; this is where the model
    actually 'thinks' about what it just gathered, independently at each
    position (no information crosses between positions here — every
    position runs through this same small network on its own).
    Linear -> ReLU -> Linear: the first Linear expands each position's
    n_embd-wide vector out to 4x as many numbers (more room to compute
    with), ReLU (a simple nonlinearity: replace every negative number with
    0, leave positive numbers unchanged) lets the network represent
    non-straight-line relationships instead of just weighted sums, and the
    second Linear projects back down to n_embd so the output can be added
    back onto the residual stream (see Block below). The 4x expand-then-
    contract shape matches the original Transformer paper and GPT-2."""

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
    """One transformer block: communicate (attention) then compute (MLP),
    each wrapped in a residual connection and preceded by LayerNorm.
    "Pre-norm" means LayerNorm runs BEFORE attention/feedforward, not after
    — the original 2017 "Attention Is All You Need" paper used post-norm,
    but pre-norm trains more stably at depth, which is why GPT-2 and every
    modern LLM uses it instead."""

    def __init__(self, n_embd, n_head, use_residual=True):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # normalizes before attention
        self.ln2 = nn.LayerNorm(n_embd)  # normalizes before the MLP
        self.use_residual = use_residual

    def forward(self, x):
        if self.use_residual:
            # x + self.sa(...): the ORIGINAL x survives untouched, with the
            # attention output added on top — not replaced by it. Same
            # pattern for the feedforward step. See the lesson doc for why
            # this "+x" matters for training something this deep.
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            # THE ABLATION: no skip connection — each step's output
            # REPLACES x instead of adding to it. Everything else (weights
            # init, seed, hyperparameters, training loop) is identical to
            # the WITH-residual path above; this is the ONLY thing that
            # changes. Watch the results table in the lesson doc for what
            # this one change does to val loss.
            x = self.sa(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Same token+position embedding setup as lesson 4, but now the middle
    of the model is n_layer stacked Blocks (this is where "depth" comes
    from) instead of a single attention layer, plus one final LayerNorm
    (ln_f) before the output head."""

    def __init__(self, use_residual=True):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # n_layer=4 separate Block instances, chained one after another —
        # nn.Sequential just means "run x through each of these in order."
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
    """Run the standard training loop (see roadmap glossary) for one model
    variant, printing train/val loss periodically. Called twice below with
    use_residual=True and False — same seed both times, so the only thing
    that can explain a difference in results is that one flag."""
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
