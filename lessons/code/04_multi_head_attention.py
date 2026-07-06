"""Lesson 04: multi-head attention — the first TRAINED attention model.

Wire lesson 3's single Head (unchanged — see that file for the detailed
query/key/value/masking comments) into several heads running in parallel,
and for the first time plug attention into an actual trainable language
model, with position embeddings added (since attention itself has no notion
of order — see the lesson doc). This is the first head-to-head loss
comparison against lesson 2's bigram baseline.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import vocab_size, get_batch, estimate_loss, decode

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 8    # context length — see lesson 1 / roadmap
batch_size = 32
n_embd = 32       # width of every position's embedding vector
n_head = 4        # how many attention heads run in parallel (see MultiHeadAttention)
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3


class Head(nn.Module):
    """Identical to lesson 3's Head class — same query/key/value projections,
    same scaled dot product, same causal mask. See lessons/code/
    03_self_attention.py for the full line-by-line explanation; nothing
    about the mechanism itself changed, it's just being trained now."""

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
    """Several independent Head instances run in parallel — each learns its
    own query/key/value projections, so each can specialize in noticing a
    different kind of relationship between positions. Their outputs are
    glued together (torch.cat along the last dimension) and passed through
    one more nn.Linear ("proj") that mixes/blends the heads' separate
    findings back into one n_embd-wide vector per position.

    head_size = n_embd // n_head (set by the caller below) means: split one
    32-wide embedding into 4 heads of size 8 each. Concatenating 4 heads of
    size 8 gets back to width 32 — the SAME total parameter budget as one
    32-wide head, just restructured into 4 independent, smaller
    "conversations" instead of one big one."""

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
        # Attention has no built-in sense of position — the query/key dot
        # product only sees each position's CONTENT, not where it sits in
        # the sequence (see the lesson doc's "cat sat" vs "sat cat"
        # example). This table gives each position 0..block_size-1 its own
        # learned vector purely encoding "where am I," independent of
        # what character is there.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head, n_embd // n_head)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # projects back to per-character logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd) — "what character is here"
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd) — "where is here"
        # Add the two together: every position's vector now encodes BOTH
        # "which character" and "which position" simultaneously. pos_emb
        # (shape (T, n_embd)) broadcasts across the batch dimension — the
        # same position-vectors apply to every sequence in the batch.
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.sa_heads(x)  # let positions exchange information via attention
        logits = self.lm_head(x)  # (B, T, vocab_size) — back to per-character scores

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # NEW vs. lesson 2: crop to the last block_size characters
            # before every forward pass. The bigram model could ignore this
            # because it only ever used the very last token regardless of
            # how much history you handed it. This model's
            # position_embedding_table only HAS block_size rows (positions
            # 0..block_size-1) — feed it a longer sequence and there's no
            # position embedding to look up for the extra positions. This is
            # your first hard "context window" limit, not just a training
            # convenience.
            idx_cond = idx[:, -block_size:]
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
