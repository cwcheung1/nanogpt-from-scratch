"""Lesson 02: bigram baseline model — the first actual model in this repo.

The simplest possible neural language model: predict the next character
using only the single character right before it, via one learned lookup
table. No attention, no memory of anything further back. This is the floor
— every later lesson's whole point is beating this model's loss number.

Term definitions (embedding, logits, loss, training loop, batch/block size)
live in lessons/00-roadmap.md — read that first if any of this is unfamiliar.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import vocab_size, get_batch, estimate_loss, decode

# Fixes every "random" number below (batch sampling, weight init) to a
# specific reproducible sequence — otherwise two runs of this exact code
# would get different random batches and you couldn't tell whether a change
# you made actually helped or you just got a luckier draw.
torch.manual_seed(1337)
# Use the GPU if one's available (much faster), otherwise fall back to CPU —
# every lesson from here on follows this same pattern.
device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 8      # characters of history per example (see roadmap glossary)
batch_size = 32     # independent examples processed per training step
max_iters = 3000    # how many training-loop steps to run
eval_interval = 300  # print train/val loss every this many steps
learning_rate = 1e-2  # how big a step the optimizer takes each update


# Inheriting from nn.Module is what lets PyTorch automatically discover
# which numbers inside this object are learnable parameters, so the
# optimizer below can find and update them without any manual bookkeeping.
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        # Plain Python, not PyTorch-specific: run nn.Module's own setup
        # first, before adding this model's own layer below. Always the
        # first line of __init__ for any nn.Module subclass.
        super().__init__()
        # A normal embedding table maps token id -> a learned vector of some
        # OTHER size (see roadmap). Here we deliberately make the output
        # size ALSO vocab_size, so row i of this table isn't just "a
        # representation of character i" — it directly IS the logits (raw
        # scores) for "what comes after character i." There's no separate
        # output layer bolted on afterward: this one lookup table is the
        # entire model, start to finish.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) integers — a batch of B character-index sequences,
        # each T characters long (T <= block_size). Look each one up in the
        # table to get its row of logits.
        logits = self.token_embedding_table(idx)  # shape (B, T, vocab_size)

        if targets is None:
            # No targets means we're just generating text (see generate()
            # below), not training — skip the loss computation entirely.
            loss = None
        else:
            # We have B*T independent next-character predictions in this
            # batch (every position in every sequence is its own flashcard —
            # see lesson 1). PyTorch's cross_entropy wants one row of logits
            # per row of targets, so flatten the (B, T, C) tensor down to
            # (B*T, C) — same numbers, just relabeled as one long list of
            # predictions instead of a (batch, position) grid of them.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # cross_entropy does two things internally: softmax the logits
            # into real probabilities, then score how much probability mass
            # landed on the actually-correct character (averaged over all
            # B*T predictions). Lower = the model is more confident about
            # the right answer, on average. This IS the "loss" number
            # you'll watch drop throughout training.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Actually write new text: repeatedly ask the model for a next-
        character distribution, sample one character from it, append, and
        feed the (now one-longer) sequence back in. This is the literal
        "ask the model, sample, repeat" loop described in the roadmap."""
        # idx: (B, T) tensor of character indices — the "prompt" to continue.
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, C) — recomputes the WHOLE sequence
            logits = logits[:, -1, :]  # every step, but only the LAST position's
            # dim=-1 = "operate across the last dimension" — here, turn each
            # row's 65 raw logits into 65 probabilities that sum to 1.
            probs = F.softmax(logits, dim=-1)  # logits get used — a bigram
            # model has no way to use earlier context even if it were given
            # it, so recomputing the full sequence here is pure waste. This
            # exact wastefulness is what KV-caching (a real inference-
            # serving trick, not covered in this repo) is designed to fix —
            # it only starts to matter once a model's earlier context is
            # actually worth reusing.
            idx_next = torch.multinomial(probs, num_samples=1)  # sample ONE
            # character per batch row, weighted by probs — NOT torch.argmax
            # (always pick the single most likely character), because argmax
            # would make the model output the exact same "most common
            # letter" forever after the first repeat. Sampling keeps output
            # varied, the way real text generation works.
            idx = torch.cat((idx, idx_next), dim=1)  # append the new character
        return idx


if __name__ == "__main__":
    model = BigramLanguageModel(vocab_size).to(device)
    # AdamW: the optimizer — the piece of code that actually nudges every
    # parameter based on the gradients loss.backward() computes. You won't
    # need to implement or tune the math inside it in this repo; treat it as
    # "the standard, off-the-shelf way to turn gradients into weight
    # updates."
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"device: {device}")
    # Every learnable number in the model, counted up. For this model that's
    # exactly vocab_size * vocab_size = 65*65 = 4225 — the entire lookup
    # table, nothing else.
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    # Start generation from a single "empty" character (index 0) and see what
    # a completely untrained (random-weights) model babbles — pure noise,
    # since nothing has been learned yet.
    # dtype=torch.long: store these as whole integers, since they're
    # character indices (into the vocab), not measurements. The single 0
    # is the "start of sequence" character to generate from.
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- before training ---")
    # [0] drops the batch dimension (only 1 sequence here); .tolist() turns
    # the tensor of character-indices back into a plain Python list, which
    # decode() (a plain dict lookup) can iterate over.
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

    # --- the training loop (see roadmap glossary for the 4-step breakdown) ---
    for it in range(max_iters):
        if it % eval_interval == 0:
            # Pause training briefly to measure loss cleanly (averaged over
            # many batches — see estimate_loss in common.py) on both splits.
            losses = estimate_loss(model, block_size, batch_size, device)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", block_size, batch_size, device)  # step 1: grab a batch
        _, loss = model(xb, yb)  # step 2: forward pass -> logits + loss
        optimizer.zero_grad(set_to_none=True)  # clear old gradients before computing new ones
        loss.backward()  # step 3: PyTorch computes every parameter's gradient
        optimizer.step()  # step 4: nudge every parameter to reduce the loss

    print("\n--- after training ---")
    # Same starting point as "before training" above — compare the two
    # outputs directly. This one should look vaguely word-shaped (correct
    # letter frequencies, plausible short letter-pairs) even though it's
    # still meaningless — that's the ceiling of "1 character of memory."
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))
