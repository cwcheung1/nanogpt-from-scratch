"""Shared data-prep code, introduced and explained in lesson 01.

Every later lesson imports from here instead of re-deriving the tokenizer.
See lessons/01-tokenization-and-data.md for the full plain-language writeup —
these comments are the quick-reference version.
"""
import torch

# Fix PyTorch's random number generator to a specific starting point ("seed").
# Anything in this codebase that asks for "random" numbers — which random
# 8-character windows get(batch) picks, how model weights get initialized in
# later lessons — will now produce the EXACT SAME sequence of "random" values
# every time you run the script. Without this, two runs of the same code
# would get different random batches and produce different loss numbers,
# making it impossible to tell whether a change you made actually helped or
# you just got a luckier/unluckier shuffle. This is purely for
# reproducibility; it has nothing to do with how the model learns.
torch.manual_seed(1337)

# Read the entire training corpus into one big Python string in memory.
# tiny-shakespeare is ~1MB — small enough to just hold as a string, no
# streaming/chunking needed.
with open("data/input.txt", "r") as f:
    text = f.read()

# --- Build the vocabulary ---------------------------------------------------
# The vocabulary is the fixed, complete list of symbols the model is allowed
# to use. Here, one "symbol" = one character (this is a CHARACTER-level
# tokenizer — see the lesson doc for why, and what the alternative, BPE, is).
#
#   set(text)   -> every character that appears ANYWHERE in the whole 1MB
#                  file, with duplicates thrown away (a Python set holds no
#                  duplicates, and has no guaranteed order).
#   sorted(...) -> put those unique characters into one fixed, repeatable
#                  order. This matters because we're about to assign each
#                  character a number equal to its position in this list —
#                  if the order could change between runs, "the number for
#                  the letter A" would change too, and nothing would be
#                  consistent.
#   list(...)   -> sorted() already returns a list; this call is redundant
#                  but harmless (kept for clarity/historical reasons).
chars = sorted(list(set(text)))

# How many unique characters exist in the corpus. For tiny-shakespeare this
# is 65 — uppercase + lowercase letters, digits, space, newline, and a
# handful of punctuation marks. This number is the model's entire "alphabet
# size": every single thing the model ever predicts is "which one of these
# 65 symbols comes next", full stop.
vocab_size = len(chars)

# Two lookup tables, one for each direction of translation. `stoi`/`itos`
# aren't a standard industry acronym — they're just short variable names
# (this naming comes from Andrej Karpathy's original nanoGPT lectures, which
# this repo is modeled on) that literally spell out what they do once you
# expand them:
#   stoi = "STring TO Int":  character -> the integer that represents it
#   itos = "Int TO String":  integer -> the character it represents
# Example: if 'a' happens to be at position 57 in `chars`, then
# stoi['a'] == 57 and itos[57] == 'a'.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """Text -> numbers. One integer per character, in order, via `stoi`.
    e.g. encode("hi") -> [stoi['h'], stoi['i']], a list of exactly 2 ints.
    A neural net can only do math on numbers, so this is the mandatory first
    step before any text can be fed to a model."""
    return [stoi[c] for c in s]


def decode(ids):
    """Numbers -> text. The exact inverse of encode(): look each integer up
    in `itos` and glue the resulting characters back into one string.
    This is how we turn a model's numeric predictions back into something a
    human can read."""
    return "".join(itos[i] for i in ids)


# Encode the ENTIRE corpus (all ~1 million characters) into one long 1D
# tensor of integers. A torch.tensor is, for our purposes, "a list of numbers
# that PyTorch can do fast, vectorized math on" — think of it as a regular
# Python list that's been handed to a math library that knows how to batch
# operations efficiently (and, later, run them on a GPU). dtype=torch.long
# means "store these as whole integers", which makes sense since they're
# indices into the vocabulary, not measurements or fractions.
data = torch.tensor(encode(text), dtype=torch.long)

# --- Train/validation split --------------------------------------------------
# We hold back the last 10% of the corpus and never train the model on it.
# Why bother, if we already have the "answers" (the text itself)? Because the
# actual goal isn't to memorize Shakespeare — it's to learn the general
# statistical patterns of English/Shakespearean text well enough to predict
# characters in text the model has NEVER SEEN. `train_data` is what the model
# studies from (its weights get updated based on errors made here).
# `val_data` is a quiz the model never studies from — after training, we
# check the model's loss on val_data to see whether it actually learned
# general patterns, or just memorized the training text word-for-word
# (memorizing would show up as GREAT performance on train_data but BAD
# performance on val_data — a gap called "overfitting").
#
# We take a straight 90/10 CUT of the text rather than shuffling first,
# because this is one continuous document, not a pile of independent,
# shuffleable examples. Shuffling individual characters would destroy the
# very sequential structure we're trying to model, and would let
# training-time context "peek" at what's supposed to be held-out text.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, block_size, batch_size, device="cpu"):
    """Build one batch of labeled flashcards — NOT a prediction step. No
    model exists in this file; this just cuts (history, correct-next-
    character) pairs out of the real text, in the shape a future model
    (lesson 2+) will train on. `batch_size` independent pairs, each
    `block_size` characters long, cut from random locations in the text.

    Concretely, with block_size=8 and batch_size=4, this returns two tensors
    each shaped (4, 8):
      - x: 4 separate 8-character windows (the "history"/context side of
        the flashcard)
      - y: the SAME 4 windows, shifted one character to the right (the
        correct-answer side — what actually comes next in the real text) —
        see the lesson doc for why one shifted copy supplies 8 separate
        flashcards, not 1.

    "block_size" = context length: how many characters of history each
    flashcard's front shows. This becomes a hard architectural limit once a
    model is built on this data in lesson 2+ — it will be physically unable
    to see further back than this, because it's simply never shown more.

    "batch_size" = how many of these flashcard-windows we cut out as one
    unit. The 4 windows are UNRELATED to each other (4 random, independent
    snippets of the corpus) — we only group them because, once a model
    exists (lesson 2+), a GPU/CPU can multiply a batch of 4 things through
    it in about the same time as 1, via vectorized math, instead of looping
    over them one at a time. That's what "parallel" means here: parallel on
    the HARDWARE, not "these examples relate to each other." Changing
    batch_size affects training speed/stability, not what the model is
    capable of learning.
    """
    d = train_data if split == "train" else val_data

    # Pick `batch_size` random starting indices into the data. The upper
    # bound is `len(d) - block_size` so that every `i + block_size` slice
    # stays inside the array (never reads past the end).
    ix = torch.randint(len(d) - block_size, (batch_size,))

    # For each random start i: x gets the block_size characters starting at
    # i; y gets the block_size characters starting one position LATER
    # (i + 1) — i.e. y is x shifted right by one. torch.stack takes the list
    # of `batch_size` 1D slices and glues them into a single 2D tensor of
    # shape (batch_size, block_size).
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()  # we're only reading the model's loss here, not training —
# skip building the machinery loss.backward() would need, which saves memory
# and time.
def estimate_loss(model, block_size, batch_size, device="cpu", eval_iters=200):
    """A single batch's loss bounces around too much (it's just 4-64 random
    examples) to tell whether training is actually progressing. This grabs
    `eval_iters` (default 200) fresh random batches for EACH split and
    averages their losses — a much more stable read on "how good is the
    model right now," for both train and val, side by side. This is the
    function behind every "step N: train loss ..., val loss ..." line
    you'll see printed in lessons 2 and up.

    model.eval() / model.train(): some layers (Dropout, introduced in
    lesson 6) behave differently during training vs. evaluation — eval()
    switches them to their "just report your best answer" mode, train()
    switches back before returning so the actual training loop isn't
    affected by this brief measurement detour."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
