"""Lesson 01: tokenization & data prep.

NO MODEL EXISTS IN THIS FILE. This script only turns text into numbers and
cuts labeled (history, correct-next-character) flashcards out of it, in the
shape a future model (lesson 2+) will train on — nothing here predicts
anything. All the actual logic lives in common.py (read that file's comments
first); this script just calls into it and prints what's happening at each
step so you can see it work. Full plain-language writeup:
lessons/01-tokenization-and-data.md.
"""
from common import chars, vocab_size, encode, decode, train_data, val_data, get_batch

# "Context length": how many characters of history each flashcard's front
# shows. This becomes a hard limit once a model is built on this data in
# lesson 2+ — it will be physically unable to look further back than this.
# See the get_batch() docstring in common.py for the full explanation.
block_size = 8

# How many independent, unrelated flashcard-windows we cut out and bundle
# into one tensor at once. "Independent" is the key word — these are NOT 4
# parts of one story, they're 4 random unrelated snippets. See the
# get_batch() docstring in common.py for why this is called "parallel"
# (parallel on the hardware, once a model exists to process them — not
# "related to each other").
batch_size = 4

if __name__ == "__main__":
    print(f"vocab size: {vocab_size}")
    print(f"vocab: {''.join(chars)!r}")  # every unique character, in order

    # Prove encode/decode are true inverses: turning "First Citizen" into
    # numbers and then back into text must reproduce the exact original
    # string. If this assert ever fails, the tokenizer itself is broken.
    sample = "First Citizen"
    encoded = encode(sample)
    print(f"\nencode({sample!r}) -> {encoded}")
    # Notice: len(encoded) == len(sample) == 13. One number per character,
    # always — including the space, which is why you'll see the same integer
    # (whatever stoi[' '] is) repeat wherever a space appears.
    print(f"decode(...) -> {decode(encoded)!r}")
    assert decode(encoded) == sample, "round trip broke"

    print(f"\ntrain chars: {len(train_data)}, val chars: {len(val_data)}")
    # train_data: what a future model will study from (lesson 2+ updates its
    # weights based on errors made here). val_data: a quiz that model will
    # never study from, used only to check whether it learned general
    # patterns or just memorized the training text. See
    # lessons/01-tokenization-and-data.md for more on why this split exists.

    # Ask common.py for one batch: 4 independent 8-character flashcard-
    # windows (x, the history side) and their "shifted right by one
    # character" answers (y, the correct-next-character side).
    xb, yb = get_batch("train", block_size, batch_size)
    print(f"\nbatch shapes: x={tuple(xb.shape)}, y={tuple(yb.shape)}")
    # Shape (4, 8) reads as (batch_size, block_size): "4 independent
    # flashcard-windows, each 8 characters long."

    print("--- one flashcard-window unrolled into its 8 separate (history -> answer) pairs ---")
    # Look at just the FIRST window in the batch (index 0) and walk through
    # it one character at a time. This reveals that a single block_size=8
    # window is secretly 8 separate flashcards, not 1: at step t, the
    # "history" shown is xb[0, 0:t+1] and the recorded correct answer is
    # yb[0, t] — the character that actually comes next in the real text.
    # At t=0 the history is just 1 character; by t=7 it's the full 8-character
    # window. (Once a model exists in lesson 2+, "history" becomes what it's
    # shown and "answer" becomes what it's scored against.)
    for t in range(block_size):
        context = xb[0, : t + 1]
        target = yb[0, t]
        print(f"  context {decode(context.tolist())!r:12} -> target {decode([target.item()])!r}")
