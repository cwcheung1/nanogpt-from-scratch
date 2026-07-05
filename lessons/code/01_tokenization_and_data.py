"""Lesson 01: tokenization & data prep.

Character-level tokenizer over tiny-shakespeare, train/val split, and the
get_batch() function every later lesson reuses. The actual logic lives in
common.py; this script demonstrates it.
"""
from common import chars, vocab_size, encode, decode, train_data, val_data, get_batch

block_size = 8  # how many characters of context the model sees at once
batch_size = 4  # how many independent sequences we train on in parallel

if __name__ == "__main__":
    print(f"vocab size: {vocab_size}")
    print(f"vocab: {''.join(chars)!r}")

    sample = "First Citizen"
    encoded = encode(sample)
    print(f"\nencode({sample!r}) -> {encoded}")
    print(f"decode(...) -> {decode(encoded)!r}")
    assert decode(encoded) == sample, "round trip broke"

    print(f"\ntrain chars: {len(train_data)}, val chars: {len(val_data)}")

    xb, yb = get_batch("train", block_size, batch_size)
    print(f"\nbatch shapes: x={tuple(xb.shape)}, y={tuple(yb.shape)}")
    print("--- what the model actually sees, one training example unrolled ---")
    for t in range(block_size):
        context = xb[0, : t + 1]
        target = yb[0, t]
        print(f"  context {decode(context.tolist())!r:12} -> target {decode([target.item()])!r}")
