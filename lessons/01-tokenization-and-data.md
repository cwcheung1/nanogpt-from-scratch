# 01 — Tokenization & Data Prep

**Concept**: before any neural net can touch text, you need a lossless
mapping from characters to integers, and a way to carve a long document into
many small (context, next-character) training examples.

**Analogy**: the tokenizer is a phrasebook with exactly one entry per
character — flip to the entry, get a number; flip by number, get the
character back. It's dumb on purpose: no subword merging, no vocabulary
learning, just a lookup table. That dumbness is the point for lesson 1 — it
isolates "how does text become tensors" from "how do you build a smart
tokenizer" (BPE comes later, in a different repo/lesson).

**How it works** (`lessons/code/01_tokenization_and_data.py`):
- `chars = sorted(set(text))` — the vocabulary is just every unique
  character that appears in tiny-shakespeare (65 of them: letters,
  punctuation, newline, space).
- `stoi`/`itos` are the two directions of the lookup table. `encode`/`decode`
  are one-liners built on top of them.
- The whole dataset becomes one long 1D tensor of integers. First 90% is
  `train_data`, last 10% is `val_data` — a simple contiguous split, not
  shuffled (this is a corpus, not i.i.d. samples).
- `get_batch(split)` is the function every later lesson reuses unchanged. It
  picks `batch_size` random starting offsets, and for each one slices out
  `block_size` characters as input (`x`) and the *same slice shifted one
  character to the right* as the target (`y`). That shift is the whole
  supervision signal for a language model: predict the next character.
- The unrolled printout at the bottom is the important part to stare at: one
  8-character window actually produces 8 separate training examples (predict
  char 2 from char 1, predict char 3 from chars 1-2, ...). This is why a
  transformer trained on one batch of `(4, 8)` tensors is actually learning
  from 32 next-token predictions, not 4.

**Look at**: `lessons/code/01_tokenization_and_data.py`, `data/input.txt`

**Run this**:
```
make lesson1
```
You should see vocab size 65, a clean encode/decode round trip on
`"First Citizen"`, and the unrolled context->target table at the bottom.

**You'll know it clicked when**: you can explain why `y = x` shifted by one
character is sufficient to train next-token prediction, without needing
separate labels — and why a single row of `block_size=8` gives you 8 training
signals, not 1.
