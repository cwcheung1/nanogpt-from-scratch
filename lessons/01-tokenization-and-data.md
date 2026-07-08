# 01 — Tokenization & Data Prep

*New to this project? Read [00 — Roadmap](00-roadmap.md) first — it explains
what we're building overall and why this lesson comes first.*

**Jargon buster — new terms this lesson's code uses** (full definitions in
the roadmap's [PyTorch/Python idioms](00-roadmap.md#pytorchpython-idioms-the-code-level-words-not-the-ml-concept-words)
section):

- **tensor** — PyTorch's basic data container, a multi-dimensional array of
  numbers (a Python list, but with fast/GPU-capable math built in).

- **`dtype=torch.long`** — store these numbers as whole integers, since
  they're character indices, not measurements.

- **`torch.manual_seed(1337)`** — makes this script's "random" batch
  sampling reproducible run-to-run.

## The problem this lesson solves

A neural network is a pile of math — multiplication, addition, that's it. It
cannot "read" the letter `A`. Before we can train anything, we need to turn
text into numbers, and we need a way to carve one giant block of text into
many small labeled flashcards ("here's some text, and here's the character
that actually comes next in Shakespeare"). That's the entire job of this
lesson — **no neural net exists yet, and nothing is predicting anything.**
We're only building the practice materials that lesson 2's model will later
train on.

## Part 1: turning characters into numbers

Open `data/input.txt` — it's ~1MB of Shakespeare's plays, as one long plain
text file. Here's what `lessons/code/common.py` does to it, step by step:

1. **Find every unique character that appears anywhere in the file.** Not
   words — single characters. Letters (`a`-`z`, `A`-`Z`), digits, spaces,
   punctuation, newlines. For tiny-shakespeare there happen to be exactly
   **65** distinct characters. This list of 65 is called the **vocabulary**
   — it's the entire "alphabet" the model will ever know about.

2. **Give each character in that vocabulary a fixed number**, based on its
   position in a sorted list. So maybe `'\n'` (newline) is `0`, `' '` (space)
   is `1`, `'A'` is `20`, and so on — the exact numbers don't matter, what
   matters is that the mapping is *fixed*: `'A'` always means the same
   number, every time.

3. **Yes — every single character gets its own number, one-to-one, no
   exceptions.** This was verified by actually running the code:
   ```
   encode('First Citizen') -> [18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52]
   ```
   `"First Citizen"` is 13 characters (`F`,`i`,`r`,`s`,`t`,` `,`C`,`i`,`t`,`i`,`z`,`e`,`n`)
   and the output is a list of exactly 13 numbers. Notice the space character
   became `1`, and it shows up again as the 6th number — the same character
   always encodes to the same number. That list of 13 numbers is the *only*
   thing the model ever sees; it never sees the letters themselves.

This whole char↔number lookup table is called a **tokenizer**. Ours is the
dumbest possible kind — one "token" per character, no cleverness.

In the code, the two lookup tables are named `stoi` and `itos`. These are
**not** industry-standard acronyms — they're just abbreviated variable names
(borrowed from Andrej Karpathy's original nanoGPT lecture, which this repo
follows) that spell out what they do once expanded:

- `stoi` = **s**tring **to** **i**nt → given a character, look up its number
- `itos` = **i**nt **to** **s**tring → given a number, look up its character

`encode()` is built on `stoi` (character→number direction), `decode()` is
built on `itos` (number→character direction, to turn the model's numeric
output back into readable text).

### What is BPE, and why doesn't lesson 1 use it?

BPE = **Byte Pair Encoding**. It's the tokenization scheme real production
models (GPT-2/3/4, etc.) actually use, and it's a genuinely different idea
from what we're doing here — worth naming so you know what you're *not*
looking at yet.

Instead of one token per character, BPE starts from individual characters
and repeatedly **merges the most frequently-occurring adjacent pair into a
single new token**, thousands of times over, on a huge training corpus. The
end result is a vocabulary of ~50,000 tokens, most of which are common
whole words (`"the"`, `"ing"`) or word-fragments, with rare words falling
back to smaller pieces or individual characters. This makes sequences
shorter (one token can be a whole word instead of 5+ characters) which is a
big efficiency win at scale — but the *merging algorithm* and the *resulting
vocabulary* are both extra machinery with their own training process.

Lesson 1 deliberately skips all of that. A character-level tokenizer needs
zero training (the vocabulary is just "whatever unique characters exist in
the file"), so this lesson can isolate one question — *how does text become
numbers a model can use* — from a completely separate question — *how do
you build a smart, efficient vocabulary*. BPE would be the right topic for a
follow-up lesson, not this one; every concept in this lesson (encode/decode,
`get_batch`, `block_size`, `batch_size`) works identically regardless of
which tokenizer produced the numbers.

### What are train/validation data for?

After `text` is fully encoded into one long tensor of integers, the code
splits it: the first 90% becomes `train_data`, the last 10% becomes
`val_data`, and it never touches `val_data` while actually updating the
model's weights.

Why bother holding data back, when we already have "the answers" (the text
itself)? Because the goal was never to memorize Shakespeare word-for-word —
it's to learn general patterns of the language well enough to predict
characters in text the model has **never seen during training**.

- `train_data` — what the model studies from. Every weight update comes from
  errors made on batches pulled from here.

- `val_data` — a quiz the model never studies from. After training (or
  periodically during it), we compute the model's loss on `val_data` only to
  *check* whether it generalized, without ever letting that data influence
  the weights.

If the model does great on `train_data` but badly on `val_data`, that gap
means it memorized the training text instead of learning general patterns —
a failure mode called **overfitting**. Comparing the two losses is the main
way you'll know, in later lessons, whether training is actually working.

The split is a straight 90/10 **cut** of the text, not a shuffle, because
this is one continuous document rather than a pile of independent, freely
reorderable examples — shuffling individual characters first would destroy
the sequential structure we're trying to model, and would let training data
"leak" context from what's supposed to be held out.

## Part 2: manufacturing practice problems (no model exists yet!)

**Rule for reading this whole section: any sentence that talks about
"guessing" or "predicting" is describing lesson 2, in the future tense —
never this lesson.** There is no neural net anywhere in `common.py` or
`01_tokenization_and_data.py`. Everything below is just slicing a known
array of numbers two different ways. Keep that separate from what lesson 2
will later do with the result, and the rest of this section should be
mechanical, not conceptual.

### The concrete mechanism, first — no analogy yet

The real text at one point in the corpus is these 9 characters, already
encoded as 9 numbers by the tokenizer from Part 1 (shown here as letters for
readability):

```
index:  0  1  2  3  4  5  6  7  8
char:   L  e  t  '  s  _  h  e  a      (_ = space)
```

`block_size = 8` means we cut out an 8-long slice starting at some index —
say index 0 — as `x`. Then `y` is **the exact same underlying array, sliced
starting one index later**:

```
x = data[0:8]   ->   L  e  t  '  s  _  h  e
y = data[1:9]   ->   e  t  '  s  _  h  e  a
```

Line them up and `y[t]` is always "whatever real character sits one
position after `x`'s first `t+1` characters" — because `y` was read out of
the *same source text*, just starting one character later. Nothing was
computed, transformed, or guessed to produce `y` — it's a second, offset
read of data we already have in full. That's the entirety of "shifted by
one": two overlapping slices of one known array.

Reading `x` and `y` this way, column by column, gives you 8 separate
(context, true-next-character) pairs for free — `x[0:1]` paired with
`y[0]`, `x[0:2]` paired with `y[1]`, and so on up to the full window paired
with `y[7]`. Printed out (this is real output from running the script):

```
context 'L'          -> target 'e'
context 'Le'         -> target 't'
context 'Let'        -> target "'"
context "Let'"       -> target 's'
context "Let's"      -> target ' '
context "Let's "     -> target 'h'
context "Let's h"    -> target 'e'
context "Let's he"   -> target 'a'
```

That's why one `block_size=8` window yields 8 training signals instead of
1: each row above is just a different-length prefix of `x`, paired with the
single already-known character that follows it in `y`.

### If an analogy helps: flashcards written before any student arrives

Same fact as above, restated: a `(x, y)` pair is like a flashcard where the
front shows some text and the back shows the character that *actually*
comes next in the real play — written by someone who already has the
answer key (`data/input.txt`), not by someone predicting. "Predict" only
becomes the accurate word once lesson 2's model is shown only the front and
has to guess the back, then gets scored against what's already written
there.

### `block_size` and `batch_size`, named

`block_size = 8` is the width of the slice above — how much history `x`
carries. It isn't just a data-prep choice: in lesson 2+, the model trained
on this data becomes *architecturally incapable* of looking further back
than `block_size` characters, because it's simply never shown more. Later
lessons call this the **context length**. Bigger `block_size` = more
history per example, more compute per example once a model exists.

`batch_size = 4` means we don't cut just one `(x, y)` slice pair like the
one above — we cut 4 of them at once, from 4 random, unrelated starting
indices in the corpus (one might land in Act 1, another in Act 3). They
have nothing to do with each other; grouping them is purely so a GPU can
push all 4 through a future model in about the same time as 1, via
vectorized math. "Parallel" here means parallel *on the hardware*, not
"related in content."

Put together: `get_batch()` returns two tensors of shape `(batch_size,
block_size)` = `(4, 8)` — read that as **"4 independent flashcard-windows,
each 8 characters long."** `x` is the fronts of the cards (the history), `y`
is the same windows shifted right by one character (the correct answers
written on the backs).

## Look at

`lessons/code/01_tokenization_and_data.py` and `lessons/code/common.py` — both
are now heavily commented, line by line, explaining what each piece does and
why. Read the comments in `common.py` first; that's where `get_batch` and the
tokenizer actually live.

## Run this

```
make lesson1
```

You should see: vocab size 65, `encode`/`decode` round-tripping `"First
Citizen"` back to itself unchanged, and the unrolled context→target table
(8 rows) at the bottom.

## You'll know it clicked when

You can answer, in your own words, without looking anything up:

- Why does `encode("hi")` return a list of exactly 2 numbers?
- Why is `y` just `x` shifted one character to the right — what does that
  shifted copy represent?

- If `block_size=8`, how many separate training signals come out of one
  window? (Answer: 8, not 1.)

- If `batch_size=4`, are those 4 examples related to each other in the text?
  (Answer: no — they're 4 random, independent locations. "Parallel" refers to
  the hardware processing them simultaneously, not to any relationship
  between the examples themselves.)
