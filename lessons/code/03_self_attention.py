"""Lesson 03: self-attention mechanics. NO TRAINED MODEL HERE — everything
runs on random input (torch.randn), on purpose. We're only building and
inspecting the mechanism itself; training details (loss, optimizer, batches
of real text) would just be noise on top of "what does this math actually
compute," so this lesson strips them away entirely.

Part A: "the trick" — averaging past positions efficiently via a masked
matrix-multiply, built up in three equivalent forms so you can see that real
attention is just a generalization of a plain average.

Part B: a real single self-attention head (query/key/value, scaled dot
product, causal mask) applied to that same random input, so we can inspect
the actual attention weight matrix directly.

Term definitions (tensor, embedding) live in lessons/00-roadmap.md.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# B = batch size (independent sequences), T = "time"/sequence position (how
# many characters long each sequence is), C = channels (the width of each
# position's embedding vector — how many numbers represent one character).
# These are just descriptive names for a random tensor's shape in this
# lesson; nothing is loaded from real data here.
B, T, C = 4, 8, 32


def trick_v1_loop(x):
    """The GOAL, stated as plainly as possible: for every position t, average
    together the vectors of every position from 0 up to and including t.
    Written as an actual double for-loop so there's no cleverness to parse —
    just "for each thing, average everything before it." Correct, but slow
    (a real Python loop, not vectorized math) — the next two functions
    compute this exact same result faster."""
    out = torch.zeros((B, T, C))
    for b in range(B):  # for each independent sequence in the batch...
        for t in range(T):  # ...and each position within it...
            out[b, t] = x[b, : t + 1].mean(0)  # ...average positions 0..t
    return out


def trick_v2_matmul(x):
    """Same result as trick_v1, but as ONE matrix multiply instead of a
    Python loop — the standard trick for turning slow loops into fast
    vectorized math. `torch.tril(torch.ones(T, T))` builds the
    lower-triangular matrix of 1s described in the lesson doc (row i has 1s
    in columns 0..i, 0s after). Dividing each row by its own sum turns those
    1s into 1/(i+1) each — i.e. "average of i+1 things." Multiplying that
    matrix by x then computes, for every row, a weighted sum of all of x's
    rows using those weights — which for THIS weight matrix means "the
    average of everything up to and including position i." Same answer as
    the loop, computed as pure matrix math."""
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    return wei @ x  # (T, T) @ (B, T, C) broadcasts to (B, T, C)


def trick_v3_softmax(x):
    """Same result a THIRD time, via a detour through softmax — this detour
    looks unnecessary here (more steps for the same answer), but it's the
    form that generalizes to real attention, which is the entire point of
    building it this way.
    Instead of starting from a matrix of 1s (hard-coded "equal weight to
    everyone before me"), start from a matrix of all 0s — meaning "no
    preference yet, every position equally uninteresting." masked_fill(...,
    -inf) then overwrites every position that's NOT allowed (everything
    strictly after position i) with negative infinity. Softmax turns -inf
    into a probability of exactly 0 (e^-inf = 0) — so those forbidden future
    positions get precisely zero weight, no matter what. Softmax then turns
    the remaining (all still-0, i.e. equal) scores into equal probabilities
    — which is why this reproduces the same plain average as v1/v2.
    The payoff, used in the real Head class below: replace those all-zero
    scores with LEARNED, DATA-DEPENDENT scores, and this exact same
    masked-softmax machinery turns them into a valid, causally-masked
    weighted average instead of a hard-coded equal one."""
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    return wei @ x


class Head(nn.Module):
    """One real self-attention head — same masked-softmax shape as
    trick_v3, but with LEARNED scores instead of all-zeros.

    Every position produces 3 vectors, each via its own learned nn.Linear
    layer applied to that position's embedding:
      - query ("what am I looking for?")
      - key   ("what do I contain, that others might want?")
      - value ("what do I actually hand over, if someone attends to me?")
    "Affinity" (relevance) between position i and position j = i's query
    dot-producted with j's key (a dot product measures how aligned two
    vectors are — bigger = more aligned = more relevant). The final output
    at position i is a weighted average of every ALLOWED position's value
    vector, weighted by those affinities (after masking + softmax)."""

    def __init__(self, head_size, n_embd=C, block_size=T):
        super().__init__()
        # bias=False matches nanoGPT/GPT-2's convention for these 3
        # projections — one less set of parameters, negligible difference
        # at this scale, kept for fidelity to the real architecture.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer: store this tensor as part of the model (so it
        # moves to the GPU along with everything else via .to(device)) but
        # NOT as a learnable parameter — this mask is fixed, never trained.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size) — every position's "what I contain"
        q = self.query(x)  # (B, T, head_size) — every position's "what I want"
        head_size = k.shape[-1]

        # Every query dotted with every key, all at once: (B,T,hs) @ (B,hs,T)
        # -> (B, T, T). Entry [b, i, j] = how much position i's query aligns
        # with position j's key, for batch item b.
        # The `* head_size**-0.5` scaling: as head_size grows, dot products
        # of random vectors tend to grow larger in magnitude too (more terms
        # summed together). Very large/small values pushed through softmax
        # saturate — nearly all the probability collapses onto just the
        # single largest score, making the "weighted average" barely
        # different from "just pick the top one." Dividing by
        # sqrt(head_size) keeps the scores in a range where softmax
        # produces a genuinely graded distribution instead of a collapsed
        # one. This is the literal "scaled" in "scaled dot-product
        # attention."
        wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, T)
        # Same causal mask as trick_v3: zero out (via -inf -> softmax) any
        # position j > i, so position i can never attend to the future.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # each row -> a real probability distribution

        v = self.value(x)  # (B, T, head_size) — every position's "what I hand over"
        # Weighted average of VALUES (not keys!) using the attention
        # weights — keys were only ever used for matching/scoring, values
        # are the actual content that flows through to the output.
        return wei @ v, wei  # (B, T, head_size) output, plus wei for inspection


if __name__ == "__main__":
    # Random "fake embeddings" — 4 sequences, 8 positions each, 32 numbers
    # per position. Nothing here comes from real text; the whole point of
    # this lesson is inspecting the MATH, not training on data.
    x = torch.randn(B, T, C)

    # Prove all 3 "tricks" compute the exact same numbers, just via
    # different amounts of work — the loop, the plain matmul, and the
    # masked-softmax detour should be indistinguishable up to floating-point
    # rounding (atol = allowed tolerance).
    out1, out2, out3 = trick_v1_loop(x), trick_v2_matmul(x), trick_v3_softmax(x)
    print("v1 (loop) vs v2 (matmul) match:", torch.allclose(out1, out2, atol=1e-6))
    print("v2 (matmul) vs v3 (softmax) match:", torch.allclose(out2, out3, atol=1e-6))

    head_size = 16  # width of the query/key/value vectors this head produces
    head = Head(head_size)
    out, wei = head(x)
    print(f"\nHead output shape: {tuple(out.shape)}  (expect ({B}, {T}, {head_size}))")

    print("\nattention weights for batch element 0 (rows=query pos, cols=key pos):")
    torch.set_printoptions(precision=2, sci_mode=False)
    print(wei[0])
    # Read this printed grid as: row i is "how much does position i listen
    # to each position j (columns), as a probability distribution." Row 0
    # must be all weight on column 0 (nothing else exists yet); row 7 can
    # spread weight across columns 0-7, however the learned query/key
    # vectors decided (untrained here, so it's whatever random init produced
    # — no claim that these particular weights are "good," just that they're
    # VALID: causal and summing to 1).

    # Sanity-check the two properties that make this a valid causal
    # attention matrix, rather than trusting the printed numbers by eye:
    is_causal = torch.allclose(torch.triu(wei[0], diagonal=1), torch.zeros(T, T))
    row_sums = wei[0].sum(dim=-1)
    print(f"\nupper triangle is all zero (causal): {is_causal}")
    print(f"each row sums to 1: {torch.allclose(row_sums, torch.ones(T))}")
