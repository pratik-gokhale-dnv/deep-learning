from qdsl import tf, single, multi, num, T

S = "sizes"

BASE = ("Use the base Transformer of \"Attention Is All You Need\": d_{model} = 512, h = 8 heads, d_{k} = d_{v} = d_{model}/h = 64, "
        "d_{ff} = 2048, N = 6 layers. Ignore biases, LayerNorm and the batch dimension unless the question says otherwise.")

CHEAT = """Shape cheat-sheet for one head (n = source length, m = target length, d_k = d_v = 64):

```
                Q      K      V      scores   head output
encoder self    n×64   n×64   n×64   n×n      n×64
decoder self    m×64   m×64   m×64   m×m      m×64   (masked)
cross-attn      m×64   n×64   n×64   m×n      m×64
```

Concatenating the 8 heads gives (rows)×512; W^{O} is 512×512 and maps it back to (rows)×512."""

# ---------------------------------------------------------------- single (9)
single(S, BASE + "\n\nWhat is the shape of the query projection matrix W^{Q}_{i} of a single attention head, in the paper's notation (Q_{i} = X·W^{Q}_{i} with X ∈ ℝ^{n × d_model})?",
       ("512 × 64", "W^{Q}_{i} ∈ ℝ^{d_model × d_k}: it maps a 512-dimensional token vector to a 64-dimensional query."),
       [("512 × 512", "That would be a head that keeps the full model width. With h = 8 heads each head projects down to d_{k} = d_{model}/h = 64 (512 × 512 is the combined size of all 8 heads' W^{Q}_{i} together)."),
        ("64 × 512", "This is the transpose. The layer input X has shape n × 512, so X·W^{Q}_{i} requires W^{Q}_{i} to have 512 rows. (PyTorch's nn.Linear(512, 64).weight is stored transposed, as 64 × 512, because it computes X·Wᵀ.)"),
        ("64 × 64", "This ignores the input width: each projection must accept a d_{model} = 512 vector.")],
       """Per head, W^{Q}_{i}, W^{K}_{i} ∈ ℝ^{d_model × d_k} = 512 × 64 and W^{V}_{i} ∈ ℝ^{d_model × d_v} = 512 × 64.

For an input X ∈ ℝ^{n × 512}: Q_{i} = X·W^{Q}_{i} ∈ ℝ^{n × 64}. Stacking the 8 heads' matrices side by side gives one 512 × 512 matrix, which is why implementations often use a single 512×512 projection and then split it into heads.""",
       calc=True)

single(S, BASE + "\n\nA source sentence has n = 20 tokens. In the **encoder's** self-attention, what are the shapes of Q_{i}, K_{i} and V_{i} for one head?",
       ("Each is 20 × 64", "All three come from the same 20 × 512 input and are projected to d_{k} = d_{v} = 64 columns."),
       [("Each is 20 × 512", "20 × 512 is the shape of the layer input X (and of the output of the full multi-head layer), not of a single head's projections."),
        ("Each is 20 × 20", "20 × 20 is the shape of the score matrix Q_{i}K_{i}^{T}, not of the projections."),
        ("Q is 20 × 64, K is 64 × 20 and V is 20 × 64", "K itself is 20 × 64; it is only its transpose K^{T} (64 × 20) that appears inside the product Q_{i}K_{i}^{T}.")],
       """Every encoder position produces a query, a key and a value, so each matrix has one row per token (n = 20) and d_{k} = d_{v} = 64 columns.

`Q_i = X W^Q_i (20×512 · 512×64 = 20×64)`, and likewise K_{i} and V_{i}. The scores Q_{i}K_{i}^{T} are (20×64)·(64×20) = 20×20.""",
       calc=True)

single(S, BASE + "\n\nFor one encoder head with n = 12 source tokens, what are the shape of the attention-weight matrix softmax(Q_{i}K_{i}^{T}/√d_{k}) and its normalisation?",
       ("12 × 12; each row sums to 1", "Row j is the distribution of query j over all 12 keys; the softmax runs along the key axis."),
       [("12 × 12; each column sums to 1", "The softmax normalises over the keys for each query (along a row), not over the queries (down a column)."),
        ("12 × 64; each row sums to 1", "12 × 64 is the shape of the head's output after multiplying the weights by V_{i}, not of the weights."),
        ("12 × 12; all 144 entries together sum to 1", "Normalisation is done row by row, so the whole matrix sums to 12 (one per row), not 1.")],
       """The score matrix has one entry per (query, key) pair: n × n = 12 × 12. A row-wise softmax turns each row into a probability distribution (non-negative, sum 1), so the full matrix sums to n = 12.

Then the weights are multiplied by V_{i}: (12×12)·(12×64) = 12×64.""",
       calc=True)

single(S, BASE + "\n\nIn the encoder (n = 20 tokens), the outputs of the 8 heads are concatenated before the final projection W^{O}. What is the shape of the concatenated matrix?",
       ("20 × 512", "Eight heads × d_{v} = 64 columns each = 512 columns, one row per token."),
       [("20 × 64", "That is the output of a single head; concatenation joins the 8 heads side by side along the feature axis."),
        ("160 × 64", "Stacking the heads vertically (8 × 20 rows) would mix up positions with heads; concatenation is along the feature dimension."),
        ("20 × 4,096", "This would happen if every head had d_{v} = 512. Each head has d_{v} = d_{model}/h = 64, so 8 × 64 = 512.")],
       """`concat(head_1, …, head_8)` ∈ ℝ^{n × (h·d_v)} = ℝ^{20 × 512}.

Multiplying by W^{O} ∈ ℝ^{512 × 512} returns an n × d_{model} = 20 × 512 matrix, which can be added to the residual input X.""",
       calc=True)

single(S, BASE + "\n\nIn the decoder's **encoder–decoder (cross) attention**, the source has n = 15 tokens and the target prefix has m = 9 tokens. What are the shapes of Q_{i}, K_{i}, V_{i} for one head?",
       ("Q is 9 × 64; K and V are 15 × 64", "Queries come from the decoder (9 positions); keys and values come from the encoder output (15 positions)."),
       [("Q is 15 × 64; K and V are 9 × 64", "The roles are swapped: in cross-attention the decoder asks the questions (Q), the encoder supplies keys and values."),
        ("All three are 9 × 64", "That describes the decoder's self-attention, where Q, K and V all come from the target side."),
        ("All three are 15 × 64", "That would be encoder self-attention; cross-attention takes its queries from the decoder.")],
       "In cross-attention Q_{i} = Y·W^{Q}_{i} uses the decoder states Y (9×512), while K_{i} = E·W^{K}_{i} and V_{i} = E·W^{V}_{i} use the encoder output E (15×512). Hence Q: 9×64, K: 15×64, V: 15×64.",
       calc=True)

single(S, BASE + "\n\nIn the decoder's cross-attention the source has n = 15 tokens and the target prefix has m = 9 tokens. What is the shape of the attention-weight matrix of one head?",
       ("9 × 15", "One row per decoder position (query), one column per encoder position (key)."),
       [("15 × 9", "This is the transpose: rows must correspond to queries (decoder, 9), columns to keys (encoder, 15)."),
        ("9 × 9", "9 × 9 is the shape of the decoder's masked self-attention scores, which pair target positions with target positions."),
        ("15 × 15", "15 × 15 is the shape of the encoder's self-attention scores.")],
       "Scores = Q_{i}K_{i}^{T} = (9×64)·(64×15) = 9×15. Each of the 9 rows is a softmax over the 15 source positions, i.e. a soft alignment of one target word with the source sentence.",
       calc=True)

single(S, BASE + "\n\nIn the decoder's cross-attention the source has n = 15 tokens and the target prefix has m = 9 tokens. What is the shape of one head's output, i.e. softmax(QK^{T}/√d_{k})·V, before the heads are concatenated?",
       ("9 × 64", "(9×15)·(15×64) = 9×64: one output row per query, with the value dimension d_{v} = 64."),
       [("15 × 64", "That is the shape of V_{i}; the output has one row per query (decoder position), not per source position."),
        ("9 × 15", "That is the shape of the weight matrix; multiplying by V_{i} replaces the 15 columns by d_{v} = 64."),
        ("9 × 512", "9 × 512 is the shape after concatenating all 8 heads (8 × 64), not the output of a single head.")],
       "The weights (9×15) times V_{i} (15×64) give 9×64. Concatenating 8 heads gives 9×512, and W^{O} keeps it 9×512, so the result can be added to the decoder's residual stream.",
       calc=True)

single(S, BASE + "\n\nHow many trainable parameters (weights only, no biases) does one multi-head attention block have, counting all 8 heads and the output projection W^{O}?",
       ("1,048,576", "8 heads × 3 × (512 × 64) = 786,432 for Q, K, V plus 512 × 512 = 262,144 for W^{O}, i.e. 4 × 512^{2}."),
       [("786,432", "This counts only the Q, K, V projections (3 × 512^{2}) and forgets the output projection W^{O}."),
        ("131,072", "This is one head's share: 4 × 512 × 64. The block contains 8 heads."),
        ("8,388,608", "This gives every head full-width 512 × 512 matrices (8 × 4 × 512^{2}); heads actually use d_{k} = 512/8 = 64 columns.")],
       """- Per head: W^{Q}_{i}, W^{K}_{i}, W^{V}_{i}, each 512 × 64 = 32,768 → 98,304
- 8 heads: 786,432 (= 3 × 512²)
- Output projection W^{O}: (8·64) × 512 = 262,144
- Total: **1,048,576** = 4 × d_{model}^{2}

The count does not depend on the number of heads (as long as d_{k} = d_{model}/h): more heads means narrower heads, not more weights. It also does not depend on the sequence length.""",
       calc=True)

single(S, BASE + "\n\nA batch of 32 sentences, each padded to 10 tokens, goes through one encoder self-attention layer. With all heads computed in parallel as one batched matrix product, what is the shape of the full score tensor Q K^{T}/√d_{k} (batch, heads, rows, columns)?",
       ("[32, 8, 10, 10]", "One 10 × 10 score matrix per head and per sentence: 32 × 8 of them."),
       [("[32, 8, 10, 64]", "That is the shape of Q (and K, V) with the layout batch × heads × length × d_{k}, not of the scores."),
        ("[32, 10, 10]", "This has no head axis: each of the 8 heads produces its own score matrix."),
        ("[32, 8, 64, 64]", "A 64 × 64 matrix would come from K^{T}K (feature × feature); scores pair positions with positions, so the last two axes are 10 × 10.")],
       """With layout [batch, heads, length, d_k]:

- Q, K, V: [32, 8, 10, 64]
- Scores = Q·K^{T}: [32, 8, 10, 64] · [32, 8, 64, 10] = [32, 8, 10, 10]
- Weights · V: [32, 8, 10, 10] · [32, 8, 10, 64] = [32, 8, 10, 64]
- Merge heads: [32, 10, 512]

Memory for the scores grows with the **square** of the sequence length.""",
       calc=True)

# ---------------------------------------------------------------- numeric (7)
num(S, BASE + "\n\nA source sentence has 20 tokens. How many attention weights (softmax outputs) does the **whole encoder** compute for it, counting all 6 layers and all 8 heads?", 19200,
    """Each head in each layer produces an n × n weight matrix for self-attention: 20 × 20 = 400 numbers.

`6 layers × 8 heads × 400 = 19,200`

No causal mask is used in the encoder, so all 400 entries per matrix are used.""",
    tol=0.5, hint="integer", disp="19,200")

num(S, BASE + "\n\nA training pair has a source of n = 30 tokens and a target of m = 25 tokens. Count every attention-weight entry computed in the entire model (all 6 encoder layers: self-attention; all 6 decoder layers: masked self-attention **and** cross-attention; 8 heads each; include the masked entries).", 109200,
    """Entries per head: encoder self n² = 900; decoder self m² = 625; cross m·n = 750.

```
encoder self-attention : 6 × 8 × 900 = 43,200
decoder self-attention : 6 × 8 × 625 = 30,000
cross-attention        : 6 × 8 × 750 = 36,000
total                                 109,200
```""",
    tol=0.5, hint="integer", disp="109,200")

num(S, BASE + "\n\nIn one encoder self-attention layer with n = 100 tokens, how many scalar multiply-accumulate operations are needed to compute **QK^{T}** for all 8 heads? (Ignore the projections, the 1/√d_{k} scaling and the softmax.)", 5120000,
    """Per head: Q_{i} is 100×64 and K_{i}^{T} is 64×100, so the product is 100×100 and each of its 10,000 entries needs a 64-term dot product.

`100 × 100 × 64 = 640,000` per head; `× 8 heads = 5,120,000`.

In general this is n²·d_{k}·h = n²·d_{model}, i.e. O(n²·d), the term in the paper's complexity table.""",
    tol=0.5, hint="integer", disp="5,120,000")

num(S, "A larger Transformer has h = 16 attention heads. For one layer and **one** sequence of n = 1,024 tokens, the n × n score matrix of every head is stored in float32 (4 bytes per number). How many MiB (1 MiB = 2^{20} bytes) does that take?", 64,
    """Entries: 16 heads × 1,024 × 1,024 = 16,777,216.

Bytes: 16,777,216 × 4 = 67,108,864 bytes = 67,108,864 / 2^{20} = **64 MiB** per layer per sequence.

The cost is quadratic in length: doubling n to 2,048 would give 256 MiB.""",
    tol=0.5, hint="number of MiB", disp="64")

num(S, BASE + "\n\nHow many weights does one **decoder** layer contain? Count the masked self-attention block, the cross-attention block and the position-wise feed-forward network (two linear layers 512→2048→512); ignore biases and LayerNorm.", 4194304,
    """```
masked self-attention : 4 × 512²      = 1,048,576
cross-attention       : 4 × 512²      = 1,048,576
feed-forward          : 2 × 512×2048  = 2,097,152
total                                  4,194,304
```

An encoder layer has no cross-attention block, so it has 1,048,576 + 2,097,152 = 3,145,728 weights.""",
    tol=0.5, hint="integer", disp="4,194,304")

num(S, BASE + "\n\nIn the decoder's masked self-attention for a target of m = 7 tokens, how many entries of one head's 7 × 7 score matrix are **allowed** (not masked out)?", 28,
    """Position i may attend to positions 1..i, so row i has i allowed entries:

`1 + 2 + … + 7 = 7·8/2 = 28`

The other 49 − 28 = 21 entries (strictly above the diagonal) are masked, i.e. they get zero attention weight.""",
    tol=0.5, hint="integer", disp="28")

num(S, "Consider a Transformer layer with d_{model} = 768 and h = 12 heads (so d_{k} = 768/12 = 64). How many weights are there in all the **query projection matrices W^{Q}_{i} of one multi-head attention block** combined (i = 1..12)?", 589824,
    """Each head has W^{Q}_{i} ∈ ℝ^{768 × 64} = 49,152 weights.

`12 × 49,152 = 589,824 = 768 × 768`

Together, the heads' query projections form exactly one d_{model} × d_{model} matrix (the same for the keys and the values), which is why splitting into heads costs no extra parameters.""",
    tol=0.5, hint="integer", disp="589,824")

# ---------------------------------------------------------------- multi (2)
multi(S, BASE + "\n\nThe decoder's masked self-attention processes a target prefix of m = 7 tokens. Which statements about one head are correct? Select all that apply.",
      [("The score matrix has shape 7 × 7", "Queries and keys both come from the 7 target positions."),
       ("Scores for future positions (above the diagonal) are set to −∞ before the softmax, so their weights are exactly 0", "exp(−∞) = 0, and the remaining entries of each row are renormalised among themselves."),
       ("Row i of the weight matrix still sums to 1; its non-zero entries lie in columns 1 … i", "The softmax is computed over the allowed positions only.")],
      [("The score matrix has shape 7 × n, where n is the source length", "That is the shape of the **cross**-attention scores; masked self-attention only involves target positions."),
       ("Future positions are masked by setting their scores to 0 before the softmax", "A score of 0 gives exp(0) = 1, which still receives probability mass; the mask must use −∞ (or a very large negative number).")],
      "Masked self-attention keeps the model autoregressive: position i may only look at positions ≤ i. The shapes are the same as unmasked self-attention (Q, K, V: 7×64; scores: 7×7; output: 7×64).",
      calc=True)

multi(S, BASE + "\n\nIn the decoder's cross-attention, let n be the **source** length and m the **target** length. Which of these shapes depend on n? Select all that apply.",
      [("The encoder output E that supplies the keys and values, with shape n × 512", "E has one row per source token, so its first dimension is n."),
       ("K_{i}, with shape n × 64", "Keys are projections of the encoder output, which has one row per source token."),
       ("V_{i}, with shape n × 64", "Values come from the same encoder output as the keys."),
       ("The attention-weight matrix, with shape m × n", "One row per target position and one column per source position.")],
      [("The projection matrix W^{K}_{i}, with shape 512 × 64", "Weight matrices depend only on d_{model} and d_{k}; the same weights work for any sentence length."),
       ("One head's output, with shape m × 64", "Multiplying the m×n weights by the n×64 values removes n: the output has one row per target position.")],
      "Parameters never depend on sequence length. Activations that come from the source side (E, K, V, and the weight matrix's columns) scale with n; the cross-attention output has one row per query, so it scales with m only.\n\n" + CHEAT,
      calc=True)
