from qdsl import tf, single, multi, num, T

X = "transformer"

# ------------------------------------------------------------------ true / false
tf(X, "In the Transformer decoder, the self-attention sub-layer is masked so that position i cannot attend to any position j > i.", True,
   """The decoder is autoregressive. Illegal (future) connections get a score of −∞ before the softmax, so their attention weight becomes exactly 0.

Together with shifting the target sequence right by one position, this guarantees that the prediction for position i depends only on the known outputs at positions < i. That is the same condition that holds at test time, when future tokens do not exist yet.""")

tf(X, "Sinusoidal positional encodings are concatenated to the token embeddings, so the input of the first layer has dimension 2·d_{model}.", False,
   """They are **added** element-wise to the embeddings, which is why the positional encoding has the same dimension d_{model} as the embedding.

Concatenation would double the input width and force every weight matrix to change shape; it is not what the paper does.""")

tf(X, "A self-attention layer connects any two positions of a length-n sequence with a constant number of sequential operations, whereas a recurrent layer needs O(n) sequential operations.", True,
   """This is the complexity table of Vaswani et al.: self-attention has O(1) sequential operations and an O(1) maximum path length between any two positions; a recurrent layer has O(n) for both.

That is why self-attention parallelizes well and makes long-range dependencies easier to learn.""")

tf(X, "The per-layer complexity of self-attention is O(n·d^{2}), whereas that of a recurrent layer is O(n^{2}·d) (n = sequence length, d = representation dimension).", False,
   """The two costs are swapped.

- Self-attention: O(n^{2}·d). It builds an n×n score matrix and every entry is a d-dimensional dot product.
- Recurrent layer: O(n·d^{2}). There are n sequential steps and each is a d×d matrix–vector product.

Hence self-attention needs fewer operations per layer when n < d.""")

tf(X, "Layer normalization in the Transformer normalizes each position's d_{model}-dimensional vector with that vector's own mean and variance, so it does not depend on the batch size.", True,
   """LayerNorm computes its statistics over the feature dimension of a single token and then applies a learned gain and bias.

Unlike batch normalization it needs no batch statistics, so it behaves identically for batch size 1 at inference and for padded, variable-length sequences.""")

tf(X, "During training, the Transformer decoder has to produce the target tokens one at a time, so training the decoder is just as sequential as decoding.", False,
   """With teacher forcing the whole target sentence is known, so all target positions are processed **in parallel** in a single forward pass; the causal mask only prevents position i from seeing the future.

Only at **inference** is generation sequential, because token t+1 needs the model's own token t as input.""")

tf(X, "The position-wise feed-forward network applies the same two linear maps, with a ReLU in between, to every position independently.", True,
   """FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2}, applied identically at each position (it is like two 1×1 convolutions). The parameters differ from layer to layer.

In the base model it expands 512 → 2048 → 512. Information is exchanged between positions only in the attention sub-layers.""")

tf(X, "Because the sinusoidal positional encodings are learned parameters, the model cannot be applied to a sentence longer than the longest one seen in training.", False,
   """The sinusoids are a **fixed** function of the position and have no trainable parameters, so they can be computed for any position. The authors chose them partly because they might let the model extrapolate to longer sequences.

Learned positional embeddings (which performed nearly identically in the paper) would indeed be limited to a maximum length.""")

# ------------------------------------------------------------------ single answer
single(X, "What are the sub-layers of one decoder layer in the original Transformer, in order?",
      ("Masked multi-head self-attention → encoder–decoder (cross) multi-head attention → position-wise feed-forward network",
       "Causal self-attention over the target prefix comes first, then cross-attention queries the encoder output, then the FFN. Each sub-layer is wrapped in a residual connection and layer norm."),
      [("Multi-head self-attention → position-wise feed-forward network, each wrapped in a residual connection and layer norm",
        "That is an encoder layer. The decoder layer inserts a third sub-layer that attends to the encoder output, and its self-attention is masked."),
       ("Encoder–decoder attention → masked multi-head self-attention → position-wise feed-forward network",
        "The two attention blocks are in the wrong order: the decoder first builds a representation of the target prefix, and that representation then queries the encoder output."),
       ("Masked multi-head self-attention → position-wise feed-forward network → encoder–decoder attention",
        "Cross-attention sits between the self-attention and the feed-forward network, not after the FFN.")],
      "In addition to the two sub-layers of an encoder layer, the decoder inserts a third sub-layer that performs multi-head attention over the output of the encoder stack. The self-attention sub-layer is modified (masked) to prevent attending to later positions.")

single(X, "In the original Transformer, how is the output of each sub-layer (self-attention or feed-forward) combined with the sub-layer's input x?",
      ("LayerNorm(x + Sublayer(x)): add the input to the output (residual), then normalize the sum",
       "This is the post-LN arrangement of Vaswani et al. All sub-layer outputs have dimension d_{model} so that the sum is possible."),
      [("x + Sublayer(LayerNorm(x)): normalize first, then add the sub-layer output to the unnormalized input",
        "This 'pre-LN' variant was introduced later because it trains more stably; the original paper normalizes after the addition."),
       ("LayerNorm(Sublayer(x)): the skip connection is dropped because attention already mixes all positions",
        "Residual connections are used around every sub-layer; they make the deep stacks easier to optimize."),
       ("BatchNorm(x · Sublayer(x)): the sub-layer output gates the input multiplicatively",
        "The skip connection is additive and the normalization is layer normalization, not batch normalization.")],
      "Each sub-layer's output is `LayerNorm(x + Sublayer(x))`. Residual connections require all sub-layers and embedding layers to produce outputs of the same dimension, d_{model} = 512.")

single(X, "Why does the Transformer need positional encodings?",
      ("Self-attention is permutation-equivariant, so the input would otherwise be treated as an unordered bag of tokens",
       "Attention weights depend only on the content of the query and key vectors; nothing tells the model where a token sits in the sentence unless position information is injected."),
      [("They reduce the O(n^{2}) memory needed for the attention score matrix by giving each position a compact code",
        "Adding a vector to each embedding changes the values of the n×n score matrix but not its size, so the cost stays O(n^{2})."),
       ("They allow the encoder and the decoder to share the same layer parameters, which halves the model size",
        "Positional encodings are unrelated to weight sharing; the encoder and decoder have separate layer parameters."),
       ("They normalize the embeddings to unit length so that the dot products used in attention stay in a stable range",
        "Keeping dot products stable is the job of the 1/√d_{k} scaling and of layer norm. Positional encodings only inject order information.")],
      "Unlike RNNs (which read tokens in order) or CNNs (which use local, ordered windows), self-attention has no built-in notion of order. Permuting the input tokens merely permutes the outputs, so the model must be given position information, here by adding positional encodings to the input embeddings.")

single(X, "Which expression is the sinusoidal positional encoding of position pos for an even dimension index 2i in the original Transformer?",
      ("PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})",
       "Even dimensions use sine and odd dimensions use cosine. The wavelength grows geometrically from 2π to 10000·2π as i increases."),
      [("PE_{(pos, 2i)} = sin(pos · 10000^{2i/d_{model}})",
        "The position must be divided by the factor, not multiplied; the frequencies have to decrease with i."),
       ("PE_{(pos, 2i)} = cos(pos / 10000^{2i/d_{model}})",
        "Cosine is used for the odd dimensions 2i+1; the even dimensions use sine."),
       ("PE_{(pos, 2i)} = sin(pos / 10000^{i})",
        "The exponent must be 2i/d_{model} so that it runs from 0 to 1 across the d_{model} dimensions.")],
      """The two formulas are

`PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`  and  `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`.

Each dimension corresponds to a sinusoid, with wavelengths forming a geometric progression from 2π to 10000·2π.""")

single(X, "Which property did Vaswani et al. hypothesize as a reason for choosing sinusoidal positional encodings?",
      ("PE_{pos+k} is a linear function of PE_{pos} for every fixed offset k, so relative positions are easy to attend to",
       "That is the hypothesis stated in the paper; it follows from the angle-addition identities for sine and cosine."),
      [("Every position gets its own unique one-hot vector, so the network can index absolute positions like a lookup table",
        "The encodings are dense d_{model}-dimensional vectors, not one-hot vectors."),
       ("They are trained jointly with the word embeddings, so they adapt to the typical word order of the training language",
        "They are fixed, not trained; the learned alternative gave nearly identical results."),
       ("They make the attention matrix sparse, which reduces the complexity of self-attention from O(n^{2}) to O(n log n)",
        "Positional encodings only add a vector to the input; attention remains dense and O(n^{2}).")],
      "Two reasons were given: the linear-offset property (relative positions are easy to express) and the possibility of extrapolating to sequence lengths longer than those seen during training. The authors also tried learned positional embeddings and found nearly identical results.")

single(X, "According to Vaswani et al., when is a self-attention layer computationally cheaper (per layer) than a recurrent layer?",
      ("When the sequence length n is smaller than the representation dimension d",
       "Self-attention costs O(n^{2}·d) and a recurrent layer O(n·d^{2}); their ratio is n/d, which is below 1 exactly when n < d."),
      [("When the sequence length n is larger than the representation dimension d",
        "It is the other way round: for n > d the n^{2}·d term of self-attention dominates the n·d^{2} term of the recurrent layer."),
       ("Never — it always needs more operations because it compares every pair of positions",
        "For n < d (typical of sentence-level models with word-piece vocabularies) the n^{2}·d term is the smaller one."),
       ("Always, because computing the attention scores takes only a constant number of operations",
        "A constant number of *sequential* operations is not a constant number of arithmetic operations; the cost is still O(n^{2}·d).")],
      "Sentence representations in machine translation usually have n ≈ 20–100 tokens while d = 512 or more, so n < d holds and self-attention is cheaper than recurrence. (Parallelism is an additional advantage: O(1) sequential operations instead of O(n).)")

single(X, "What is the maximum path length between any two input positions in a (full) self-attention layer?",
      ("O(1)",
       "Every position attends directly to every other position, so a signal between two positions needs a single hop; this makes long-range dependencies easy to learn."),
      [("O(n)",
        "That is the maximum path length of a recurrent layer, where information must pass through every intermediate state."),
       ("O(log_{k} n)",
        "That is the path length of a stack of dilated convolutions with kernel size k."),
       ("O(n/r)",
        "That is the path length of restricted self-attention in which each position only attends to a neighbourhood of size r.")],
      "The paper's complexity table lists, for each layer type, the per-layer complexity, the number of sequential operations and the maximum path length. Self-attention: O(n^{2}·d), O(1), O(1). Recurrent: O(n·d^{2}), O(n), O(n). Convolutional: O(k·n·d^{2}), O(1), O(log_{k} n). Restricted self-attention: O(r·n·d), O(1), O(n/r).")

single(X, "In the Transformer's encoder–decoder (cross) attention sub-layer, where do the queries, keys and values come from?",
      ("Queries come from the previous decoder layer; keys and values come from the output of the encoder stack",
       "The decoder asks the questions and the encoder's final representations provide the memory that is searched and read."),
      [("Queries come from the encoder output; keys and values come from the previous decoder layer, which is then read by the encoder",
        "The roles are reversed: the decoder is the one that queries."),
       ("Queries, keys and values all come from the previous decoder layer, exactly as in the masked self-attention",
        "That describes the decoder's (masked) self-attention sub-layer, not the cross-attention sub-layer."),
       ("Queries, keys and values all come from the final layer of the encoder, as in the encoder's own self-attention",
        "That would be self-attention over the source; it is not cross-attention and gives the decoder no way to query the source.")],
      "This mimics the typical encoder–decoder attention of seq2seq models: every decoder position can attend over all positions of the source sequence. Every decoder layer uses the same encoder output (computed once) as keys and values.")

single(X, "What effect did label smoothing (ε_{ls} = 0.1) have in the original Transformer experiments?",
      ("It hurt perplexity but improved accuracy and BLEU",
       "The model is trained to be less than fully certain, which raises perplexity, yet accuracy and BLEU improved (Vaswani et al., §5.4)."),
      [("It improved perplexity as well as BLEU, since the model became better calibrated",
        "Perplexity got worse: the model learns to put some probability mass on the other tokens."),
       ("It lowered BLEU but improved perplexity, since the targets carried less information",
        "This is reversed: BLEU and accuracy improved while perplexity got worse."),
       ("It had no measurable effect on perplexity, accuracy or BLEU",
        "The paper reports a clear trade-off: worse perplexity, better accuracy and BLEU.")],
      "Label smoothing replaces the one-hot target by a distribution that puts 1−ε on the correct token and spreads ε over the rest. The model learns to be more unsure, which hurts perplexity (a metric that rewards confident correct predictions) but improves accuracy and BLEU.")

single(X, "Which learning-rate schedule did the original Transformer use with the Adam optimizer?",
      ("Linear warm-up for the first 4000 steps, then decay proportional to the inverse square root of the step number",
       "lrate = d_{model}^{−0.5} · min(step^{−0.5}, step · warmup_steps^{−1.5}) with warmup_steps = 4000."),
      [("A constant learning rate of 10^{−3} with no warm-up, since Adam adapts the step sizes automatically and needs no schedule",
        "The schedule is not constant; the warm-up is important for stable training of the post-LN Transformer."),
       ("Cosine decay from a large initial value to zero over the whole training run, with no warm-up phase at all",
        "Cosine schedules are common elsewhere but are not what the paper used."),
       ("Exponential growth for the first 4000 steps, followed by a constant plateau until the end of training",
        "The rate grows linearly (not exponentially) during warm-up and then decays as step^{−0.5}.")],
      "The rate rises linearly for the first warmup_steps = 4000 steps and decreases afterwards proportionally to the inverse square root of the step number. The optimizer settings were Adam with β_{1} = 0.9, β_{2} = 0.98 and ε = 10^{−9}.")

single(X, "Why does multi-head attention use d_{k} = d_{v} = d_{model}/h for each of the h heads instead of d_{model} per head?",
      ("So that the total computational cost is similar to single-head attention with full dimensionality",
       "h heads of width d_{model}/h cost about the same as one head of width d_{model}, while giving h separate attention patterns."),
      [("So that the number of parameters of the attention layer grows linearly with the number of heads, giving more capacity",
        "The parameter count stays at about 4·d_{model}^{2} regardless of h, because each head becomes narrower as h grows."),
       ("Because the softmax saturates whenever d_{k} exceeds 64, so wider heads cannot be trained",
        "There is no such hard threshold; the 1/√d_{k} scaling exists to cope with large d_{k}."),
       ("So that the concatenated head outputs have dimension h·d_{model}, matching the width of the FFN",
        "The concatenation has dimension h·d_{v} = d_{model}, and it is then projected by W^{O}; it does not match the FFN width.")],
      "With the reduced dimension of each head, the total cost is similar to that of single-head attention with full dimensionality (d_{model} = 512, h = 8, d_{k} = d_{v} = 64).")

single(X, "What is the main advantage of multi-head attention over a single attention head?",
      ("It lets the model attend jointly to different representation subspaces at different positions",
       "Each head has its own projections, so different heads can focus on different kinds of relations at the same time; a single head's averaging inhibits this."),
      [("It reduces the complexity of attention from O(n^{2}) to O(n), because the work is split across the heads",
        "Every head still computes an n×n score matrix; the complexity is unchanged."),
       ("It removes the need for positional encodings, because every head learns its own notion of position",
        "Heads are permutation-equivariant just like a single head; position information must still be injected."),
       ("It guarantees that each head specializes in one interpretable linguistic role such as syntax or coreference",
        "Some heads do show such behaviour, but nothing in the architecture guarantees it.")],
      "With a single head, averaging over positions inhibits the model from focusing on different aspects at once. With h heads the model can, for example, track the syntactic head of a word in one head and a nearby modifier in another.")

single(X, "What does 'output embeddings shifted right' mean for the Transformer decoder's input?",
      ("The decoder input is the target offset by one position, so position i is predicted only from earlier tokens",
       "During training the decoder input is <start>, y_{1}, …, y_{m−1} and the targets are y_{1}, …, y_{m}; with the causal mask this prevents the model from simply copying the answer."),
      [("The target tokens are cyclically rotated so that the last word is moved to the front of the decoder input",
        "A rotation would put the final target word at the start of the input and leak it to every other position."),
       ("The source sentence is shifted right by one position so that encoder and decoder positions line up",
        "The shift applies to the target sequence only; the source sentence is fed to the encoder unchanged."),
       ("The positional encoding of every target token is increased by one so that decoder positions are aligned with the encoder",
        "Positions are not re-indexed; the shift is about which token is the input and which is the prediction target.")],
      "The decoder predicts token y_{i} from y_{<i}. Shifting the output embeddings right by one position, combined with masking, ensures that the predictions for position i can depend only on the known outputs at positions less than i.")

single(X, "How are the embedding layers and the output softmax layer treated in the base Transformer of the paper?",
      ("Both embedding layers and the pre-softmax layer share one weight matrix; embeddings are multiplied by √d_{model}",
       "Weight tying saves parameters; the paper only states that the embedding weights are multiplied by √d_{model} (a common explanation: it keeps the embedding scale comparable to the added positional encodings)."),
      [("Three separate, unrelated matrices are used for the input embedding, the output embedding and the pre-softmax layer",
        "The paper explicitly shares the same weight matrix between them."),
       ("The embeddings are divided by √d_{model} so that they stay smaller than the positional encodings added to them",
        "They are multiplied, not divided, by √d_{model}."),
       ("The pre-softmax linear layer is shared with the first layer of the decoder's feed-forward network to save memory",
        "No such sharing exists; the tie is with the embedding matrices.")],
      "Section 3.4 of the paper: the same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation, and in the embedding layers those weights are multiplied by √d_{model}.")

# ------------------------------------------------------------------ multiple answer
multi(X, "Which of the following are present in every layer of the **encoder** stack of the original Transformer?",
      [("Multi-head self-attention",
        "The first sub-layer lets every source position attend to all source positions (bidirectionally, without a causal mask)."),
       ("A position-wise feed-forward network",
        "The second sub-layer: two linear maps with a ReLU in between, applied to each position separately."),
       ("A residual connection followed by layer normalization around each sub-layer",
        "Both sub-layers are wrapped as LayerNorm(x + Sublayer(x)); every output has dimension d_{model} so the addition is possible.")],
      [("Encoder–decoder (cross) attention",
        "That sub-layer exists only in the decoder; the encoder has no other sequence to attend to."),
       ("A causal mask on the self-attention scores",
        "Encoder self-attention is bidirectional. Only the decoder's self-attention is masked (apart from padding masks)."),
       ("A recurrent (LSTM or GRU) sub-layer",
        "The Transformer dispenses with recurrence and convolution entirely; that is the point of the paper.")],
      "An encoder layer has exactly two sub-layers (multi-head self-attention and a position-wise feed-forward network), each with a residual connection and layer normalization. Cross-attention and the causal mask belong to the decoder.")

multi(X, "Which statements about the sinusoidal positional encoding of the original Transformer are correct?",
      [("It is added element-wise to the token embedding, so it has dimension d_{model}",
        "Addition requires equal dimensions; that is why PE has the same width as the embeddings."),
       ("Different dimensions use sinusoids of different wavelengths, forming a geometric progression from 2π to 10000·2π",
        "Low dimensions oscillate quickly and high dimensions slowly, so each position gets a distinctive pattern."),
       ("It is a fixed function of the position with no trainable parameters",
        "The sinusoids are computed from the position index and d_{model} alone, so there is nothing to learn.")],
      [("It depends on which word occupies the position, so the same position in two sentences gets different codes",
        "It depends only on the position index (and d_{model}), never on the content of the token."),
       ("It is added only to the queries and keys inside each attention layer, not to the embeddings at the bottom of the stacks",
        "In the original Transformer the encodings are added once to the input embeddings at the bottom of both stacks; injecting position inside the attention layers is characteristic of later relative-position variants."),
       ("It is applied only to the encoder input; the decoder relies on its causal mask to know the word order",
        "Both the encoder and the decoder embeddings get positional encodings; the causal mask only restricts visibility and does not encode order.")],
      "Positional encodings are added to the input embeddings at the bottom of both stacks. Being a deterministic function of position they have no parameters to learn.")

multi(X, "Which statements comparing self-attention and recurrent layers (as in the paper's complexity table) are correct?",
      [("Self-attention needs O(1) sequential operations per layer; a recurrent layer needs O(n)",
        "All positions of a self-attention layer can be computed at once; recurrence has to go step by step."),
       ("The maximum path length between two positions is O(1) for self-attention and O(n) for a recurrent layer",
        "Self-attention connects any pair directly; a recurrent signal has to pass through every intermediate state."),
       ("Self-attention has per-layer complexity O(n^{2}·d); a recurrent layer has O(n·d^{2})",
        "n^{2}·d for the pairwise dot products, n·d^{2} for n matrix–vector products with d×d matrices."),
       ("Restricted self-attention, where each position attends only to r neighbours, has complexity O(r·n·d) and maximum path length O(n/r)",
        "Restricting the neighbourhood lowers the cost for very long sequences but lengthens the paths between distant positions.")],
      [("The attention score matrix needs only O(n·d) memory because each position attends to d others",
        "Each head stores an n×n matrix of scores, so the memory is O(n^{2}); this quadratic cost limits the context length."),
       ("Recurrent layers are cheaper than self-attention for every n and d, so self-attention can never be faster",
        "For n < d self-attention needs fewer operations per layer, and it also needs far fewer sequential steps.")],
      "Self-attention trades a quadratic dependence on n for a constant number of sequential operations and a constant path length. It is cheaper than recurrence when n < d; for very long sequences the O(n^{2}) cost becomes the bottleneck.")

multi(X, "Which of the following were part of the training setup of the original Transformer?",
      [("The Adam optimizer with β_{1} = 0.9, β_{2} = 0.98 and ε = 10^{−9}",
        "Adam was used with a warm-up learning-rate schedule; these non-default β_{2} and ε values are reported in the paper."),
       ("Residual dropout (P = 0.1 in the base model) on the sub-layer outputs and on the sums of embeddings and positional encodings",
        "Dropout is applied to each sub-layer's output before it is added to the input and normalized, and to the embedding + positional-encoding sums."),
       ("Label smoothing with ε_{ls} = 0.1",
        "Label smoothing replaces the one-hot target by a softened distribution, so the model is trained to be less over-confident."),
       ("A byte-pair-encoded subword vocabulary shared between source and target (about 37,000 tokens for English–German)",
        "BPE gives a fixed-size vocabulary in which any word made of known characters can be written; sharing it between languages is what makes tying the embedding matrices possible.")],
      [("Batch normalization after each sub-layer",
        "Layer normalization is used, not batch normalization."),
       ("Scheduled sampling: feeding the model's own predictions to the decoder during training",
        "Training uses standard teacher forcing with the ground-truth previous tokens, which allows all positions to be processed in parallel.")],
      "The training recipe consists of BPE-tokenized data, Adam with a warm-up schedule, residual/embedding dropout and label smoothing, with layer normalization and residual connections inside the network.")

multi(X, "Which statements about masking in the Transformer are correct?",
      [("In decoder self-attention, the scores of illegal (future) positions are set to −∞ before the softmax, giving them zero attention weight",
        "Setting the scores to −∞ makes the softmax output exactly 0 for those positions, and the remaining weights still sum to 1."),
       ("Together with the right-shifted target input, the mask makes the prediction for position i depend only on positions before i",
        "This preserves the autoregressive property while still allowing all positions to be computed in parallel during training."),
       ("Padded positions can be masked in the same way so that real tokens never attend to padding",
        "Padding masks are applied to the scores just like the causal mask; they matter when sentences in a batch have different lengths.")],
      [("The encoder's self-attention is also causally masked, so source position i cannot see later source words",
        "The encoder is bidirectional: each source position attends to the whole source sentence."),
       ("The mask is applied after the softmax by zeroing the probabilities, leaving the other weights unnormalized",
        "The mask is applied before the softmax, so the surviving weights are renormalized automatically and sum to 1.")],
      "The mask is an additive −∞ on the entries above the diagonal of the score matrix (before the softmax). It is used only in the decoder's self-attention to stop information flowing from future positions; padding masks are used wherever sequences are padded.")

multi(X, "Which statements about layer normalization as used in the Transformer are correct?",
      [("For each position it computes a mean and a variance over the d_{model} features of that position",
        "The statistics are taken over the feature dimension of a single token vector."),
       ("The output for a token does not depend on the other sequences in the batch",
        "No batch statistics are used, so the result is the same for any batch size and at training and test time.")],
      [("It computes its statistics across the batch dimension, like batch normalization",
        "That is batch normalization; layer normalization works within each example."),
       ("It computes its mean and variance across the sequence positions (separately for each feature), so the output of one token depends on the other tokens of the sentence",
        "The statistics are taken over the d_{model} features of one token, not over positions; normalizing over positions would couple the tokens of a sentence and make the result depend on sentence length and padding."),
       ("It is applied only once, after the final layer, rather than around every sub-layer",
        "In the original Transformer every sub-layer output is normalized: LayerNorm(x + Sublayer(x)).")],
      "Layer normalization is `LN(x) = γ ⊙ (x − μ)/σ + β` with μ and σ computed over the d_{model} entries of x, followed by a learned per-feature gain γ and bias β. This makes it independent of the batch size and well suited to variable-length sequences.")

multi(X, "Which of the following explain why Transformers train faster than recurrent seq2seq models on parallel hardware?",
      [("All source positions are processed simultaneously in the encoder",
        "Position t does not depend on the hidden state of position t−1, so the whole sentence is processed in one pass."),
       ("With teacher forcing, all target positions are processed in one pass; the causal mask enforces the autoregressive property",
        "The decoder does not need its own earlier outputs during training because the ground truth is available."),
       ("Most of the computation consists of large matrix multiplications, which GPUs and TPUs execute very efficiently",
        "Projections, score computation and the FFN are all dense matrix products.")],
      [("Self-attention needs fewer arithmetic operations than a recurrent layer for every n and d",
        "O(n^{2}·d) is smaller than O(n·d^{2}) only if n < d; the speed-up comes from parallelism, not from fewer operations in general."),
       ("Generating a translation at test time needs only a single forward pass for the whole output sentence",
        "Inference is still autoregressive: each token is generated after the previous ones.")],
      "The training speed-up comes from the O(1) sequential depth per layer (versus O(n)) and the hardware-friendly operations. At inference the decoder is still sequential.")

multi(X, "Which of the following are found in the decoder layers but NOT in the encoder layers of the original Transformer?",
      [("A masked (causal) self-attention sub-layer",
        "The decoder must not look at future target positions."),
       ("A sub-layer that attends over the output of the encoder stack",
        "Cross-attention: queries from the decoder, keys and values from the encoder output.")],
      [("A position-wise feed-forward network",
        "Both the encoder and the decoder layers contain one."),
       ("Residual connections around the sub-layers",
        "Both stacks use them (followed by layer normalization)."),
       ("Positional encodings added to the input embeddings",
        "Both stacks receive embeddings plus positional encodings.")],
      "Encoder layer = self-attention + FFN. Decoder layer = masked self-attention + cross-attention + FFN. Embeddings with positional encodings, residual connections and layer normalization are shared design elements of both stacks.")

multi(X, "Which statements about the base Transformer configuration of the paper are correct?",
      [("The encoder and the decoder each have N = 6 identical layers",
        "Both stacks consist of six identical layers."),
       ("d_{model} = 512 and the inner feed-forward dimension is d_{ff} = 2048",
        "The FFN expands 512 → 2048 → 512, i.e. a factor of 4."),
       ("h = 8 heads with d_{k} = d_{v} = 64",
        "d_{k} = d_{model}/h = 512/8 = 64."),
       ("Dropout P_{drop} = 0.1 and label smoothing ε_{ls} = 0.1",
        "These are the regularization settings of the base model.")],
      [("h = 16 heads with d_{k} = d_{v} = 32",
        "The base model has h = 8 (d_{k} = d_{v} = 64). h = 16 with d_{k} = d_{v} = 32 is only one of the ablation variants in the Table 3 block on attention heads; the big model uses h = 16 with d_{model} = 1024, so d_{k} = d_{v} = 64."),
       ("d_{ff} = d_{model} = 512, so the feed-forward network does not change the width",
        "The inner layer is four times wider: d_{ff} = 2048.")],
      "Base model: N = 6, d_{model} = 512, d_{ff} = 2048, h = 8, d_{k} = d_{v} = 64, dropout 0.1. Big model: d_{model} = 1024, d_{ff} = 4096, h = 16.")

multi(X, "Which statements about generating a translation with a trained Transformer are correct?",
      [("Output tokens are generated autoregressively: each new token is fed back into the decoder",
        "The decoder has to know the previous outputs before it can predict the next token."),
       ("The encoder output is computed once and reused as keys and values at every decoding step",
        "The encoder depends only on the source sentence, so it does not need to be re-run."),
       ("The paper used beam search with beam size 4 and a length penalty of α = 0.6",
        "Those were the decoding settings reported for the translation experiments.")],
      [("All output tokens are predicted simultaneously in a single forward pass of the decoder",
        "Without the earlier tokens as input this is impossible at inference; simultaneous prediction is only possible during training with teacher forcing."),
       ("The encoder must be re-run on the growing target prefix at every step",
        "The encoder never sees the target; only the decoder processes the prefix.")],
      "At inference the source is encoded once; the decoder then generates one token per step (greedy or beam search), attending to the fixed encoder output and to its own previous outputs.")

# END
