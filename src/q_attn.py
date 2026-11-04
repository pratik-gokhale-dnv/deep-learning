from qdsl import tf, single, multi, num, T

A = "attn"

# ------------------------------------------------------------------ true / false
tf(A, "In a seq2seq model with attention, the context vector used at decoder step t is a weighted average of the encoder hidden states, with weights obtained by a softmax over alignment scores.", True,
   """`c_t = Σ_i α_(t,i) · h_i` with `α_t = softmax(e_t)`, where e_{t,i} scores how well encoder state h_{i} matches the current decoder state.

The weights are non-negative and sum to 1, and they are recomputed at every decoder step, so the model can focus on different source words when producing different target words.""")

tf(A, "The factor 1/√d_{k} in scaled dot-product attention is there to make the attention weights sum to one.", False,
   """Normalization to a probability distribution is already done by the softmax.

The scaling counteracts the growth of the dot products with d_{k}: if the components of q and k are independent with mean 0 and variance 1, then q·k has variance d_{k}. For large d_{k} the scores become large in magnitude, the softmax saturates (almost one-hot) and its gradients become tiny.""")

tf(A, "Soft attention is differentiable and can be trained with ordinary backpropagation, whereas hard attention (selecting one location by sampling) needs a different gradient estimator such as REINFORCE.", True,
   """Soft attention uses the expectation over locations, i.e. a weighted average, which is a smooth function of the scores. Hard attention picks a discrete location, so gradients cannot flow through the sampling step and the model is trained with sampling-based estimators (REINFORCE-style or variational lower bounds).

Xu et al. (2015, *Show, Attend and Tell*) compared both variants for image captioning.""")

tf(A, "Large attention weights prove that the corresponding input tokens caused the model's prediction, so attention maps are guaranteed to be faithful explanations.", False,
   """Attention weights are often informative and make nice alignment plots, but they are not guaranteed to be explanations.

The output also depends on the value vectors, on other heads and layers, and on the residual paths; and different attention distributions can lead to the same prediction (Jain & Wallace, 2019, *Attention is not Explanation*).""")

tf(A, "Without positional encodings, an *unmasked* self-attention layer is permutation-equivariant: permuting the input tokens permutes the output vectors in exactly the same way.", True,
   """Each output is a weighted sum of value vectors whose weights depend only on the contents of the queries and keys. Reordering the inputs just reorders the rows of Q, K and V consistently, so the output rows are reordered in the same way.

That is why the Transformer has to add positional information to the embeddings. (This holds for unmasked, encoder-style self-attention; a causal mask breaks the symmetry.)""")

tf(A, "In additive (Bahdanau-style) attention the alignment score is the plain dot product of the decoder state and the encoder state, with no learned parameters.", False,
   """Additive attention scores a pair with a small feed-forward network, `e_(t,i) = v^T tanh(W_s s_(t−1) + W_h h_i)`, whose parameters W_{s}, W_{h} and v are learned.

The parameter-free dot product is one of the scoring functions compared by Luong et al., together with the bilinear 'general' score and the 'concat' score (essentially Bahdanau's additive form).""")

# ------------------------------------------------------------------ single answer
single(A, "Which limitation of the plain encoder–decoder (seq2seq) model motivated neural attention?",
      ("The entire source sentence must be squeezed into one fixed-size vector, a bottleneck for long sentences",
       "Bahdanau et al. argued that this fixed-length vector is why quality drops quickly as the source sentence gets longer."),
      [("A recurrent encoder cannot process source sentences whose length differs from the sentences seen in training",
        "An RNN is simply unrolled for as many steps as the sentence has tokens, so it can process any length; the problem is that quality degrades for long sentences because everything must be squeezed into one fixed-size vector."),
       ("The decoder cannot be trained with teacher forcing when the encoder is recurrent, so it needs another mechanism",
        "Teacher forcing works with any encoder; it concerns the decoder inputs during training and is unrelated to attention."),
       ("The softmax output layer has too many classes to be trained efficiently when the target vocabulary is large",
        "A large output vocabulary is a real cost, but attention does nothing to the output layer.")],
      "In the basic encoder–decoder model the encoder summarizes the entire source sentence in its final hidden state, and the decoder has to generate the whole translation from that single vector. Attention lets the decoder look back at all encoder states at every step and choose the relevant ones.")

single(A, "In what order are the computations carried out at one decoder step of an attention model?",
      ("Score each encoder state against the decoder state → softmax over the source positions → weighted sum of encoder states",
       "scores e_{t,i} → weights α_{t,i} = softmax(e_{t})_{i} → context c_{t} = Σ_{i} α_{t,i} h_{i}."),
      [("Average the encoder states uniformly → score the average against the decoder state → use that score as the context vector",
        "The encoder states must be scored individually; a uniform average throws away exactly the selectivity attention is meant to provide."),
       ("Apply a softmax to the raw encoder states → use them as the scores → sum the decoder states using these weights as coefficients",
        "The softmax is applied to the alignment scores between the decoder state and each encoder state, not to the encoder states themselves; and it is the encoder states that are averaged."),
       ("Sum all encoder states → score the sum against the decoder state → scale the decoder state by the resulting number",
        "Summing first removes the per-position information; the context vector is a weighted sum that depends on the per-position weights.")],
      "Every decoder step repeats the same three stages: (1) alignment scores between the decoder state and every encoder state, (2) normalization of the scores with a softmax into weights, (3) the context vector as the weighted sum of the encoder states. The context is then combined with the decoder state to predict the next word.")

single(A, "How do Bahdanau et al. (2015) and Luong et al. (2015) differ in the decoder state they use to compute the attention weights?",
      ("Bahdanau uses the previous decoder state s_{t−1}; Luong uses the current decoder state h_{t}",
       "In Bahdanau attention the context c_{t} computed from s_{t−1} is an input of the decoder RNN step; in Luong attention the context is computed from the current state and combined with it afterwards, as in the attentional vector tanh(W_{c}[c_{t}; h_{t}])."),
      [("Bahdanau uses the current decoder state h_{t}; Luong uses the previous decoder state s_{t−1}",
        "This swaps the two models: it is Bahdanau who uses the previous state s_{t−1}."),
       ("Both use only the final encoder state, so the context vector is identical at every decoder step",
        "In both models the weights are recomputed at every decoder step, so the context vector changes with t."),
       ("Bahdanau attends over one sampled source position (hard); Luong averages over all positions (soft)",
        "Both are soft attention models; neither samples a single source position.")],
      "Bahdanau (additive) attention: `s_t = f(s_(t−1), y_(t−1), c_t)` with `c_t` computed from `s_(t−1)` and the encoder states. Luong attention: first `h_t = f(h_(t−1), y_(t−1))`, then `c_t` from `h_t`, then `h~_t = tanh(W_c [c_t; h_t])` is used to predict the next word.")

single(A, "Why are the dot products divided by √d_{k} in scaled dot-product attention?",
      ("For large d_{k} the dot products have large variance (≈ d_{k}), pushing the softmax into saturated regions with tiny gradients",
       "If q and k have independent zero-mean, unit-variance components, q·k has variance d_{k}; dividing by √d_{k} brings it back to variance 1."),
      [("To normalize the queries and keys to unit length, so that the dot product becomes a cosine similarity",
        "Dividing by the constant √d_{k} only rescales the scores; a cosine similarity would divide by the data-dependent norms ‖q‖·‖k‖."),
       ("To reduce the cost of attention from O(n^{2}·d) to O(n·d) by making the score matrix sparse for long sequences",
        "The scaling is one cheap element-wise operation; the cost of QK^{T} is unchanged."),
       ("To make the scores independent of the sequence length n, so that longer sentences do not get sharper weights",
        "The variance of a dot product depends on the vector dimension d_{k}, not on how many keys there are.")],
      "Without scaling, the magnitude of q·k grows like √d_{k}. Very large inputs make the softmax nearly one-hot, and its gradient nearly zero, so learning stalls. Dividing by √d_{k} keeps the scores at a scale where the softmax has useful gradients.")

single(A, "What is the main idea of Luong et al.'s *local* attention compared with *global* attention?",
      ("Each target step attends only to a window of source positions around an aligned position p_{t}, not to all positions",
       "The window [p_{t}−D, p_{t}+D] bounds the number of scored positions, which makes long sentences cheaper; p_{t} is either monotonic (local-m) or predicted (local-p)."),
      [("Each target step attends only to the last encoder state, which is cheaper because only one score has to be computed",
        "That would be the plain encoder–decoder model without attention."),
       ("Each target step attends to one randomly sampled source position, so no softmax is needed and the model is not differentiable",
        "Sampling a single position is hard attention; Luong's local attention is soft and differentiable inside its window."),
       ("The weights are computed without a softmax, so they do not have to sum to one over the candidate source positions",
        "Local attention still normalizes the weights with a softmax over the window; only the set of candidate positions changes.")],
      "Global attention attends to all source words at each step; local attention first picks an aligned position p_{t} and then uses a window around it (optionally with a Gaussian weighting centred at p_{t}). It is a compromise between soft and hard attention that stays differentiable.")

single(A, "Why is (scaled) dot-product attention preferred over additive attention in the Transformer?",
      ("It maps onto highly optimized matrix multiplication, so it is faster and more space-efficient in practice",
       "QK^{T} is a single large matrix product; additive attention needs a feed-forward network for every query–key pair."),
      [("It has more trainable parameters than additive attention, which gives it greater modelling capacity on long sequences",
        "Plain dot-product scoring has no scoring parameters at all (only the surrounding Q, K, V projections)."),
       ("It is the only scoring function whose resulting weights are guaranteed to sum to one over the keys",
        "Any score function followed by a softmax yields weights that sum to one."),
       ("It outperforms additive attention for every value of d_{k}, even without any 1/√d_{k} scaling factor",
        "Without scaling, additive attention outperforms dot-product attention for large d_{k}; that is why the scaling was introduced.")],
      "The two are similar in theoretical complexity, but dot-product attention maps onto matrix multiplication, which hardware executes very efficiently. For small d_{k} the two perform similarly; for larger d_{k} unscaled dot products hurt, which the 1/√d_{k} factor repairs.")

single(A, "What is the key idea of self-attention (intra-attention)?",
      ("Every position attends to all positions of the same sequence to compute its new representation",
       "Queries, keys and values are all computed from the same sequence, so each output mixes information from the whole sequence."),
      [("Every decoder position attends to the positions of a different sequence, the source sentence encoded by the encoder",
        "That describes encoder–decoder (cross) attention, where queries and keys/values come from different sequences."),
       ("Every position attends only to the immediately preceding position, like the recurrent connection of an RNN",
        "Self-attention connects all pairs of positions directly; the connection pattern is not restricted to the predecessor."),
       ("Every position attends only to a fixed-size window of neighbouring positions, like the kernel of a convolution",
        "Restricted (local) self-attention exists, but standard self-attention covers the whole sequence.")],
      "Self-attention relates different positions of a single sequence in order to compute a representation of the sequence. It has been used for reading comprehension, summarization and sentence representations, and it is the core of the Transformer.")

single(A, "In multi-head attention with h heads, how are the outputs of the heads combined?",
      ("They are concatenated and multiplied by an output projection matrix W^{O}",
       "MultiHead(Q,K,V) = Concat(head_{1}, …, head_{h}) W^{O}, which maps the concatenation back to d_{model} dimensions."),
      [("They are averaged over the heads, and no further projection is applied",
        "Averaging would blur the different views the heads learned; the paper concatenates and projects."),
       ("Only the head with the sharpest attention distribution is kept and the others are discarded",
        "All heads contribute to the output; none is dropped at inference."),
       ("They are summed and passed through another softmax to produce the final weights",
        "The softmax is inside each head; the head outputs are vectors, not scores that need normalizing.")],
      "Each head i computes `softmax(Q W_i^Q (K W_i^K)^T / √d_k) V W_i^V`. The h results (each n × d_v) are concatenated into an n × (h·d_v) matrix and multiplied by W^O of shape (h·d_v) × d_model.")

single(A, "In the attention weight matrix of a translation model (rows = target steps, columns = source words), which statement is true?",
      ("Each row is a probability distribution over the source words, summing to 1",
       "The weights of one target step come from a softmax over the source positions, so the row sums to 1 and every entry is positive."),
      [("Each column sums to 1, because every source word has to be translated exactly once",
        "Nothing forces column sums to 1: a source word may be attended to by several target words, or by none."),
       ("Each row is one-hot, because every target word is aligned to exactly one source word",
        "Soft attention yields a distribution spread over several positions; one-hot rows would be hard attention."),
       ("Entries can be negative, which lets the model actively suppress a source word",
        "Softmax outputs are strictly positive; suppression is expressed by a weight close to 0.")],
      "Soft attention weights are positive and normalized per target step. For similar languages the matrix is roughly diagonal with local reorderings (for example adjective–noun order in English–French), which is why the weights are often visualized as soft word alignments.")

single(A, "In the key–value (dictionary look-up) view of attention, which description is correct?",
      ("The query is compared with the keys to get weights; the output is the weighted sum of the values",
       "Keys are used for matching, values for reading: output = Σ_{i} softmax(score(q, k_{i})) · v_{i}."),
      [("The query is compared with the values, and the output is then the weighted sum of the keys",
        "This swaps the roles: matching is done on keys and the content that is returned comes from the values."),
       ("The output is the single value whose key is most similar to the query, as in an exact-match dictionary lookup",
        "Soft attention returns a weighted average of all values; an argmax lookup would be a hard version."),
       ("Keys and values must be identical vectors, otherwise the attention weights cannot be computed",
        "Keys and values can differ (the Transformer uses separate projections W^{K} and W^{V}); in the classic seq2seq model they happen to coincide with the encoder states.")],
      "Attention is a differentiable memory lookup. The query asks a question, the keys are 'addresses' that are matched against the query, and the values are the 'contents' that are blended according to the match.")

# ------------------------------------------------------------------ multiple answer
multi(A, "Which statements about scaled dot-product attention, softmax(QK^{T}/√d_{k})V, are correct?",
      [("Q and K must have the same width d_{k}, because their dot products define the scores",
        "Each score is the dot product of one query row with one key row."),
       ("K and V must have the same number of rows, one value per key",
        "The weights over the n_{k} keys are used to average the n_{k} value vectors."),
       ("The output has one d_{v}-dimensional row per query",
        "The result is an n_{q} × d_{v} matrix: each query receives its own weighted average of the values.")],
      [("The softmax is taken over the queries, so that the weights of each key sum to 1",
        "The softmax is applied along the key dimension (row-wise): the weights of each query sum to 1."),
       ("V must have the same width d_{k} as Q and K",
        "The width d_{v} of the values is independent of d_{k}; it only determines the width of the output.")],
      "Shapes: Q is n_{q}×d_{k}, K is n_{k}×d_{k}, V is n_{k}×d_{v}; the score matrix QK^{T} is n_{q}×n_{k}, the softmax acts on each row, and the output is n_{q}×d_{v}.")

multi(A, "Which of the following are benefits of adding attention to an RNN encoder–decoder model?",
      [("The decoder can access all encoder states directly, which shortens the path between a source word and the target words it influences",
        "Gradients and information no longer have to travel through the single final encoder state."),
       ("Translation quality degrades less as the source sentence gets longer",
        "Bahdanau et al. showed that the attention model is much more robust to sentence length than the plain encoder–decoder."),
       ("The attention weights give a soft alignment between source and target words that can be inspected",
        "Visualizing the weights shows which source words the decoder used at each step.")],
      [("The recurrent encoder no longer has to process the source sentence step by step",
        "Attention changes how the decoder reads the encoder states; an RNN encoder is still sequential."),
       ("The decoder no longer needs its own previous output word as input at each step",
        "Generation is still autoregressive: the previous word remains an input of the decoder step.")],
      "Attention removes the fixed-size bottleneck, improves long-sentence translation and yields soft alignments that can be visualized (though attention weights are not guaranteed to be faithful explanations). It does not remove the sequential nature of RNN encoders and decoders; removing it requires dropping recurrence altogether, as in the Transformer (or convolutional seq2seq models).")

multi(A, "Which of the following are differences between Bahdanau et al. (2015) and Luong et al. (2015) attention?",
      [("Bahdanau computes the attention from the previous decoder state s_{t−1}; Luong uses the current decoder state h_{t}",
        "The context enters the RNN update in Bahdanau's model, but is combined with the current state afterwards in Luong's."),
       ("Bahdanau uses an additive score (a small feed-forward network); Luong et al. compare it with the multiplicative dot and general scores",
        "The dot and general scores are parameter-light multiplicative alternatives to the additive form (Luong's 'concat' score is essentially the additive form)."),
       ("Bahdanau uses a bidirectional recurrent encoder; Luong et al. used stacked unidirectional LSTMs",
        "Bahdanau's annotations h_{j} concatenate forward and backward states; Luong's model uses multi-layer unidirectional LSTMs."),
       ("Luong et al. introduced the global and local attention variants",
        "Global attends to all source positions, local to a window around an aligned position.")],
      [("Bahdanau uses hard attention, while Luong uses soft attention",
        "Both models use soft attention and are trained with ordinary backpropagation."),
       ("Only Luong's model normalizes the scores with a softmax",
        "Both normalize the alignment scores with a softmax over the source positions.")],
      "Both are soft attention mechanisms that compute a context vector as a weighted sum of encoder states; they differ in the decoder state used, the score function, the encoder architecture and in the global/local variants introduced by Luong et al.")

multi(A, "Which of the following are valid ways to score the compatibility of a decoder state s with an encoder state h?",
      [("The dot product s^{T}h",
        "The simplest, parameter-free scoring function (Luong 'dot')."),
       ("A bilinear form s^{T}Wh with a learned matrix W",
        "Luong's 'general' score."),
       ("An additive score v^{T}tanh(W_{1}s + W_{2}h)",
        "Bahdanau-style scoring by a one-hidden-layer network; Luong's 'concat' score is of this type as well."),
       ("A scaled dot product s^{T}h/√d",
        "The scoring function of the Transformer: a dot product with variance-stabilizing scaling.")],
      [("The vector sum s + h",
        "A score has to be a scalar; the sum of two vectors is a vector and measures nothing about compatibility."),
       ("softmax(h), the softmax of the encoder state alone",
        "That does not involve the decoder state, so it cannot express a match between the two.")],
      "Any function that maps a (query, key) pair to a scalar can serve as a score; the softmax over the scores of all keys then gives the attention weights. The common choices are dot, general (bilinear), additive/concat and scaled dot product.")

multi(A, "Which statements about multi-head attention are correct?",
      [("Each head has its own learned projections W_{i}^{Q}, W_{i}^{K} and W_{i}^{V}",
        "That is what allows different heads to look at different representation subspaces."),
       ("The heads are computed in parallel, their outputs are concatenated and then projected with W^{O}",
        "The concatenation has dimension h·d_{v}, and W^{O} maps it back to d_{model}."),
       ("With d_{k} = d_{model}/h the total cost is similar to single-head attention of full dimensionality",
        "h heads of width d_{model}/h cost about as much as one head of width d_{model}.")],
      [("All heads are forced to produce the same attention pattern, otherwise the concatenation is not defined",
        "Heads are independent and typically learn different patterns; the concatenation is defined for any patterns."),
       ("Using more heads raises the complexity in the sequence length from O(n^{2}) to O(n^{3})",
        "Each head costs O(n^{2}·d_{k}) with d_{k} = d_{model}/h, so the total stays O(n^{2}·d_{model}).")],
      "Multi-head attention gives the model several attention 'views' of the same sequence at roughly the cost of one full-width head; the parameter count is also unchanged by h (about 4·d_{model}^{2}).")

multi(A, "Which statements about self-attention and cross-attention are correct?",
      [("In self-attention, queries, keys and values are all computed from the same sequence",
        "Each position of the sequence attends to all positions of that same sequence."),
       ("In encoder–decoder attention the queries come from the decoder, while the keys and values come from the encoder",
        "Every decoder position can read from all encoder positions."),
       ("Decoder self-attention needs a causal mask, but encoder self-attention and cross-attention do not",
        "The decoder must not see future target tokens; the complete source sentence is available to the encoder and to the cross-attention.")],
      [("In cross-attention the number of queries must equal the number of keys",
        "The number of queries is the target length and the number of keys is the source length; the two can differ (the score matrix is n_{q}×n_{k})."),
       ("Self-attention can only be used in the encoder, never in the decoder",
        "The Transformer decoder has a masked self-attention sub-layer in every layer.")],
      "Self-attention: Q, K, V from one sequence (encoder: unmasked; decoder: causally masked). Cross-attention: Q from the decoder, K and V from the encoder output, no causal mask.")

multi(A, "Which of the following are true about global and local attention (Luong et al., 2015)?",
      [("Global attention considers all source positions at every target step",
        "The context vector is a weighted average over the whole source sentence."),
       ("Local attention considers only a window of source positions centred at an aligned position p_{t}",
        "Weights are computed inside the window [p_{t}−D, p_{t}+D]."),
       ("Local attention bounds the number of scored positions, which reduces the cost for long source sentences",
        "The cost per target step is proportional to the window size instead of the source length.")],
      [("Local attention always gives better translations than global attention",
        "Local attention can help, but the quality depends on the alignment model and on the data; it is not better in every setting."),
       ("Global attention picks a single source word by sampling, so that it needs a REINFORCE-style estimator",
        "Global attention is soft and differentiable; sampling one position is hard attention.")],
      "Global attention is the standard soft attention over the whole source. Local attention keeps the model differentiable but restricts the candidates to a window, which is cheaper for long inputs.")

multi(A, "Which statements about the paper 'Attention Is All You Need' are correct?",
      [("It proposes the Transformer, an encoder–decoder model whose layers are built from attention and position-wise feed-forward sub-layers (plus residual connections and layer normalization)",
        "Attention is the only mechanism that mixes information between positions; the feed-forward layers act on each position separately."),
       ("It dispenses with recurrence and convolutions entirely",
        "That is the claim of the title and abstract: no RNN and no CNN layers are used."),
       ("It uses multi-head attention in three ways: encoder self-attention, masked decoder self-attention and encoder–decoder attention",
        "These are the three attention sub-layer types of the model.")],
      [("Its main scoring function is additive attention with a learned vector v",
        "The Transformer uses scaled dot-product attention."),
       ("It keeps an LSTM in the decoder to generate the output tokens autoregressively",
        "The decoder is a stack of attention and feed-forward layers; there is no recurrent unit.")],
      "The Transformer replaces recurrence by (multi-head, scaled dot-product) self-attention and encoder–decoder attention, and uses positional encodings to inject word order.")

multi(A, "Which of the following are computed on the fly for every input in an additive attention layer, rather than being learned parameters?",
      [("The alignment scores e_{t,i}",
        "They are functions of the decoder state (s_{t−1} in Bahdanau attention) and the encoder states."),
       ("The attention weights α_{t,i}",
        "They are the softmax of the scores and therefore differ for every sentence and every decoder step."),
       ("The context vector c_{t}",
        "It is the weighted sum of the encoder states under the current weights.")],
      [("The matrices W_{s} and W_{h}",
        "They are trainable parameters of the scoring network, shared by all inputs and all time steps."),
       ("The scoring vector v",
        "It is a trainable parameter as well; it is fixed (shared) after training."),
       ("The bias terms of the scoring network",
        "Biases are trainable parameters, not input-dependent quantities.")],
      "Learned parameters (W_{s}, W_{h}, v, biases) are fixed after training; the scores, weights and context vectors are activations that are recomputed for every input and decoder step.")

# END
