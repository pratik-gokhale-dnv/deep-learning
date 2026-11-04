from qdsl import tf, single, multi, num, T

R = "rnn"

# ---------------------------------------------------------------- update rules
tf(R, "In a vanilla RNN, the same weight matrices (W_{xh}, W_{hh}, W_{hy}) are reused at every time step.", True,
   """Parameter sharing across time is what lets one set of weights process sequences of any length: the unrolled network is just T copies of the same cell.

Consequences: (1) the number of parameters does not depend on the sequence length, and (2) the gradient of a shared weight is the **sum** of its contributions from every time step.""")

single(R, "Which equation is the standard (Elman) RNN hidden-state update for input x_{t}?",
      ("h_{t} = tanh(W_{hh} h_{t−1} + W_{xh} x_{t} + b_{h})",
       "The previous state and the current input are combined linearly and then squashed by one nonlinearity — the Elman recurrence."),
      [("h_{t} = W_{hh} h_{t−1} + tanh(W_{xh} x_{t} + b_{h})",
        "The recurrent term bypasses the nonlinearity, giving a linear recurrence plus a squashed input — not the Elman RNN."),
       ("h_{t} = tanh(W_{hh} h_{t+1} + W_{xh} x_{t} + b_{h})",
        "It depends on the future state h_{t+1}; a causal RNN may only use h_{t−1}."),
       ("h_{t} = softmax(W_{hy} h_{t−1})",
        "That is an output distribution computed from a state, not a state update; it never mixes in the current input x_{t}.")],
      """The vanilla RNN update is `h_t = tanh(W_hh·h_(t−1) + W_xh·x_t + b_h)`, with output `y_t = W_hy·h_t + b_y` (followed by a softmax for classification or language modeling).""")

multi(R, "Which of the following are genuine limitations of a vanilla (Elman) RNN? Select all that apply.",
      [("Gradients can vanish when back-propagated through many time steps",
        "Repeatedly multiplying by Jacobians of norm < 1 shrinks the gradient exponentially, so distant inputs barely influence learning."),
       ("Gradients can explode, making training unstable",
        "If the Jacobians' norm is > 1 the gradient grows exponentially with the number of steps (huge updates/NaNs)."),
       ("Time steps must be processed one after another, which limits parallelism",
        "h_{t} depends on h_{t−1}, so the forward pass over n tokens needs O(n) sequential operations.")],
      [("It cannot process sequences of different lengths",
        "It can: the same cell is simply unrolled for as many steps as the input has."),
       ("The number of parameters grows linearly with sequence length",
        "Weights are shared across time, so the parameter count is independent of the sequence length.")],
      "Vanilla RNNs struggle with long-range credit assignment (vanishing gradients), can be unstable (exploding gradients) and are inherently sequential. They handle variable-length input fine and have a fixed parameter count.")

single(R, "What is the key difference between the update rules of a vanilla RNN and an LSTM?",
      ("A separate cell state is updated additively under gate control, c_{t} = f_{t} ⊙ c_{t−1} + i_{t} ⊙ g_{t}, rather than being rewritten at every step",
       "The additive, gated path lets information (and gradient) pass through many steps nearly unchanged."),
      [("The LSTM removes the recurrent connection, so every time step is processed independently of the others and can run in parallel",
        "It is still recurrent: h_{t−1} and c_{t−1} feed step t, so the steps remain sequential."),
       ("The LSTM replaces tanh by ReLU in the recurrence, which guarantees that gradients can never shrink along the sequence",
        "LSTMs use sigmoid gates and tanh; swapping in ReLU is not what prevents vanishing gradients."),
       ("The LSTM uses a different weight matrix at every time step, which avoids the weight sharing that causes vanishing gradients",
        "LSTM weights are shared across time exactly as in a vanilla RNN.")],
      """Vanilla RNN: `h_t = tanh(W·[h_(t−1), x_t] + b)` — the whole state is overwritten each step.

LSTM: gates `f_t, i_t, o_t = σ(W·[h_(t−1), x_t] + b)`, candidate `g_t = tanh(W_g·[h_(t−1), x_t] + b_g)`, then `c_t = f_t ⊙ c_(t−1) + i_t ⊙ g_t` and `h_t = o_t ⊙ tanh(c_t)`.""")

tf(R, "Because of its gating, an LSTM completely eliminates the exploding-gradient problem, so gradient clipping is unnecessary when training LSTMs.", False,
   "Gates mainly fix **vanishing** gradients along the cell-state path (∂c_{t}/∂c_{t−1} = f_{t}). Gradients can still explode through the other paths (the recurrent weights inside the gates and candidate), so gradient clipping remains standard practice for LSTMs.")

single(R, "Which LSTM component decides how much of the previous cell state c_{t−1} is retained?",
      ("The forget gate f_{t}",
       "c_{t} = f_{t} ⊙ c_{t−1} + …: f_{t} scales each component of the old cell state between 0 (erase) and 1 (keep)."),
      [("The input gate i_{t}",
        "It scales the new candidate g_{t} that is written into the cell, not the old contents."),
       ("The output gate o_{t}",
        "It controls how much of tanh(c_{t}) is exposed as the hidden state h_{t}; it does not change c_{t}."),
       ("The candidate activation g_{t}",
        "g_{t} = tanh(…) proposes new content; it is not a gate and does not scale the old state.")],
      "Forget gate → what to keep from the past; input gate → how much new content to write; output gate → how much of the cell to reveal as h_{t}.")

multi(R, "Which of these are gates in a standard LSTM cell? Select all that apply.",
      [("Input gate", "Controls how much of the new candidate content is written into the cell state."),
       ("Forget gate", "Controls how much of the previous cell state is kept."),
       ("Output gate", "Controls how much of tanh(c_{t}) becomes the hidden state h_{t}.")],
      [("Reset gate", "Belongs to the GRU: it controls how much of the previous state is used when computing the candidate."),
       ("Update gate z_{t}, which interpolates between h_{t−1} and a candidate state h̃_{t}",
        "A single gate that interpolates between the old state and the candidate is the GRU's update gate. The LSTM instead has separate input and forget gates (some courses call the LSTM's input gate its “update gate”, but that gate only controls writing; it does not interpolate with the old state).")],
      "A standard LSTM has three gates (input, forget, output) plus a tanh candidate. Reset and update gates (the interpolating gate z_{t}) are the GRU's two gates.")

tf(R, "If an LSTM's forget gate is ≈ 1 and its input gate is ≈ 0 at every step, the cell state is carried forward essentially unchanged.", True,
   "c_{t} = f_{t} ⊙ c_{t−1} + i_{t} ⊙ g_{t} ≈ c_{t−1} when f_{t} ≈ 1 and i_{t} ≈ 0. This is the 'constant error carousel': the derivative ∂c_{t}/∂c_{t−1} ≈ 1, so both information and gradient survive many steps.")

single(R, "In a standard LSTM, how is the hidden state h_{t} obtained from the cell state?",
      ("h_{t} = o_{t} ⊙ tanh(c_{t})",
       "The cell state is squashed by tanh and filtered element-wise by the output gate."),
      [("h_{t} = c_{t}",
        "The cell state is not exposed directly; it can grow beyond [−1, 1], whereas h_{t} is a gated, squashed read-out of it."),
       ("h_{t} = f_{t} ⊙ tanh(c_{t−1})",
        "That uses the forget gate and the previous cell state; the output gate and the current c_{t} define h_{t}."),
       ("h_{t} = i_{t} ⊙ g_{t}",
        "That product is the new content written into the cell, not the hidden state.")],
      "h_{t} = o_{t} ⊙ tanh(c_{t}) is what is passed to the next layer/step and to the output layer; c_{t} is the internal memory.")

multi(R, "Which statements about the LSTM cell state c_{t} are true? Select all that apply.",
      [("It is updated with element-wise operations and an addition, not by multiplying c_{t−1} with a full weight matrix",
        "c_{t} = f_{t} ⊙ c_{t−1} + i_{t} ⊙ g_{t}: only Hadamard products and a sum touch c_{t−1}."),
       ("The hidden state is a gated, squashed read-out of it: h_{t} = o_{t} ⊙ tanh(c_{t})",
        "The output gate decides which cell components are revealed."),
       ("It has the same dimensionality as the hidden state h_{t}",
        "Gates, candidate, c_{t} and h_{t} are all H-dimensional vectors (element-wise operations require matching sizes).")],
      [("It is always bounded to the interval (−1, 1) because of the tanh",
        "tanh is applied only when reading out h_{t}; c_{t} is a running sum and can exceed ±1."),
       ("It reaches the next time step only through the output gate",
        "c_{t} is passed directly to step t+1 (that is the point of the design); only h_{t} is gated by o_{t}.")],
      "The cell state is the LSTM's long-term memory: additive, element-wise updates; unbounded; the same size as h_{t}. Only the hidden state h_{t} is filtered by the output gate.")

single(R, "Why do LSTMs mitigate the vanishing-gradient problem?",
      ("Along the cell-state path ∂c_{t}/∂c_{t−1} = f_{t} (element-wise), so with f_{t} ≈ 1 the gradient flows back through many steps almost unchanged",
       "The additive cell update provides a short, nearly linear gradient highway."),
      [("The gates are hard binary 0/1 switches, so the gradient is preserved exactly regardless of the sequence length",
        "Gates are sigmoids with values in (0, 1); they are soft and learned, not binary."),
       ("A larger hidden state automatically increases the gradient magnitude at every time step",
        "Size does not change the repeated-product structure of a vanilla RNN; the additive cell path does."),
       ("LSTMs are trained with second-order optimizers whose curvature estimates rescale vanishing gradients",
        "LSTMs are trained with ordinary first-order methods (SGD, Adam); the benefit comes from the architecture.")],
      "In a vanilla RNN, ∂h_{t}/∂h_{t−1} = diag(1 − h_{t}²) W_{hh} is multiplied over many steps, so the product tends to vanish or explode. In an LSTM the cell path multiplies only by the forget gate values (ignoring indirect dependencies), which the network can learn to keep near 1.")

single(R, "Which technique is the standard remedy for **exploding** gradients in RNN training?",
      ("Gradient clipping: rescale the gradient whenever its norm exceeds a threshold",
       "It bounds the size of each update while keeping the gradient's direction."),
      [("Replacing tanh with the sigmoid activation in the recurrent transition",
        "σ'(x) ≤ 0.25, so gradients shrink faster (vanishing); it does not tame an oversized recurrent matrix."),
       ("Adding more teacher forcing so the decoder always sees ground-truth inputs",
        "Teacher forcing concerns what is fed as input; it does not control gradient magnitude."),
       ("Increasing the sequence length so that gradient contributions average out",
        "Longer unrolls multiply more Jacobians, making explosion (or vanishing) worse.")],
      "Clip by norm: if ‖g‖ > c then g ← c·g/‖g‖ (Pascanu et al., 2013). Clipping handles exploding gradients; gated cells (LSTM/GRU) address vanishing gradients.")

single(R, "What is the main practical consequence of vanishing gradients in a vanilla RNN?",
      ("The network struggles to learn dependencies between inputs that are many steps apart",
       "The error signal from a late loss term reaches early time steps with almost no magnitude, so early inputs cannot be credited or blamed."),
      [("The loss quickly becomes NaN because the activations overflow numerically",
        "That is the typical symptom of exploding, not vanishing, gradients."),
       ("The number of parameters grows linearly with the length of the training sequences",
        "Weights are shared across time; the count is fixed."),
       ("The network can no longer handle inputs of different lengths within a batch",
        "Variable length is unaffected — the issue is long-range credit assignment.")],
      "‖∂L_{T}/∂h_{k}‖ contains a product of T−k Jacobians; if each has norm < 1 it decays exponentially with T−k, so the network effectively only learns short-range dependencies.")

multi(R, "Which of the following help to control **exploding** gradients in RNNs? Select all that apply.",
      [("Clipping the gradient norm (rescale g whenever ‖g‖ exceeds a threshold)",
        "Caps the size of every update while keeping the gradient's direction (Pascanu et al., 2013)."),
       ("Clipping each gradient component to a fixed range [−c, c] (value clipping)",
        "The other common form of clipping (e.g. clip_grad_value_ in PyTorch): it bounds every component, at the cost of possibly changing the gradient's direction.")],
      [("Raising the learning rate", "A larger step makes an already exploded gradient even more destructive."),
       ("Unrolling over much longer sequences without clipping",
        "More multiplied Jacobians means more chances for the norm to blow up."),
       ("Initializing W_{hh} with large random entries so that signals travel far",
        "Large recurrent weights push the largest singular value above 1, which makes exploding gradients more likely, not less.")],
      "Exploding gradients are handled by clipping (by norm or by value). Gated cells and careful initialisation (orthogonal/identity W_{hh}) mainly target vanishing gradients, and truncated BPTT is a cost-saving approximation that does not by itself prevent a blow-up inside its K-step window.")

single(R, "What does truncated BPTT (back-propagation through time) do?",
      ("The hidden state is carried forward across segments, but gradients flow back through only the last K steps",
       "Truncation affects the backward pass only: cost per update is bounded by K, at the price of ignoring dependencies longer than K (a biased gradient)."),
      [("The hidden state is reset to zero every K steps, so that no information at all crosses segment boundaries",
        "Standard truncated BPTT keeps the state in the forward pass and only 'detaches' the gradient."),
       ("It computes the exact gradient of the full-sequence loss but only reports it for the first K time steps",
        "The gradient is deliberately cut after K steps; it is an approximation, not an exact partial gradient."),
       ("It freezes the recurrent weights and trains only the input and output weights over a window of K steps",
        "All weights are still trained; only the length of the gradient path is limited.")],
      "For a sequence of length 1000 and K = 50 you process 20 segments; each update back-propagates at most 50 steps, so memory and compute per update stay small, but long-range gradient signals beyond K steps are lost.")

tf(R, "A vanilla RNN processes all time steps of a sequence in parallel during training.", False,
   "The recurrence h_{t} = f(h_{t−1}, x_{t}) makes step t depend on step t−1, so the forward pass needs O(n) sequential steps (parallelism exists only across batch items). Removing this sequential bottleneck is one of the main motivations for the Transformer.")

single(R, "For which task is a bidirectional RNN a natural fit?",
      ("Part-of-speech tagging of a complete sentence",
       "The whole input is available, so each tag can use both left and right context by concatenating forward and backward states."),
      [("Generating a sentence one word at a time, where each next word is unknown until produced",
        "Future words do not exist yet during generation, so only a left-to-right model can be used."),
       ("Streaming prediction in which the later inputs have not arrived yet at decision time",
        "The backward pass needs the future inputs, which are unavailable online."),
       ("Autoregressive decoding of a translation, one target token after another",
        "A decoder generates left to right, so it cannot be bidirectional (although the encoder can).")],
      "Bidirectional RNNs need the entire input up front. They are used for tagging, classification and as **encoders**, but not for autoregressive generation.")

multi(R, "Which of the following help with **vanishing** gradients over long sequences? Select all that apply.",
      [("Gated cells such as the LSTM or GRU", "Their additive, gated state updates create a short path for gradients."),
       ("Skip connections across time steps (e.g. h_{t−k} feeds directly into h_{t})", "They give gradients a direct route across time that avoids a long product of Jacobians."),
       ("ReLU activations with an identity-initialised recurrent matrix",
        "With W_{hh} = I and ReLU (the IRNN idea) the state is copied across time at initialisation, so gradients do not decay.")],
      [("Using sigmoid activations in the hidden-to-hidden transition",
        "σ' ≤ 0.25 shrinks gradients even faster than tanh (tanh' ≤ 1)."),
       ("Increasing the batch size", "Batch size changes gradient noise, not how the gradient decays along time.")],
      "Architectural fixes (gates, skip connections across time) and initialisation (identity/orthogonal recurrent weights) mainly tackle vanishing gradients; clipping addresses explosion instead.")

multi(R, "Which statements correctly compare a GRU with an LSTM? Select all that apply.",
      [("A GRU has no separate cell state; it keeps a single hidden state h_{t}", "The cell and hidden states are merged."),
       ("A GRU uses an update gate z_{t} and a reset gate r_{t}", "One common form is h_{t} = (1 − z_{t}) ⊙ h_{t−1} + z_{t} ⊙ h̃_{t} (Cho et al. 2014, PyTorch and D2L put z_{t} on the old state instead, which is the same architecture); r_{t} gates h_{t−1} inside the candidate h̃_{t}."),
       ("For the same input and hidden sizes a GRU has fewer parameters", "It has 3 weight blocks (update, reset, candidate) versus 4 for the LSTM, about 25% fewer parameters.")],
      [("A GRU has an output gate that scales tanh(c_{t})", "GRUs have neither an output gate nor a cell state; the hidden state is exposed directly."),
       ("A GRU cannot learn long-range dependencies at all",
        "Its interpolation between h_{t−1} and the candidate also provides an additive path, and GRUs perform about as well as LSTMs on many tasks.")],
      "GRU = simplified gated RNN: two gates, no separate memory cell, fewer parameters, similar behaviour to the LSTM in practice.")

single(R, "Why do LSTM gates use a sigmoid activation?",
      ("Its output lies in (0, 1), so an element-wise product acts as a smooth 'fraction to let through'",
       "0 blocks a component, 1 passes it fully, and values in between scale it."),
      [("Its output lies in (−1, 1), so a gate can flip the sign of the cell state when needed",
        "That is the range of tanh, which is used for the candidate and the read-out, not for gates."),
       ("Its derivative equals 1 near the origin, so gradients passing through a gate never shrink at all",
        "σ'(x) ≤ 0.25; gates do not rely on a unit derivative."),
       ("It is the only differentiable squashing function, so no other activation could serve as a gate",
        "Many functions are differentiable; sigmoid is chosen for its (0, 1) range.")],
      "Gates need to act like soft switches → sigmoid. The candidate g_{t} and the read-out use tanh so values can be positive or negative and stay bounded.")

multi(R, "Which task → RNN-configuration pairings are correct? Select all that apply.",
      [("Sentiment classification of a review → many-to-one", "Read the whole text, then classify from the final hidden state."),
       ("Image captioning → one-to-many", "One image vector in, a word sequence out."),
       ("Machine translation → many-to-many with unaligned lengths (encoder–decoder)", "Source and target lengths differ, so an encoder reads the input and a decoder generates the output.")],
      [("Per-token part-of-speech tagging → many-to-one", "A tag is needed for every token, i.e. an aligned many-to-many setup."),
       ("Training or scoring a next-word language model on a given text → one-to-many",
        "Not one-to-many: at every position the input is the current word and the target is the next word, i.e. aligned many-to-many. Only free-running generation from a single seed has a one-to-many shape.")],
      "RNN input/output shapes: many-to-one (classification), one-to-many (captioning; free-running generation from a single seed), aligned many-to-many (tagging, LM training/scoring), unaligned many-to-many (seq2seq).")

tf(R, "The hidden state h_{t} of a vanilla RNN can be viewed as a lossy, fixed-size summary of the inputs x_{1}, …, x_{t}.", True,
   "h_{t} has a fixed dimension H no matter how long the prefix is, so it must compress the history, and the compression is lossy. That limited capacity is also why a single final state is a bottleneck when it must summarise a whole source sentence in seq2seq models.")

tf(R, "Gradient clipping fixes vanishing gradients by scaling up gradients whose norm is very small.", False,
   "Clipping only rescales a gradient whose norm **exceeds** a threshold (g ← c·g/‖g‖); it never amplifies small gradients, so it targets **exploding** gradients. Vanishing gradients are tackled architecturally (LSTM/GRU gates, skip connections), by initialisation of the recurrent weights, or by the choice of activation.")

multi(R, "Which statements correctly compare an LSTM layer with a vanilla RNN layer that has the same input and hidden sizes? Select all that apply.",
      [("The LSTM has four times as many parameters, because it learns four weight blocks (three gates and the candidate) instead of one",
        "A vanilla layer has H(H + D) + H weights; the LSTM has 4 × that."),
       ("The LSTM passes two vectors from one step to the next (h_{t} and c_{t}), the vanilla RNN only one (h_{t})",
        "The cell state c_{t} is the extra, gated long-term memory."),
       ("Both must process the sequence one step after another, so neither can be parallelised across time",
        "Step t needs h_{t−1} (and c_{t−1}); gating does not remove this sequential dependency.")],
      [("Only the vanilla RNN's hidden state is bounded; the LSTM's hidden state can grow without limit",
        "Both hidden states are bounded: h_{t} = o_{t} ⊙ tanh(c_{t}) lies in (−1, 1). It is the LSTM's **cell state** c_{t} that is unbounded."),
       ("The LSTM cannot be trained with back-propagation through time because of its multiplicative gates",
        "Gates are differentiable; LSTMs are trained with ordinary (truncated) BPTT.")],
      "An LSTM layer is a vanilla RNN with extra machinery: four weight blocks instead of one, a second state vector (the cell), and sigmoid gates. It remains sequential and is trained with BPTT; the cell's additive update is what improves gradient flow.")

# ---------------------------------------------------------------- calc
H, D = 256, 128
per_gate = H * (H + D) + H
total = 4 * per_gate
single(R, T("An LSTM layer has input size D = ${D} and hidden size H = ${H}, with **one bias vector** per gate and for the candidate. How many trainable parameters does the layer have?", D=D, H=H),
       (f"{total:,}", T("Correct: 4 × (H(H + D) + H) = 4 × (${HD:,} + ${H}) = ${total:,}.", HD=H * (H + D), H=H, total=total)),
       [(f"{4 * H * (H + D):,}", "Forgot the bias vectors: this is only the weight matrices 4·H·(H + D)."),
        (f"{3 * per_gate:,}", "Counted only 3 of the 4 blocks (that is the GRU count); an LSTM has input, forget, output and candidate blocks."),
        (f"{4 * (H * H + H):,}", "Ignored the input weights W_{x·} (H × D per block) and counted only the recurrent part.")],
       T("""An LSTM has four blocks (input gate, forget gate, output gate, candidate), each with its own weights on [h_(t−1), x_t] and a bias:

1. Per block: H·(H + D) + H = ${H} × ${HD} + ${H} = ${pg:,}.
2. Four blocks: 4 × ${pg:,} = **${total:,}**.""", H=H, HD=H + D, pg=per_gate, total=total),
       calc=True)
