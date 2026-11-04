from qdsl import tf, single, multi, num, T

B = "bptt"

LIN = ("A scalar linear RNN is unrolled for 3 steps: h_{t} = w·h_{t−1} + u·x_{t}, with h_{0} = 0, w = 0.5, u = 1 and inputs "
       "(x_{1}, x_{2}, x_{3}) = (1, 2, 3). Only the final state is scored: L = ½ (h_{3} − y*)^{2} with target y* = 4.")

LIN_FWD = """Forward pass (always do this first and keep the activations):

- h_{1} = 0.5·0 + 1·1 = 1
- h_{2} = 0.5·1 + 1·2 = 2.5
- h_{3} = 0.5·2.5 + 1·3 = 4.25"""

# ------------------------------------------------------------------ 1. forward value
num(B, LIN + "\n\nCompute the forward-pass value h_{3}.", 4.25, 
    LIN_FWD + "\n\nBackprop through time always starts with this forward pass, because every local derivative (h_{t−1}, tanh', …) is evaluated at the stored activations.",
    tol=0.005, hint="decimal number", disp="4.25")

# ------------------------------------------------------------------ 2. error signal at h_1
num(B, LIN + "\n\nUsing BPTT, compute ∂L/∂h_{1}, the error signal that reaches the first hidden state.", 0.0625,
    LIN_FWD + """

Backward pass (from the loss to the first step):

1. ∂L/∂h_{3} = h_{3} − y* = 4.25 − 4 = 0.25
2. h_{3} = w·h_{2} + u·x_{3}, so ∂h_{3}/∂h_{2} = w and ∂L/∂h_{2} = w · 0.25 = 0.125
3. In the same way ∂L/∂h_{1} = w · 0.125 = **0.0625**

The error signal is multiplied by the Jacobian (here simply w) once per step it travels back, so ∂L/∂h_{1} = w^{2} (h_{3} − y*). With |w| < 1 this decays exponentially in the number of steps: the vanishing-gradient effect.""",
    tol=0.003, hint="decimal number", disp="0.0625")

# ------------------------------------------------------------------ 3. dL/dw (MCQ)
single(B, LIN + "\n\nWhat is ∂L/∂w, the gradient with respect to the **shared** recurrent weight?",
       ("0.75", "Correct: Σ_{t} δ_{t}·h_{t−1} = 0.0625·0 + 0.125·1 + 0.25·2.5 = 0.75."),
       [("0.625", "This keeps only the last step, δ_{3}·h_{2} = 0.25 × 2.5. A shared weight gets a contribution from every time step, not just the last."),
        ("0.875", "This uses δ_{t} = 0.25 at every step (0.25 × (0 + 1 + 2.5)); it forgets that the error signal is multiplied by w each time it moves back one step."),
        ("1.4375", "This pairs δ_{t} with h_{t} instead of h_{t−1}: 0.0625·1 + 0.125·2.5 + 0.25·4.25. The weight w multiplies the **previous** state.")],
       LIN_FWD + """

Error signals δ_{t} = ∂L/∂h_{t}: δ_{3} = 0.25, δ_{2} = w·δ_{3} = 0.125, δ_{1} = w·δ_{2} = 0.0625.

Because h_{t} = w·h_{t−1} + u·x_{t}, the weight w enters every step through the factor h_{t−1}. Summing over time:

`∂L/∂w = Σ_t δ_t · h_(t−1) = 0.0625·0 + 0.125·1 + 0.25·2.5 = 0.75`

Cross-check with forward sensitivities: dh_{t}/dw = h_{t−1} + w·dh_{t−1}/dw gives (0, 1, 3) for t = 1, 2, 3, so ∂L/∂w = (h_{3} − y*)·3 = 0.25 × 3 = 0.75.""",
       calc=True)

# ------------------------------------------------------------------ 4. dL/du
num(B, LIN + "\n\nCompute ∂L/∂u, the gradient with respect to the input weight.", 1.0625,
    LIN_FWD + """

Error signals: δ_{3} = 0.25, δ_{2} = 0.125, δ_{1} = 0.0625.

The input weight u multiplies x_{t} at every step, so its gradient is the sum over time of δ_{t}·x_{t}:

`∂L/∂u = Σ_t δ_t · x_t = 0.0625·1 + 0.125·2 + 0.25·3 = 0.0625 + 0.25 + 0.75 = 1.0625`

Compare ∂L/∂w, which pairs δ_{t} with h_{t−1} (not with x_{t}).""",
    tol=0.003, hint="decimal number", disp="1.0625")

# ------------------------------------------------------------------ 5. per-step losses
single(B, "A scalar linear RNN (h_{t} = w·h_{t−1} + u·x_{t}, h_{0} = 0, w = 0.5, u = 1, inputs (1, 2, 3), so the states are h_{1}, h_{2}, h_{3} = 1, 2.5, 4.25) is scored at **every** step: L = ½ Σ_{t=1}^{3} (h_{t} − y*_{t})^{2} with targets y*_{1}, y*_{2}, y*_{3} = 0.5, 2, 4. What is ∂L/∂w?",
       ("1.25", "Correct: δ_{3} = 0.25, δ_{2} = 0.5 + 0.5·0.25 = 0.625, δ_{1} = 0.5 + 0.5·0.625 = 0.8125, so ∂L/∂w = 0.8125·0 + 0.625·1 + 0.25·2.5 = 1.25."),
       [("1.125", "This uses only the local errors e_{t} (0.5, 0.5, 0.25) and forgets the term w·δ_{t+1} that arrives from later steps."),
        ("0.75", "That is the gradient when only the last step is scored (the single-final-loss version of this RNN); here the losses at steps 1 and 2 contribute as well."),
        ("3.4375", "This pairs δ_{t} with h_{t} instead of h_{t−1}; w multiplies the **previous** state.")],
       """With a loss at every step, the error signal at step t has two sources: the local error and the signal coming back from step t+1:

`δ_t = (h_t − y*_t) + w · δ_(t+1)`

1. Local errors: e_{1} = 1 − 0.5 = 0.5, e_{2} = 2.5 − 2 = 0.5, e_{3} = 4.25 − 4 = 0.25.
2. δ_{3} = 0.25; δ_{2} = 0.5 + 0.5·0.25 = 0.625; δ_{1} = 0.5 + 0.5·0.625 = 0.8125.
3. `∂L/∂w = Σ_t δ_t·h_(t−1) = 0.8125·0 + 0.625·1 + 0.25·2.5 = 1.25`

Forgetting the recursive term (using δ_{t} = e_{t}) is the classic mistake: it treats the steps as independent.""",
       calc=True)

# ------------------------------------------------------------------ 6. tanh, 2 steps
num(B, "A scalar tanh RNN is unrolled for 2 steps: h_{t} = tanh(w·h_{t−1} + u·x_{t}) with h_{0} = 0, w = 0.5, u = 1 and inputs (x_{1}, x_{2}) = (1, 1). The loss is L = ½ (h_{2} − y*)^{2} with y* = 1. Compute ∂L/∂u using BPTT (3 decimals are enough).", -0.0322,
    """Forward pass (pre-activations a_{t}):

- a_{1} = 0.5·0 + 1·1 = 1, so h_{1} = tanh(1) = 0.7616
- a_{2} = 0.5·0.7616 + 1 = 1.3808, so h_{2} = tanh(1.3808) = 0.8811

Backward pass, using the tanh derivative 1 − h^{2} (h = the activation at that step):

1. ∂L/∂h_{2} = h_{2} − y* = −0.1189
2. ∂L/∂a_{2} = ∂L/∂h_{2} · (1 − h_{2}^{2}) = −0.1189 × 0.2236 = −0.02658
3. Pass the signal back: ∂L/∂a_{1} = ∂L/∂a_{2} · w · (1 − h_{1}^{2}) = −0.02658 × 0.5 × 0.4200 = −0.00558
4. u multiplies x_{t} at every step: ∂L/∂u = ∂L/∂a_{2}·x_{2} + ∂L/∂a_{1}·x_{1} = −0.02658 − 0.00558 = **−0.0322**.

Stopping after step 2 (−0.0266) forgets the term that travels back through w and the tanh slope at step 1. (A finite-difference check gives −0.03216. For w, the step-1 term would vanish because h_{0} = 0, leaving ∂L/∂w = ∂L/∂a_{2}·h_{1} = −0.0202.)""",
    tol=0.0005, hint="decimal number (negative)", disp="≈ −0.0322")

# ------------------------------------------------------------------ 7. output layer
num(B, "A scalar linear RNN with an output weight: h_{t} = w·h_{t−1} + u·x_{t} and ŷ_{t} = v·h_{t}, with h_{0} = 0, w = 0.5, u = 1, v = 2. Inputs (x_{1}, x_{2}) = (1, 2) and targets (y_{1}, y_{2}) = (1, 4). The loss is L = ½ Σ_{t=1}^{2} (ŷ_{t} − y_{t})^{2}. Compute ∂L/∂u with BPTT.", 7,
    """Forward pass: h_{1} = 1, h_{2} = 0.5·1 + 2 = 2.5; outputs ŷ_{1} = 2, ŷ_{2} = 5; errors e_{1} = ŷ_{1} − y_{1} = 1, e_{2} = ŷ_{2} − y_{2} = 1.

Backward pass over the hidden states (each output contributes v·e_{t}):

1. ∂L/∂h_{2} = v·e_{2} = 2
2. ∂L/∂h_{1} = v·e_{1} + w·∂L/∂h_{2} = 2 + 0.5·2 = 3

Since u multiplies x_{t} at every step:

`∂L/∂u = Σ_t (∂L/∂h_t)·x_t = 3·1 + 2·2 = 7`

For reference, ∂L/∂w = 3·h_{0} + 2·h_{1} = 2 and ∂L/∂v = Σ e_{t}·h_{t} = 1·1 + 1·2.5 = 3.5.""",
    tol=0.01, hint="integer or decimal", disp="7")

# ------------------------------------------------------------------ 8. parameter count
single(B, "A vanilla RNN has input size 10, hidden size 20 and output size 5, with bias vectors b_{h} (hidden) and b_{y} (output). It is unrolled over 8 time steps for BPTT. How many trainable parameters does the model have?",
       ("725", "W_{xh}: 20×10 = 200; W_{hh}: 20×20 = 400; b_{h}: 20; W_{hy}: 5×20 = 100; b_{y}: 5; total 725."),
       [("5,800", "This counts a separate copy of the weights for each of the 8 unrolled steps (8 × 725). The weights are shared across time, so the count does not depend on the unroll length."),
        ("700", "This omits the two bias vectors (20 + 5 = 25)."),
        ("745", "This counts the initial hidden state h_{0} (20 values) as trainable; it is normally fixed to zeros and is not a parameter.")],
       """Parameters are shared over time, so the unrolled length (8) does not matter:

1. W_{xh} ∈ ℝ^{20×10} → 200
2. W_{hh} ∈ ℝ^{20×20} → 400
3. b_{h} ∈ ℝ^{20} → 20
4. W_{hy} ∈ ℝ^{5×20} → 100
5. b_{y} ∈ ℝ^{5} → 5

Total = **725**. Unrolling multiplies the **activations** that must be stored (memory), not the parameters.""",
       calc=True)

# ------------------------------------------------------------------ 9. vanishing bound
num(B, "In a long unrolled RNN every Jacobian satisfies ‖∂h_{t}/∂h_{t−1}‖ ≤ 0.9. Give the upper bound on ‖∂h_{50}/∂h_{1}‖ (3–4 significant digits).", 0.005726,
    """By the chain rule, ∂h_{50}/∂h_{1} = Π_{t=2}^{50} ∂h_{t}/∂h_{t−1}, a product of **49** Jacobians (not 50).

Using submultiplicativity of the norm: ‖∂h_{50}/∂h_{1}‖ ≤ 0.9^{49} ≈ **0.005726** (≈ 0.0057).

The signal from step 50 reaches step 1 at under 1% of its size: contributions from distant steps are drowned out by nearby ones (vanishing gradients). With a bound of 1.2 instead, the same product would be 1.2^{49} ≈ 7,600 (exploding gradients).""",
    tol=0.0002, hint="a small positive decimal", disp="≈ 0.0057")

# ------------------------------------------------------------------ 10. exploding
single(B, "While back-propagating through a long sequence, the gradient norm is multiplied by a factor of 1.2 at each of 30 successive steps (30 multiplications in total). By roughly what factor is the gradient amplified?",
       ("≈ 237", "1.2^{30} ≈ 237: a repeated multiplication compounds exponentially."),
       [("≈ 36", "This treats the growth as linear (1.2 × 30); repeated multiplication is exponential, not additive."),
        ("≈ 59", "This computes 30^{1.2}, swapping base and exponent."),
        ("≈ 198", "This is 1.2^{29}: one multiplication too few.")],
       """Each step multiplies the norm by 1.2, so after 30 steps the amplification is 1.2^{30} ≈ **237**. A factor only 20% above 1 already produces a gradient hundreds of times too large, which is why exploding gradients are tamed with **gradient clipping**.

Conversely, 0.8^{30} ≈ 0.0012: a factor slightly below 1 makes the gradient vanish.""",
       calc=True)

# ------------------------------------------------------------------ 11. Jacobian
single(B, "For the RNN h_{t} = tanh(W_{hh} h_{t−1} + W_{xh} x_{t} + b) with hidden size H, which expression is the Jacobian ∂h_{t}/∂h_{t−1} that BPTT multiplies together across time?",
       ("diag(1 − h_{t}^{2}) · W_{hh}, an H × H matrix",
        "The derivative of tanh is evaluated at the current pre-activation (1 − h_{t}^{2}), and the inner derivative with respect to h_{t−1} is W_{hh}."),
       [("W_{hh} alone, an H × H matrix",
         "This drops the tanh derivative; the diagonal factor 1 − h_{t}^{2} ≤ 1 is exactly what makes saturated units shrink gradients."),
        ("diag(1 − h_{t−1}^{2}) · W_{hh}, an H × H matrix",
         "The tanh derivative belongs to the activation computed at step t, so it uses h_{t}, not h_{t−1}."),
        ("diag(1 − h_{t}^{2}) · W_{xh}, with the shape of W_{xh}",
         "W_{xh} multiplies the input x_{t}, which does not depend on h_{t−1}; the path from h_{t−1} goes through W_{hh}.")],
       """Let a_{t} = W_{hh} h_{t−1} + W_{xh} x_{t} + b and h_{t} = tanh(a_{t}). Then

`∂h_t/∂h_(t−1) = diag(tanh'(a_t)) · W_hh = diag(1 − h_t²) · W_hh`

The gradient from step T to step k is the product of T − k such H×H Jacobians. Its norm shrinks if the largest singular value of W_{hh} times the tanh slope stays below 1 (vanishing) and can blow up if it exceeds 1 (exploding).""", calc=True)

# ------------------------------------------------------------------ 12. multi
multi(B, "Which statements about back-propagation through time (BPTT) for a simple RNN are correct? Select all that apply.",
      [("The gradient of a shared weight matrix is the sum of its contributions from every time step",
        "The same W_{hh} is used at each step, so ∂L/∂W_{hh} = Σ_{t} (∂L/∂a_{t})·h_{t−1}^{T}."),
       ("The error signal at step t is the local loss term at t plus the Jacobian-transformed signal coming back from step t+1",
        "δ_{t} = ∂L_{t}/∂h_{t} + (∂h_{t+1}/∂h_{t})^{T} δ_{t+1}: a backward recursion from the last step to the first."),
       ("Memory use grows with the unrolled length because the activations of every step must be stored for the backward pass",
        "Local derivatives such as h_{t−1} and 1 − h_{t}^{2} are needed later, so O(T·H) activations are kept; truncated BPTT bounds this.")],
      [("Gradients are computed in the forward direction, from step 1 to step T",
        "Gradients flow backward from the loss at the end (step T) to the start; the activations are computed forward."),
       ("Every time step has its own copy of the weights, updated independently of the others",
        "The weights are shared; the unrolled copies are tied, and their gradients are summed into a single update.")],
      """BPTT = ordinary back-propagation on the network unrolled over time:

1. Forward: compute and store h_{1}, …, h_{T} (and the outputs/loss).
2. Backward: start at the last step, and propagate δ_{t} = local term + J_{t+1}^{T} δ_{t+1}.
3. Accumulate the gradient of each shared weight over all steps, then apply a single update.""")
