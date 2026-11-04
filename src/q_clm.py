from qdsl import tf, single, multi, num, T

C = "clm"

# ------------------------------------------------------------------ true / false
tf(C, "Training a conditional language model by maximum likelihood amounts to minimising the token-level cross-entropy summed over the target positions, because log p(y | x) = ∑_{t} log p(y_{t} | y_{<t}, x).", True,
   """The chain rule turns the product of conditionals into a sum of logs, so maximising the likelihood of the reference outputs is the same as minimising the summed per-token negative log-likelihood, i.e. the cross-entropy between the one-hot reference token and the model's softmax at every position.

An unconditional LM is the special case in which the input x is dropped from every factor.""")

tf(C, "Under teacher forcing, if the model's most probable token at step t is wrong, that wrong token is what the decoder receives as its input at step t+1.", False,
   """Teacher forcing feeds the **reference** token y_{t} into step t+1 no matter what the model predicted; the prediction only enters the loss at step t.

Feeding back the model's own (possibly wrong) token is what happens at generation time, and during training only under student forcing (free-running training).""")

tf(C, "A Transformer decoder trained with teacher forcing receives the whole shifted reference at once, so it needs a causal mask to stop position t from attending to reference tokens that come later.", True,
   """With the shifted reference as input, position t may use `<s>, y_1, …, y_(t−1)` and must predict y_{t}. Without the mask, position t could look at the token sitting at a later position and simply copy the answer, so the loss could be driven to zero without learning to predict anything.

The mask is what makes one-pass, all-positions-in-parallel training both fast and valid.""")

tf(C, "Exposure bias is a weakness of recurrent decoders only: Transformer decoders, which are trained on all target positions in parallel, are not affected by it.", False,
   """Exposure bias comes from the mismatch between the inputs seen in training (gold prefixes) and in generation (the model's own outputs). Any autoregressive decoder trained with teacher forcing has it, whether recurrent or Transformer.

Parallel training changes how fast the model learns, not what it is conditioned on.""")

# ------------------------------------------------------------------ single answer
single(C, "A German decoder is trained with teacher forcing on the reference sentence `Das Haus ist klein`. Which choice lists the decoder inputs and the training targets correctly?",
       ("Inputs: `<s> Das Haus ist klein`; targets: `Das Haus ist klein </s>`",
        "Each position receives the previous reference token (starting from `<s>`) and must predict the current one; the final target `</s>` teaches the model when to stop."),
       [("Inputs: `Das Haus ist klein </s>`; targets: `<s> Das Haus ist klein`, i.e. the targets are the inputs delayed by one position",
         "The shift goes the wrong way: the model would be asked to predict the previous token (even `<s>` from `Das`) instead of the next one."),
        ("Inputs: `Das Haus ist klein`; targets: `Das Haus ist klein`",
         "With no shift the decoder sees the very token it must output, so it can learn to copy its input and never learns to predict the next word."),
        ("Inputs: the English source tokens `The house is small`; targets: `Das Haus ist klein </s>`",
         "The source sentence is read by the encoder. The decoder's inputs are the shifted target tokens; source information reaches it through cross-attention (or the context vector).")],
       """Written as input → target pairs, the five training positions are:

`<s>` → Das, Das → Haus, Haus → ist, ist → klein, klein → `</s>`.

All five are scored with cross-entropy against the target; with a causal mask a Transformer decoder computes them in one parallel pass.""")

single(C, "A translation model reaches a very low dev-set perplexity (computed by feeding it the reference prefix), yet at generation time a single early mistake often derails the rest of the sentence. Which phenomenon best explains this?",
       ("Exposure bias: the model was only ever trained and scored on gold prefixes, never on its own output",
        "Errors compound because the next-token distribution was never trained on inputs that contain the model's own mistakes."),
       [("Vanishing gradients: the decoder never managed to learn long-range dependencies during training",
         "A low teacher-forced perplexity shows the model predicts the reference well from correct prefixes. The failure appears only when its own errors are fed back, which is not a gradient-flow problem."),
        ("Overfitting: the model has memorised the training pairs and does not generalise to new sentences",
         "The perplexity was measured on held-out dev data; memorisation would have made it high, not very low."),
        ("Teacher forcing makes the decoder ignore the source sentence, so it drifts as soon as it starts generating text",
         "The decoder still conditions on the source via the encoder. Reaching a very low perplexity for p(y | x) would be impossible if x were ignored; the problem is mismatched inputs, not ignored ones.")],
       """Dev-set perplexity is normally computed with teacher forcing: the model is scored on the reference tokens given the **reference** prefix. It therefore cannot reveal what happens once the model conditions on its own, possibly wrong, earlier outputs.

This train/test input mismatch is called exposure bias.""")

single(C, "In this question, **student forcing** (free-running training) means that during training the decoder is fed its own previous prediction instead of the reference token, while the loss is still measured against the reference tokens. Compared with teacher forcing, what is the main trade-off?",
       ("Its inputs match generation time, but training is less stable at first and (for Transformer decoders) no longer one-pass parallel",
        "Matching the test-time input distribution reduces exposure bias. The price: each step needs the previous step's output, so a Transformer decoder loses its one-pass parallelism over target positions (an RNN decoder is sequential anyway), and a barely trained model feeds itself mostly wrong tokens, giving a noisy learning signal."),
       [("It needs no reference outputs, so the model can be trained from source-side text alone",
         "The loss is still computed against the reference tokens, so parallel (source, target) data are still required."),
        ("It allows every target position to be computed in parallel, because predictions are available without waiting for reference tokens",
         "The opposite: the input at step t is the prediction made at step t−1, so the steps must run one after another. One-pass parallelism over target positions (in Transformer decoders) is a benefit of teacher forcing."),
        ("It removes exposure bias at no cost, which is why it is the default way of training modern sequence models",
         "Teacher forcing is the default precisely because free-running training is less stable and, for Transformer decoders, slower; mixing the two (scheduled sampling) is one known compromise.")],
       """Teacher forcing: stable, fast to converge, parallel over target positions for Transformer decoders, but trained on inputs it will not see at test time.

Student forcing: matched inputs, but (for Transformers) sequential and noisy early on (the reference token may not even be a sensible continuation of the model's own prefix).

Scheduled sampling interpolates between the two.""")

single(C, "Scheduled sampling (Bengio et al., 2015) is a curriculum between teacher forcing and free-running training. How does it proceed over the course of training?",
       ("At first the decoder is mostly fed reference tokens; the chance of feeding it its own prediction instead grows on a schedule as training proceeds",
        "A curriculum from easy (guided) to hard (self-fed) inputs: the probability of using the ground truth starts near 1 and decays, so the model gradually learns to cope with its own outputs."),
       [("The decoder is first trained fully free-running; teacher forcing is switched on later to fine-tune it once its predictions are good",
         "The schedule runs the other way. Starting free-running is exactly what is slow and unstable, which is why training starts with mostly gold inputs."),
        ("A random vocabulary token replaces the reference input with a probability that grows over training, acting as input noise",
         "What replaces the reference token is the model's own prediction (its arg max or a sample), not an arbitrary word; the goal is to mimic test-time conditions, not to add noise."),
        ("The soft labels of a larger pre-trained network are mixed into the loss, and their weight is decayed over training",
         "That would be a distillation schedule. Scheduled sampling involves a single model and changes the decoder inputs, not the loss targets.")],
       """For each decoder step flip a coin: with probability ε_{i} feed the reference token, otherwise feed the model's own prediction (arg max or a sample). ε_{i} starts near 1 and decays with the training step i (linearly, exponentially or by an inverse sigmoid).

The loss is still computed against the reference tokens.""")

single(C, "Which statement correctly separates teacher forcing from the teacher–student setting of knowledge distillation?",
       ("Teacher forcing feeds gold tokens to one decoder; distillation trains a student to imitate another model's outputs",
        "Teacher forcing is about the decoder inputs of a single network; distillation involves two networks and uses the teacher's outputs as training targets."),
       [("Both feed a larger network's predictions to a smaller network at every step; they differ only in whether a softmax temperature is used",
         "Teacher forcing has no larger network and no temperature. Its 'teacher' is just the ground-truth data supplying the previous token."),
        ("Teacher forcing is applied at test time to correct the decoder using the reference, whereas distillation happens only during training",
         "At generation time no reference exists, so the decoder has to consume its own outputs. Teacher forcing can only be used where the reference is known (in training and in teacher-forced scoring such as held-out perplexity), never to correct outputs generated for new inputs."),
        ("In teacher forcing the encoder acts as the teacher and the decoder as the student, because the decoder learns from the encoder's states",
         "The encoder and decoder are two parts of one network trained jointly; nothing in teacher forcing makes one a teacher of the other.")],
       """The word “teacher” is used in two unrelated ways:

- **Teacher forcing**: a training trick for autoregressive models (feed the ground truth as the next input). One model, no soft targets.
- **Knowledge distillation**: a compression method (a trained teacher's output distributions become the student's targets). Two models.

The two can be combined, but they are independent ideas.""")

single(C, "Why can a Transformer decoder be trained on all target positions in one parallel pass, whereas generating a translation at test time needs one decoder step per output token?",
       ("Training inputs (the shifted reference) are all known up front, so one causal-masked pass suffices; at test time each input is the previous output",
        "Teacher forcing makes all inputs available up front. Generation has to wait for each new token before it can be fed back."),
       [("Training uses the parallel self-attention decoder, while generation falls back to a recurrent decoder whose steps have to run sequentially",
         "The same self-attention decoder is used in both phases. Only the availability of the inputs differs, not the architecture."),
        ("During training the decoder ignores earlier target tokens and predicts each position on its own, whereas at test time it has to attend to them",
         "Masked self-attention over earlier target tokens is how the decoder conditions on the prefix in training as well; what changes is where those tokens come from."),
        ("At test time the encoder has to be re-run from scratch for every generated token, whereas in training it only runs once per sentence",
         "The encoder runs once per source sentence in both phases. The sequential cost sits in the decoder, whose next input depends on the previous output.")],
       """Training (teacher forcing): inputs `<s>, y_1, …, y_m` are all known → one forward pass with a causal mask gives all m+1 output distributions.

Generation: y_{t} does not exist until step t has been computed → m+1 sequential decoder steps (m tokens plus the final `</s>`; cached keys/values avoid recomputing the prefix, but the steps cannot be parallelised).""")

# ------------------------------------------------------------------ multiple answer
multi(C, "Teacher forcing is the standard way of training a decoder. Which of the following claims about it are true? Select all that apply.",
      [("At decoder step t the input is the reference token y_{t−1}, even if the model's own prediction at step t−1 was different",
        "That is the definition: the ground truth, not the model's output, is fed forward, so the decoder always sees a correct prefix."),
       ("The loss at each step is still the cross-entropy between the model's predicted distribution and the reference token y_{t}",
        "Teacher forcing changes only what is fed in; the training targets are the same reference tokens in every variant."),
       ("Early in training it is faster and more stable than feeding back the model's own predictions, because the inputs are never corrupted by its poor early guesses",
        "Clean inputs give a clear learning signal from the first update, whereas a barely trained model that feeds itself would mostly see wrong tokens."),
       ("Held-out perplexity of a conditional LM is normally computed in this teacher-forced way, with the reference prefix as input",
        "Perplexity scores the probability of the reference tokens given the reference prefix, so no generation is needed. It does not directly measure the quality of free-running outputs.")],
      [("It can only be applied to recurrent decoders; a Transformer decoder has to be trained by feeding back its own predictions",
        "Transformer decoders are the main beneficiaries: with a causal mask the shifted reference is fed in and all positions are trained in one pass."),
       ("It is the technique by which a small model learns from the output distributions of a larger model",
        "That describes knowledge distillation. Teacher forcing involves a single model; its 'teacher' is simply the ground-truth data.")],
      """Teacher forcing = ground-truth previous token as input, cross-entropy against the ground-truth next token as loss.

Pros: stable, and for Transformer decoders parallel over target positions. Con: exposure bias. Dev-set perplexity is computed the same teacher-forced way, so it says little about free-running quality.""")

multi(C, "A trained translation model is now used on a new source sentence for which no reference translation exists. Which statements about generating the output are correct? Select all that apply.",
      [("Every decoder input after the start symbol is a token that the model itself produced at an earlier step, whether chosen greedily, by beam search or by sampling",
        "With no reference there is nothing to teacher-force; the decoder can only be fed its own outputs, one step at a time."),
       ("Generation ends when `</s>` is produced or a maximum output length is reached",
        "Training targets end with `</s>`, so the model has learned to emit it when the sentence is complete; the length cap is a safety net.")],
      [("All target positions can still be computed in one parallel pass, because the encoder has already processed the whole source sentence",
        "The source is not the issue: the input at position t is the model's own output at position t−1, which does not exist until that step has been computed."),
       ("Greedy decoding is guaranteed to return the output sequence with the highest probability p(y | x)",
        "Picking the locally best token at each step can miss a globally better sequence. Beam search widens the search but is not guaranteed to find the optimum either."),
       ("Candidate outputs are ranked by their cross-entropy against the reference translation",
        "There is no reference at test time; candidates can only be ranked by the model's own score, e.g. their (length-normalised) log-probability.")],
      """At test time the model runs free: `<s>` → ŷ_{1} → ŷ_{2} → … until `</s>`. Each ŷ_{t} is chosen from p(· | ŷ_{<t}, x) by greedy search, beam search or sampling.

Because every step depends on the previous output, generation needs one decoder step per token, unlike teacher-forced training.""")

multi(C, "Which of the following are naturally modelled as conditional language models, i.e. generators of a token sequence y, one token at a time, from p(y | x) for some external input x that goes beyond the tokens already generated? Select all that apply.",
      [("Image captioning, where x is an image and y is a description of it",
        "The caption is generated word by word, each word conditioned on the image and on the words so far."),
       ("Machine translation, where x is a source-language sentence",
        "The classic case: p(y | x) = ∏ p(y_{t} | y_{<t}, x) with an encoder reading x."),
       ("Abstractive summarisation, where x is the article",
        "The summary is generated token by token while conditioning on the input document."),
       ("Dialogue response generation, where x is the conversation so far",
        "The reply is generated autoregressively, conditioned on the dialogue history.")],
      [("A word-level language model trained on a monolingual corpus, which sees only the words it has generated so far",
        "This is an unconditional LM: p(y) = ∏ p(y_{t} | y_{<t}) has no external input x."),
       ("A binary sentiment classifier whose output layer is a single two-way softmax over {positive, negative}",
        "It outputs one categorical decision p(label | text), not a token sequence factorised as ∏_{t} p(y_{t} | y_{<t}, x), so it is a discriminative classifier rather than a conditional language model. (A generative model that writes the label word as a one-token output sequence, as text-to-text models such as T5 do, would be a conditional LM used as a classifier.)")],
      """A conditional LM keeps the autoregressive factorisation of an LM and adds the input x to every factor: p(y | x) = ∏_{t} p(y_{t} | y_{<t}, x). The conditioning input can be a sentence, an image, a document, a dialogue history or a prompt.

Drop x and you are back at an unconditional LM.""")

multi(C, "Select all correct statements about the standard maximum-likelihood training of a conditional LM.",
      [("By the chain rule, log p(y | x) = ∑_{t} log p(y_{t} | y_{<t}, x), so the sequence loss is a sum of per-token losses",
        "A product of conditional probabilities becomes a sum after taking logs, which is why the loss decomposes over target positions."),
       ("At each position the loss is the cross-entropy between the one-hot reference token and the softmax output, i.e. −log of the probability given to the reference token",
        "Minimising it pushes up the probability the model assigns to the reference token."),
       ("Padding positions in a batch are masked out so that they contribute nothing to the loss",
        "Padded slots are not real tokens; counting them would reward or penalise the model for predicting padding.")],
      [("BLEU can be used directly as the training loss, since it is differentiable with respect to the output probabilities",
        "BLEU is computed from discrete generated tokens (an arg max or sampling step), so it is not differentiable. Optimising it needs sequence-level methods such as minimum-risk training or reinforcement learning."),
       ("Positions where the model's arg max already equals the reference token contribute zero loss",
        "The loss is −log p(reference), which is positive unless p = 1. Even at p = 0.6 it is about 0.51 nats, and the gradient keeps pushing the probability up.")],
      """Maximum likelihood on (x, y) pairs: minimise L = −∑_{t} log p(y_{t} | y_{<t}, x), summed or averaged over the real (non-padding) target tokens.

It is a smooth, differentiable per-token objective, unlike metrics such as BLEU, which are computed on discrete outputs.""")

multi(C, "Which of the following *training-time* changes reduce the mismatch between training inputs and generation inputs that teacher forcing creates (exposure bias)? Select all that apply.",
      [("Scheduled sampling, which gradually replaces gold previous tokens by the model's own predictions as training proceeds",
        "It exposes the decoder to its own, imperfect prefixes in training, so test-time inputs look less foreign."),
       ("Free-running (student-forcing) training, in which the decoder is fed its own previous predictions throughout",
        "Training inputs then match generation, at the price of less stable training (and, for Transformer decoders, losing the one-pass parallelism)."),
       ("Sequence-level training such as minimum-risk training or policy-gradient methods that score outputs the model generated itself",
        "The model is trained and rewarded on sequences it sampled or decoded, i.e. on the inputs it will actually face at generation time.")],
      [("Label smoothing of the reference distributions",
        "It regularises over-confident output distributions, but the decoder inputs are still gold prefixes, so the mismatch remains."),
       ("Simply training for more epochs on the same reference data",
        "The decoder still only ever sees gold prefixes, so more epochs do not change the kind of inputs it meets; the mismatch with generation remains.")],
      """Ways to attack exposure bias: expose the model to its own outputs during training (scheduled sampling, free-running training) or optimise sequence-level objectives on its own samples (minimum-risk / RL).

Label smoothing and extra epochs leave the training inputs unchanged. (A wider beam at test time can help the decoder recover from some early errors, but it changes only the search, not the training.)""")
