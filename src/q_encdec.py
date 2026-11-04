from qdsl import tf, single, multi, num, T

E = "encdec"

# ---------------------------------------------------------------- true / false
tf(E, "Although the training loss is computed only from the decoder's output distributions, the encoder's weights are still updated, because gradients flow back through the context that the encoder passes to the decoder.", True,
   """The loss L = Σ_{t} −log p(y_{t} | y_{<t}, x) depends on the encoder because the decoder is initialised from (or conditioned on) the encoder's final state. Back-propagation therefore carries ∂L/∂context into the encoder and on through all of its time steps (BPTT).

The whole encoder–decoder is trained end to end with this single loss; the encoder has no output layer or loss of its own.""")

tf(E, "Since the complete reference sentence is available during training, the decoder of a seq2seq model can be made bidirectional, just like the encoder.", False,
   """The decoder is a conditional language model that generates left to right. A bidirectional decoder would use right-hand context (future target words) that does not exist at test time, so it could not generate anything, and training with it would be cheating.

The encoder is different: the whole source sentence is known before decoding starts, so reading it in both directions is allowed and often helpful.""")

tf(E, "An encoder–decoder model does not need an RNN encoder: for image captioning, a CNN can encode the image into a feature vector that conditions an RNN decoder, which then generates the caption word by word.", True,
   """The framework only needs an encoder that produces a representation the decoder can condition on. In captioning models of the Show-and-Tell type (Vinyals et al., 2015) the encoder is a CNN and the decoder an LSTM language model, so the model computes p(caption | image).

The same pattern covers speech recognition (acoustic encoder, text decoder), summarisation and dialogue (text encoder, text decoder).""")

tf(E, "In the basic RNN encoder–decoder (without attention), the decoder receives a new context vector at every output step, computed as a weighted average of all the encoder's hidden states.", False,
   """That describes attention (Bahdanau et al., 2014). In the basic model the encoder hands over one fixed-size vector, its final state, and it does not change during decoding: it either initialises the decoder (Sutskever et al.) or is fed in unchanged at every step (Cho et al.).

This single vector has to summarise the entire source sentence, which is the bottleneck that attention was introduced to remove.""")

# ---------------------------------------------------------------- single answer
single(E, "A seq2seq model defines the probability of a target sentence y = (y_{1}, …, y_{m}), whose last symbol y_{m} is the end marker `</s>`, given a source sentence x. Which factorisation does the decoder implement?",
       ("p(y | x) = ∏_{t=1}^{m} p(y_{t} | y_{<t}, x), with x reaching the decoder through the encoder's context",
        "Each target word is predicted from all previous target words plus a summary of the whole source, which makes the decoder a conditional language model."),
       [("p(y | x) = ∏_{t=1}^{m} p(y_{t} | x_{t}), so the t-th target word depends only on the t-th source word",
         "That assumes a word-by-word alignment and equal lengths. Seq2seq needs neither: the decoder sees the whole source (via the context) and all previous target words."),
        ("p(y | x) = ∏_{t=1}^{m} p(y_{t} | y_{<t}), an ordinary target-side language model that ignores the source",
         "That drops the conditioning on x; such a model can generate fluent text but cannot translate or summarise any particular input."),
        ("p(y | x) = ∏_{t=1}^{n} p(x_{t} | x_{<t}, y), i.e. the decoder reconstructs the source from the target",
         "That is the reverse direction (a noisy-channel style p(x | y)); the decoder generates target words conditioned on the source, not the other way round.")],
       """By the chain rule, p(y | x) = Π_{t} p(y_{t} | y_{<t}, x). The encoder turns x into the context c, and the decoder's hidden state summarises y_{<t} together with c. Training maximises Σ_{t} log p(y_{t} | y_{<t}, x) over parallel sentence pairs. The last factor, p(`</s>` | y_{<m}, x), is how the decoder learns when to stop.""")

single(E, "A seq2seq model is trained with teacher forcing on the target sentence “the cat sat”. Which decoder input sequence and which label (gold output) sequence are used?",
       ("Decoder input: `<s>` the cat sat; labels: the cat sat `</s>`",
        "The labels are the inputs shifted left by one: at each step the decoder sees the previous reference word and must predict the next one, starting from `<s>` and ending with `</s>`."),
       [("Decoder input: the cat sat `</s>`; labels: `<s>` the cat sat, so that the start symbol is the first word to be predicted",
         "The shift goes the wrong way: the first step would have to predict the start symbol, and every later label is the word that was fed in one step earlier, so the model would learn to look backwards instead of predicting the next word."),
        ("Decoder input: the cat sat; labels: the cat sat, with no special start or end symbols",
         "Identical sequences let the decoder copy its current input instead of predicting the next word, and without `<s>` and `</s>` it can neither start nor stop properly."),
        ("Decoder input: the source sentence followed by `<s>`; labels: the cat sat `</s>`",
         "The source goes to the encoder. The decoder is a language model over the target side, so its inputs are the (shifted) target words.")],
       """With reference y = (y_{1}, …, y_{m}), the decoder inputs are (`<s>`, y_{1}, …, y_{m}) and the labels are (y_{1}, …, y_{m}, `</s>`). One forward pass yields all m + 1 predictions, and the loss sums the cross-entropy at each position. At test time the inputs after `<s>` are the decoder's own previous outputs.""")

single(E, "Why does the translation quality of a basic RNN encoder–decoder (no attention) tend to degrade on long source sentences?",
       ("A single fixed-size vector must summarise the whole source, so detail is lost as the sentence grows",
        "A vector of fixed capacity (say 1,000 numbers) has to represent 5 words as well as 50; the longer the source, the more of it is lost or overwritten."),
       [("The context vector grows with the sentence length, which makes the decoder's input layer too large to train properly",
         "The context vector has the same dimension whatever the source length. That is precisely the problem: capacity does not grow with the input."),
        ("An RNN encoder reads all source words in parallel rather than one after another, so the order of the words is gradually lost in long sentences",
         "An RNN reads one token per time step, strictly in order, and its state is order-sensitive. What limits quality is the capacity of the single final vector, not a loss of word order."),
        ("The softmax over the target vocabulary becomes larger when the source is longer, which makes each prediction harder",
         "The softmax size equals the target vocabulary size and is unrelated to the length of the source sentence.")],
       "The decoder's only view of the source is the fixed-size context. Cho et al. (2014) and Bahdanau et al. (2014) observed that BLEU of such models falls off as sentences get longer. Attention fixes this by giving the decoder access to all encoder states, weighted differently at each step.")

single(E, "Sutskever et al. (2014) reversed the word order of every source sentence (but not of the targets) before feeding it to the LSTM encoder. What was the benefit?",
       ("The first source words end up next to the first target words, creating short-term dependencies that make optimisation easier",
        "The average source–target distance stays the same, but the minimal time lag shrinks, so the gradient has short paths to learn from early in training."),
       [("It shrinks the average distance between every source word and its translation, so all alignments become short",
         "The average distance is unchanged: the first source words move closer to their translations, but the last ones move further away."),
        ("It makes the encoder bidirectional, so the context vector reflects both left and right context of every word",
         "Reversal only changes the reading order. The encoder is still unidirectional and reads each word once."),
        ("It shortens the source sentences, so the encoder runs fewer steps and the final state can carry more information",
         "A reversed sentence has exactly the same number of words; nothing is removed or added.")],
       "The reversal trick is a data transformation, not an architecture change. It was helpful because the first target words depend mostly on the first source words, and after reversal those words are the most recently read by the encoder. Attention later removed the need for such tricks.")

single(E, "At test time the target length is unknown. How does an RNN decoder decide when to stop generating?",
       ("It learned to emit `</s>` after the last word, and decoding ends once that symbol is produced (or a maximum length is reached)",
        "Every training target ends with `</s>`, so the model learns a probability of stopping at each step, just like for any other word."),
       [("It generates exactly as many words as the source sentence contains, because the encoder stores the source length in the context vector and the decoder counts down from it",
         "Output length is not tied to source length (translations are often longer or shorter), and the context vector does not act as a step counter."),
        ("It stops once every source word has been translated at least once, which its hidden state keeps track of",
         "A basic decoder keeps no record of which source words it has covered; it only has its hidden state and the fixed context."),
        ("It stops as soon as its highest word probability drops below a fixed confidence threshold set on held-out data",
         "No such threshold is used: termination is a learned decision to emit `</s>`, not a confidence test.")],
       "Appending `</s>` to every training target teaches the decoder where sentences end. At inference, decoding stops when `</s>` is generated; implementations add a maximum length as a safety net, e.g. in case a model never emits `</s>`.")

single(E, "Suppose the encoder's final state is not passed to the decoder: the decoder starts from a zero state and has no other connection to the encoder. What does the model become?",
       ("An unconditional language model of the target language: its output no longer depends on the source, and the encoder receives no gradient",
        "With no path from the source to the decoder, the loss cannot depend on the encoder's parameters, so the encoder learns nothing and the decoder can only model p(y)."),
       [("It still translates correctly, because during teacher forcing the decoder's inputs already contain the source words",
         "Decoder inputs are (shifted) target words; source words never enter the decoder except through the context."),
        ("It trains better, since the fixed-size bottleneck between encoder and decoder has been removed",
         "Cutting the only channel does not widen the bottleneck, it removes all source information."),
        ("It turns into an attention model, because the decoder can now read every encoder state freely",
         "Attention needs an explicit mechanism that scores and combines encoder states. With no connection, the decoder cannot see them at all.")],
       "The context is what turns the decoder into a *conditional* language model p(y | x). Remove it and you are left with p(y), the same model you would train on target-language text alone. It also shows that the encoder is trained only through the context.")

# ---------------------------------------------------------------- multiple answer
multi(E, "Which statements about the context vector of a basic RNN encoder–decoder (without attention) are correct? Select all that apply.",
      [("Its dimensionality is fixed by the model and does not grow with the length of the source sentence",
        "A 1,000-dimensional context stays 1,000-dimensional for a 5-word or a 50-word source; this fixed capacity is the bottleneck."),
       ("It is typically the encoder's final hidden state (for an LSTM, together with the final cell state)",
        "The state after reading the last source token (and `</s>` if used) is used to initialise the decoder.")],
      [("It is recomputed at every decoder step from the current decoder state",
        "That is attention. In the basic model the context is computed once, from the source alone, and does not depend on the decoder's progress."),
       ("It is the average of all the encoder's hidden states, so the order of the source words does not matter",
        "The basic model uses the encoder's final state, which is order-sensitive: it was produced by reading the words one after another.")],
      "Basic seq2seq: encode once → one fixed-size vector → decode. The intermediate encoder states h_{1}, …, h_{n−1} are not passed to the decoder; they only serve to compute the final state (and to carry gradients back during training).")

multi(E, "Which of the following tasks naturally fit the encoder–decoder pattern, i.e. an input has to be encoded and a variable-length output sequence generated from it? Select all that apply.",
      [("Abstractive summarisation: a document goes in, a short summary comes out",
        "Input and output are both text sequences of different lengths, the classic seq2seq setting."),
       ("Image captioning: an image goes in, a sentence comes out",
        "A CNN encoder produces an image vector and an RNN decoder generates the caption word by word."),
       ("Speech recognition: a sequence of acoustic frames goes in, a transcript comes out",
        "An acoustic encoder reads the frames and a decoder emits the transcript, whose length differs from the number of frames.")],
      [("Assigning one sentiment label (positive or negative) to a movie review",
        "The output is a single label, so an encoder plus a classifier is enough. There is no sequence to generate."),
       ("Unconditional text generation: sampling sentences from a language model with no input at all",
        "With nothing to encode, a plain language model (a decoder alone) suffices.")],
      "Encoder–decoder models handle input-conditioned generation: MT, summarisation, dialogue response generation, captioning, speech recognition. Classification and unconditional generation do not need both halves.")

multi(E, "Which statements about the encoder and decoder RNNs in a seq2seq model are correct? Select all that apply.",
      [("They are separate networks with their own weights; only the context (state) is handed from one to the other",
        "Initialising the decoder from the encoder's final state copies a vector, not the weights."),
       ("The encoder may read the source in both directions, because the whole source sentence is available before decoding starts",
        "A bidirectional encoder concatenates forward and backward states; nothing about generation is leaked."),
       ("At each step the decoder is conditioned on the previous target word: the reference word in training, its own prediction at test time",
        "This autoregressive conditioning is what makes the decoder a language model over the target.")],
      [("Both networks must run for the same number of time steps, since each target word is aligned with exactly one source word",
        "The encoder runs once per source word and the decoder once per target word (plus `</s>`); the lengths generally differ."),
       ("The decoder must use exactly the same weight matrices as the encoder, because it starts from the encoder's final state",
        "Only the state is transferred. Encoder and decoder normally have separate parameters (they model different languages or modalities)."),
       ("In the basic model the decoder also reads the source words directly at every step, next to the previous target word",
        "The decoder sees the source only through the context vector; that is why the context is a bottleneck.")],
      "Two RNNs: an encoder (reads x, may be bidirectional) and a decoder (generates y left to right). Their parameters are separate; the encoder's final state initialises the decoder.")

multi(E, "Which statements about training and decoding a seq2seq model are correct? Select all that apply.",
      [("The loss is the cross-entropy between the decoder's predicted distributions and the reference words, summed over the target positions",
        "It equals −Σ_{t} log p(y_{t} | y_{<t}, x), computed on decoder outputs only."),
       ("At test time the decoder's own previous output, not the reference word, is fed back as the next input",
        "No reference exists at test time, so the decoder has to rely on its own predictions (greedy choice, beam search or sampling).")],
      [("The encoder has its own auxiliary loss that rewards reconstructing the source sentence from the context vector",
        "A plain seq2seq model has no such term; the encoder is trained only through the decoder's loss."),
       ("The loss is computed only at the last decoder step, on the prediction of `</s>`, because that is when the output is complete",
        "Every position contributes: each target word and `</s>` is predicted and penalised."),
       ("At test time the reference words are still fed to the decoder as its inputs, exactly as in training, to keep the output on track",
        "Teacher forcing needs the gold prefix, which does not exist at test time; this train/test difference is known as exposure bias.")],
      "Training: teacher forcing, token-level cross-entropy on decoder outputs, one backward pass through decoder and encoder. Inference: decode left to right from `<s>` until `</s>`, feeding back the model's own choices.")

multi(E, "Which statements about the fixed-size context bottleneck and its remedies are correct? Select all that apply.",
      [("Bahdanau et al. (2014) reported that the basic model's quality drops quickly as sentences get longer, while their attention model stays much more robust",
        "This observation is the empirical motivation for attention."),
       ("Attention lets the decoder use a different weighted combination of the encoder's states at each output step, instead of one fixed vector",
        "The context becomes c_{t}, recomputed for every output position."),
       ("A bidirectional encoder yields a richer summary but, without attention, still passes a single fixed-size vector to the decoder",
        "Concatenating forward and backward final states changes the content of the vector, not the fact that everything must fit into it.")],
      [("Teacher forcing removes the bottleneck, because the decoder sees the correct previous target word at every step and relies less on the context",
        "Teacher forcing feeds target-side words to the decoder; the source information must still pass through the fixed context."),
       ("Reversing the source sentence removes the bottleneck, because the decoder can then read all source words in the opposite order",
        "Reversal only changes the reading order for the encoder. It shortens some dependencies but the decoder still sees just one vector.")],
      "Bottleneck = one fixed-size vector for the whole source. Attention (a per-step weighted combination of all encoder states) removes it; tricks such as reversal or bidirectionality only soften it.")
