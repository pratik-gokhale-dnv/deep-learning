from math import log, prod

from qdsl import tf, single, multi, num, T

Z = "beam"

# ---------------------------------------------------------------- toy model for the trace question (all numbers computed here)
EOS = "</s>"
LM = {"<s>": {"X": 0.50, "Y": 0.40, EOS: 0.10},
      "X": {"X": 0.05, "Y": 0.60, EOS: 0.35},
      "Y": {"X": 0.10, "Y": 0.40, EOS: 0.50}}


def _greedy():
    seq, p, prev = [], 1.0, "<s>"
    while prev != EOS:
        w = max(LM[prev], key=LM[prev].get)
        p *= LM[prev][w]
        seq.append(w)
        prev = w
    return " ".join(seq), p


def _beam(B):
    """Keep the B best candidates; kept candidates ending in </s> are set aside as finished."""
    live, fin, lines, step = [((), 1.0)], [], [], 0
    while live:
        step += 1
        cand = [(s + (w,), p, q, p * q) for s, p in live for w, q in LM[s[-1] if s else "<s>"].items()]
        cand.sort(key=lambda c: -c[3])
        keep = cand[:B]
        lines.append("step %d" % step)
        for s, p, q, pq in cand:
            tag = ""
            if (s, p, q, pq) in keep:
                tag = "  <- kept, finished" if s[-1] == EOS else "  <- kept"
            lines.append("  %-9s %.3f x %.2f = %.3f%s" % (" ".join(s), p, q, pq, tag))
        fin += [c for c in keep if c[0][-1] == EOS]
        live = [(c[0], c[3]) for c in keep if c[0][-1] != EOS]
        if fin and (not live or max(f[3] for f in fin) > max(l[1] for l in live)):
            break
    best = max(fin, key=lambda c: c[3])
    return " ".join(best[0]), best[3], "\n".join(lines)


def _exhaustive(maxlen=8):
    res = []

    def rec(seq, p):
        for w, q in LM[seq[-1] if seq else "<s>"].items():
            if w == EOS:
                res.append((" ".join(seq + [w]), p * q))
            elif len(seq) < maxlen:
                rec(seq + [w], p * q)

    rec([], 1.0)
    return sorted(res, key=lambda r: -r[1])


G_SEQ, G_P = _greedy()
B_SEQ, B_P, TRACE = _beam(2)
assert (G_SEQ, B_SEQ) == ("X Y </s>", "Y </s>") and B_P > G_P
assert _beam(1)[0] == G_SEQ and _exhaustive()[0][0] == B_SEQ  # B=1 is greedy; Y </s> is the true optimum
TOP = [p for _, p in _exhaustive() if p > 0.01]
assert len({round(p, 9) for p in TOP}) == len(TOP)  # no ties among the complete sequences that matter
X_END = LM["<s>"]["X"] * LM["X"][EOS]  # 'expand only the best hypothesis' mistake
SUM_ERR = LM["<s>"]["Y"] + LM["Y"][EOS]  # 'add instead of multiply' mistake
TABLE = "\n".join(["%-10s" % "prev word" + "".join("%7s" % w for w in ("X", "Y", EOS))] +
                  ["%-10s" % k + "".join("%7.2f" % LM[k][w] for w in ("X", "Y", EOS)) for k in LM])

# ---------------------------------------------------------------- true / false
tf(Z, "Beam search with any beam size B ≥ 2 is guaranteed to return the exact highest-probability output sequence under the model, which is why it is preferred to greedy decoding.", False,
   """Beam search is a heuristic. It keeps only B partial hypotheses per step, so a prefix that looks mediocre early on but would have led to the best complete sequence can be pruned for good.

Exact search must consider up to V^{T} sequences (vocabulary size V, output length T), which is exponential in T. A wider beam usually reduces search errors, but nothing guarantees optimality unless the beam is large enough to keep every prefix.""")

tf(Z, "A larger beam finds output sequences with higher model probability, so the translations it returns also score higher on BLEU.", False,
   """Two different errors are being conflated. A wider beam reduces **search error** (it finds hypotheses with a higher model score), but BLEU also depends on **model error**: the model's most probable sequences are not always the best translations.

In practice BLEU typically saturates at small beams (roughly 4–10) and can even fall for very large ones, because the highest-scoring outputs tend to be short or generic. Cost grows roughly linearly with B.""")

tf(Z, "In standard maximum-likelihood training with teacher forcing, no beam search is run: the decoder is fed the reference prefix at every step, and beam search is used only when generating at test time.", True,
   """Training: at step t the decoder receives the gold word y_{t−1}, and the loss is the sum of −log p(y_{t} | y_{<t}, x) from a single forward pass. No search is needed because the reference is known.

At test time the reference is unavailable, so the system has to search for an output itself (greedy, beam search or sampling). Some sequence-level training methods do decode during training, but ordinary maximum-likelihood training does not.""")

# ---------------------------------------------------------------- single answer
single(Z, "In beam search with beam size B and vocabulary size V, how is the next beam chosen once every live hypothesis has been extended by every word?",
       ("Score all B·V extensions by cumulative log-probability and keep the B best overall, even if several share a parent",
        "Pruning is global: a strong hypothesis can contribute two or more of the survivors while a weak one contributes none."),
       [("Extend each hypothesis only with its own single most probable next word, so the survivors are B independent greedy chains",
         "That is B separate greedy decoders. Nothing is compared across hypotheses, so weak parents are never dropped and the beam never branches."),
        ("Rank the words by this step's conditional probability alone and attach the B best words to the single best hypothesis",
         "It discards the accumulated score of each prefix. Candidates must be ranked by cumulative log-probability (prefix score plus the new word's log-probability)."),
        ("Draw B candidates at random from the B·V extensions with probability proportional to their scores",
         "That describes stochastic sampling; beam search is deterministic and simply keeps the top-B scores.")],
       """One step: (1) run the decoder for each of the B live hypotheses; (2) form the B·V candidate scores `score(h) + log p(w | h, x)`; (3) keep the B highest. Because candidates compete across parents, this differs from running B greedy decoders side by side.""")

single(Z, "Why do practical beam-search implementations add log-probabilities instead of multiplying probabilities?",
       ("Long products of probabilities below 1 underflow to zero in floating point; sums of log-probabilities stay well scaled and rank hypotheses identically",
        "log is monotonic, so the ordering of hypotheses is unchanged, but 40 words of probability 0.01 each (10^{−80}) would be rounded to 0 in float32."),
       [("Summing log-probabilities removes the preference for short outputs, so no length normalisation is needed when hypotheses of different lengths are compared",
         "A sum of negative terms still decreases with every added word, so the bias against long hypotheses is still there."),
        ("The decoder's softmax already returns log-values, so multiplying them would be mathematically invalid whereas adding them is the correct operation",
         "A softmax returns ordinary probabilities in [0, 1] (log-softmax is just a numerically convenient variant). Multiplication is valid; it merely underflows."),
        ("It makes the search exact, because the logarithm of a product equals the sum of the logarithms and so no hypothesis is ever pruned wrongly",
         "The identity is exact, but it changes nothing about which hypotheses are pruned, so the search remains a heuristic.")],
       "log(p_{1} · p_{2} ⋯ p_{n}) = Σ log p_{i}. Since log is increasing, the top-B hypotheses are the same whether scores are compared as products or as sums; only the numerical behaviour (and the cheap incremental update `score + log p`) improves.")

single(Z, "What does length normalisation (a length penalty with exponent α) do in beam search?",
       ("It divides a hypothesis's log-probability by a length-dependent factor such as length^{α}, so long outputs are not unfairly penalised",
        "α = 0 leaves the raw score unchanged and α = 1 gives the average log-probability per word. The Transformer paper used α = 0.6 (with beam size 4)."),
       [("It multiplies the training loss by the target length so that long sentences contribute as much as short ones",
         "Length normalisation is a decoding-time rescaling used to rank hypotheses; it does not alter the training loss."),
        ("It truncates every hypothesis to the length of the shortest finished one so that all scores are comparable",
         "Nothing is truncated: hypotheses keep all their words and only the scores used for ranking are rescaled."),
        ("It enlarges the beam for long inputs, so that more candidate hypotheses survive pruning at every step",
         "The beam width B stays fixed; what changes is the score used to rank candidates.")],
       """Raw scores Σ log p(y_{t} | …) only go down as words are added, so short hypotheses win unfairly. Normalised score: `score(Y) = log p(Y | x) / lp(Y)` with, for example, lp = |Y|^{α} (GNMT uses ((5 + |Y|)/6)^{α}).

Larger α means stronger correction. The Transformer paper decoded with beam size 4 and α = 0.6, tuned on the development set.""")

single(Z, "A team raises the beam size of their NMT system from 5 to 200. Which outcome is the most likely?",
       ("Decoding becomes much slower and more memory-hungry, while BLEU stops improving and may even drop",
        "Cost grows roughly linearly with B, and large beams cut search error, but the model's top-scoring sequences are often not the best translations."),
       [("BLEU keeps rising steadily, because a sequence with higher model probability is by definition a better translation, so more search can only help",
         "That treats model probability as ground truth. Search error shrinks, but model error (a bias towards short or generic outputs) remains and can dominate."),
        ("Decoding time stays almost the same, since the 200 hypotheses are independent and cost nothing extra",
         "Every step needs 200 decoder states and 200·V candidate scores. Batching on a GPU hides some of this, but memory and compute still grow about linearly."),
        ("The search becomes exact, because the beam is now larger than a typical sentence length",
         "Exactness would require keeping all V^{t} prefixes at step t; the beam width has no relation to sentence length.")],
       "Wider beams find higher-scoring hypotheses, but NMT quality typically saturates at beams of about 4–10 (the Transformer used 4). Beyond that, extra search can expose the model's flaws: short, generic or even empty outputs can have the highest probability.")

single(Z, "For open-ended text generation such as story continuation, why are top-k or nucleus (top-p) sampling often preferred to beam search?",
       ("The most probable sequences tend to be bland and repetitive, while truncated sampling adds diversity and skips the unreliable tail",
        "That is the motivation for truncated sampling: keep the head of the distribution, drop the tail, and still randomise."),
       [("Beam search becomes infeasible once the vocabulary exceeds a few thousand words, because it must score B·V candidates per step, whereas sampling avoids this cost",
         "Beam search handles 30–50k word vocabularies routinely, and sampling needs the same full softmax over the vocabulary at every step, so cost is not the reason."),
        ("Sampling is deterministic and therefore reproducible for a given prompt, whereas beam search gives different output on every run",
         "It is the other way round: beam search is deterministic, and sampling is stochastic."),
        ("Nucleus sampling guarantees that the generated text is the most probable sequence under the model, which beam search cannot do",
         "Sampling does not seek the maximum. Nucleus sampling draws from the smallest set of words whose total probability reaches p.")],
       "For tasks with one right answer (translation, summarisation) a high-probability output is wanted, so beam search works well. For open-ended generation the most probable continuation is often dull or loops, so one samples from the top-k words or the top-p nucleus, with a temperature to control diversity.")

# ---------------------------------------------------------------- multiple answer
multi(Z, "Which statements about beam search (with beam size B) are correct? Select all that apply.",
      [("With B = 1 it reduces to greedy decoding",
        "A beam of one hypothesis keeps the best next word at each step, which is exactly greedy decoding."),
       ("It can be used with any autoregressive decoder, e.g. a Transformer as well as an RNN",
        "Beam search only needs the next-word distribution of each hypothesis; it does not depend on the architecture that produces it."),
       ("For a fixed model and input it returns the same output every time (apart from tie-breaking)",
        "No randomness is involved; the top-B selection is deterministic, unlike sampling.")],
      [("At test time it needs the reference translation to decide which hypotheses to keep",
        "Hypotheses are ranked purely by model probability. The reference is used for teacher forcing in training and for scoring (e.g. BLEU) afterwards, not during decoding."),
       ("Its cost per step does not depend on B, because only one hypothesis is extended at a time",
        "All B live hypotheses are extended: B decoder states are kept and B·V candidates scored per step, so the cost grows roughly linearly with B.")],
      "Beam search = deterministic, breadth-limited search over output prefixes ranked by model score. B = 1 is greedy; larger B costs about B times more and lowers (but does not eliminate) the risk of pruning the best sequence.")

multi(Z, "Which statements about scores and finished hypotheses in beam search are correct? Select all that apply.",
      [("A hypothesis's raw score is the sum of its words' log-probabilities, so extending it can never increase that score",
        "Every log-probability is ≤ 0, so each added word contributes a non-positive term. That is why a finished hypothesis that beats all live ones can safely end the search (with raw scores)."),
       ("A hypothesis that has produced `</s>` is finished: it is set aside and not extended further",
        "It is a complete output and is later compared with the other finished hypotheses.")],
      [("With raw scores, longer hypotheses are favoured because they accumulate more evidence",
        "The opposite is true: more words mean more negative terms, so raw scores favour short hypotheses."),
       ("Hypotheses of different lengths can never be compared, so the search must wait until all of them have the same length",
        "They are compared constantly, with raw or length-normalised scores; the length bias is exactly why normalisation exists."),
       ("Once any hypothesis emits `</s>`, that hypothesis is guaranteed to be the best complete output, whatever the live hypotheses score",
        "Not guaranteed: a live hypothesis can still finish with a higher score than the one that has just finished. Stopping after the first finished hypothesis is a common speed shortcut, but it gives up optimality; with raw scores, stopping is only safe once the best finished score beats every live score.")],
      "Bookkeeping: live hypotheses are extended; a hypothesis ending in `</s>` moves to the finished list. With raw log-probability scores the search can stop once the best finished score exceeds the best live score, because live scores can only fall.")

multi(Z, "Which statements about the beam width B are correct? Select all that apply.",
      [("Memory and compute per decoding step grow roughly linearly with B",
        "B decoder states are kept and B·V candidate scores are computed at each step."),
       ("Moving from B = 1 (greedy decoding) to a small beam such as B = 4 or 5 usually improves translation quality",
        "Greedy decoding cannot recover from an early, locally best but wrong word; even a small beam keeps alternatives alive.")],
      [("A beam with B = V, the vocabulary size, is guaranteed to find the best sequence",
        "At step 2 there are already V^{2} candidate prefixes competing for only V slots, so pruning happens from there on and a prefix of the best sequence can be dropped for good; exactness would need a beam that grows exponentially with the output length."),
       ("Larger beams always produce longer outputs",
        "If anything, wider beams make it easier to find short, high-probability hypotheses, which is why a length penalty is used.")],
      "B trades accuracy of the search against cost. Small beams (about 4–10 in NMT) are the practical sweet spot; very large beams cost more and may hurt BLEU.")

# ---------------------------------------------------------------- calculations
num(Z, T("A beam-search decoder uses beam size B = ${B} over a target vocabulary of V = ${V:,} words. Consider a decoding step after the first one, when all ${B} hypotheses in the beam are still unfinished. How many candidate extensions (hypothesis, next word) are scored and compared to choose the next beam?", B=5, V=32000),
    5 * 32000,
    T("""Every live hypothesis is extended by every vocabulary word: B × V = ${B} × ${V:,} = **${N:,}** candidate scores (cumulative log-probabilities). The best ${B} of them form the next beam.

At the first step only one hypothesis (`<s>`) exists, so just V = ${V:,} candidates are scored. Greedy decoding scores only V per step; beam search costs about B times more.""", B=5, V=32000, N=5 * 32000),
    tol=0.5, hint="Enter an integer.")

single(Z, T("""A toy decoder over the vocabulary {X, Y, </s>} has next-word probabilities that depend only on the previous word:

```
${table}
```

Hypotheses are ranked by their total probability (the product of the step probabilities, no length normalisation). **Greedy decoding** takes the most probable word at every step until `</s>`. **Beam search with B = 2** extends every live hypothesis by every word, keeps the 2 most probable candidates, moves a kept candidate that ends in `</s>` to a finished list (it is not extended further), and stops when the best finished hypothesis is more probable than every live one.

Starting from `<s>`, which statement is correct?""", table=TABLE),
       (T("Greedy returns `${g}` (probability ${gp:.3f}); beam search returns `${b}` (probability ${bp:.3f})", g=G_SEQ, gp=G_P, b=B_SEQ, bp=B_P),
        T("Correct. Greedy follows the locally best words and ends with ${gp:.3f}. Beam search keeps the weaker first word Y (0.40), and its extension `Y </s>` reaches ${bp:.3f}, which is strictly higher.", gp=G_P, bp=B_P)),
       [(T("Both methods return `${g}` (probability ${gp:.3f}), because the beam keeps the best first word X and extends it exactly as greedy decoding does", g=G_SEQ, gp=G_P),
         T("The highest-scoring live prefix after step 2 is `X Y` (0.300), but extending it can only lower its score: `X Y </s>` = ${gp:.3f}. The finished hypothesis `Y </s>` (${bp:.3f}) is already better than anything that extension can reach, so beam search returns it.", gp=G_P, bp=B_P)),
        (T("Greedy returns `${g}` (probability ${gp:.3f}); beam search returns `X </s>` (probability ${xe:.3f})", g=G_SEQ, gp=G_P, xe=X_END),
         T("`X </s>` (${xe:.3f}) is only the third-best candidate at step 2 (behind `X Y` 0.300 and `Y </s>` ${bp:.3f}), so a beam of size 2 prunes it and it can never be returned. Beam search also extends the second entry Y, and `Y </s>` (${bp:.3f}) outranks `X </s>`.", xe=X_END, bp=B_P)),
        (T("Greedy returns `${g}` (probability ${gp:.3f}); beam search returns `${b}` (probability ${sm:.3f})", g=G_SEQ, gp=G_P, b=B_SEQ, sm=SUM_ERR),
         T("The sequence is right but the number is not: ${sm:.3f} = 0.40 + 0.50 adds the step probabilities. A sequence probability is their product, 0.40 × 0.50 = ${bp:.3f}.", sm=SUM_ERR, bp=B_P))],
       T("""Trace (candidate = parent score × next-word probability):

```
${trace}
```

After step 2 the live hypothesis `X Y` (0.300) still beats the finished `Y </s>` (${bp:.3f}), so the search continues. Stop after step 3: the best finished hypothesis `Y </s>` (${bp:.3f}) is better than the best live one (`X Y Y`, 0.120), and live scores can only decrease.

Greedy: X (0.50) → Y (0.60) → `</s>` (0.50), giving 0.50 × 0.60 × 0.50 = ${gp:.3f}. Here beam search also happens to find the overall most probable sequence (checked by enumeration), but beam search gives no such guarantee in general.""", trace=TRACE, bp=B_P, gp=G_P),
       calc=True)

# ---------------------------------------------------------------- multiple answer built on worked numbers
H1 = [0.5, 0.4, 0.5]
H2 = [0.6, 0.5, 0.7, 0.5, 0.6, 0.4]
LP1, LP2 = sum(map(log, H1)), sum(map(log, H2))
N1, N2 = LP1 / len(H1), LP2 / len(H2)
assert LP1 > LP2 and N2 > N1  # raw score picks H1, normalised score picks H2
assert sum(H2) / len(H2) > sum(H1) / len(H1)  # H2's words are more probable on average

multi(Z, T("""Beam search has finished two hypotheses for the same source sentence. The probability of each word (the last word of each hypothesis is `</s>`) is:

```
H1 (3 words):  0.5  0.4  0.5
H2 (6 words):  0.6  0.5  0.7  0.5  0.6  0.4
```

Compare the hypotheses by natural-log probability, first as it is (raw score) and then divided by the number of words (length-normalised score). Which statements are correct? Select all that apply."""),
       [(T("On the raw score, H1 is ranked above H2 (${a:.2f} against ${b:.2f})", a=LP1, b=LP2),
         T("Correct. ln P(H1) = ln 0.5 + ln 0.4 + ln 0.5 = ${a:.3f} and ln P(H2) = ${b:.3f}. H2 multiplies six factors below 1 instead of three, so P(H2) = ${p2:.4f} lies well below P(H1) = ${p1:.3f}.", a=LP1, b=LP2, p1=prod(H1), p2=prod(H2))),
        (T("On the length-normalised score, H2 is ranked above H1 (${c:.2f} against ${d:.2f})", c=N2, d=N1),
         T("Correct. ${a:.3f}/3 = ${d:.3f} for H1 and ${b:.3f}/6 = ${c:.3f} for H2, so the per-word score is higher for H2.", a=LP1, b=LP2, c=N2, d=N1)),
        (T("The two scores disagree because H2 has the higher average word probability (${m2:.2f} against ${m1:.2f}) but pays for having six factors below 1 instead of three", m1=sum(H1) / len(H1), m2=sum(H2) / len(H2)),
         "Correct. Every extra word adds a negative term to a log-probability, so the raw score is biased towards short outputs; dividing by the length removes this effect and exposes the better per-word quality of H2.")],
       [(T("Both scores rank H1 first, because its probability (${p1:.3f}) is about four times that of H2 (${p2:.4f}) and dividing by the length cannot overturn such a gap", p1=prod(H1), p2=prod(H2)),
         T("A factor of four in probability is only a difference of ln 4 = ${l4:.2f} in log-probability. Divided by the lengths, the scores become ${d:.3f} (H1) and ${c:.3f} (H2), so the order flips.", l4=log(4), c=N2, d=N1)),
        ("The raw score ranks H2 first because longer hypotheses accumulate more evidence, and the length-normalised score ranks H1 first",
         T("Reversed. A raw log-probability is a sum of negative terms, so the longer hypothesis is penalised (H2 ${b:.2f} < H1 ${a:.2f}); normalisation removes that bias in favour of H2.", a=LP1, b=LP2)),
        (T("Both scores rank H2 first, because its words are more probable on average (mean ${m2:.2f} against ${m1:.2f}), so its total probability must be higher as well", m1=sum(H1) / len(H1), m2=sum(H2) / len(H2)),
         T("A higher average word probability does not make the product higher: H2 multiplies six factors below 1 instead of three, so P(H2) = ${p2:.4f} < P(H1) = ${p1:.3f}. Only the per-word (normalised) score favours H2.", p1=prod(H1), p2=prod(H2)))],
       T("""Raw scores: ln P(H1) = ${a:.3f}, ln P(H2) = ${b:.3f}. Every extra word adds a negative term, so raw scores are biased towards short outputs and H1 is ranked first.

Length normalisation divides by the number of words (or by length^{α}): ${a:.3f}/3 = ${d:.3f} for H1 and ${b:.3f}/6 = ${c:.3f} for H2, so H2 now wins. The GNMT variant uses ((5 + |Y|)/6)^{α} instead of |Y|; the Transformer paper used α = 0.6.""", a=LP1, b=LP2, c=N2, d=N1),
      calc=True)
