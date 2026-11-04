from qdsl import tf, single, multi, num, T

import math
from collections import Counter

M = "metrics"

# ---------------------------------------------------------------- calc data (computed, never typed by hand)
P = [1 / 2, 1 / 4, 1 / 8]                       # token probabilities of the held-out sentence
N = len(P)
bits = sum(-math.log2(p) for p in P) / N         # mean cross-entropy in bits
nats = sum(-math.log(p) for p in P) / N          # the same in nats
ppl = 2 ** bits
am_inv = sum(1 / p for p in P) / N               # arithmetic mean of the inverse probabilities
inv_am = 1 / (sum(P) / N)                        # inverse of the arithmetic mean of the probabilities
assert abs(ppl - 4) < 1e-9 and abs(math.exp(nats) - 4) < 1e-9 and abs(bits - 2) < 1e-9

NLL_TOTAL, N_TOK = 40, 20                        # total NLL (nats) of a 20-token sentence
ppl2 = math.exp(NLL_TOTAL / N_TOK)

cand = "the the the the the the the".split()
refs = ["the cat is on the mat".split(), "there is a cat on the mat".split()]
clipped = sum(min(c, max(r.count(w) for r in refs)) for w, c in Counter(cand).items())
p1 = clipped / len(cand)
assert clipped == 2 and abs(p1 - 2 / 7) < 1e-12

c_len, r_len = 8, 10
bp = 1.0 if c_len > r_len else math.exp(1 - r_len / c_len)
bp_ratio = c_len / r_len                                   # mistake: plain length ratio
bp_swapped = math.exp(-(1 - c_len / r_len))                # mistake: c/r used where r/c belongs
bp_inverted = math.exp(1 - c_len / r_len)                  # mistake: ratio inverted, BP > 1
exp_txt = f"{1 - r_len / c_len:.2f}".replace("-", "−")     # exponent written with a true minus sign
assert abs(bp - 0.7788) < 1e-4 and exp_txt == "−0.25"

# ------------------------------------------------------------------ true / false
tf(M, "A word-level language model with a perplexity of 50 on a test set is, on average, as uncertain at each position as if it were picking uniformly among 50 equally likely words.", True,
   """Perplexity is exp of the mean per-token cross-entropy. A uniform choice among K words has cross-entropy ln K and therefore perplexity K, so perplexity can be read as the model's effective branching factor.

It is an average over positions: some positions (after “New”, say) are far more predictable than others.""")

tf(M, "BLEU is a recall-oriented metric: it measures what fraction of the reference's n-grams appear in the candidate translation.", False,
   """BLEU is **precision**-oriented: its n-gram precision p_{n} asks what fraction of the candidate's n-grams is found in the references (with clipped counts). Precision alone would reward very short outputs, so a brevity penalty plays the role that recall would.

The recall-oriented family is ROUGE, which is why it is the usual choice for summarisation.""")

tf(M, "Because a uniform model over a vocabulary of V words has perplexity V, no model can have a test-set perplexity larger than V.", False,
   """A uniform model is a reference point, not an upper bound. A model that gives the reference tokens less probability than 1/V on average (for instance because it is confidently wrong) has perplexity above V.

If it gives probability 0 to a single test token the perplexity is infinite.""")

# ------------------------------------------------------------------ single answer
single(M, "Which expression gives the perplexity of a language model on a test sequence w_{1}, …, w_{N}?",
       ("exp( −(1/N) ∑_{i=1}^{N} ln p(w_{i} | w_{<i}) )",
        "The mean per-token negative log-likelihood (cross-entropy in nats), exponentiated; equivalently the inverse geometric mean of the token probabilities."),
       [("−(1/N) ∑_{i=1}^{N} ln p(w_{i} | w_{<i})",
         "This is the cross-entropy in nats per token, i.e. ln of the perplexity. Perplexity is its exponential, so the two numbers differ."),
        ("exp( −∑_{i=1}^{N} ln p(w_{i} | w_{<i}) )",
         "Without the 1/N this equals 1 / p(w_{1} … w_{N}), the inverse probability of the whole text. It grows with the length of the test set instead of being a per-token average."),
        ("( ∏_{i=1}^{N} p(w_{i} | w_{<i}) )^{1/N}",
         "That is the geometric mean of the token probabilities, a number between 0 and 1 where higher is better. Perplexity is its reciprocal."),
        ("exp( −(1/N) ∑_{i=1}^{N} log_{2} p(w_{i} | w_{<i}) )",
         "Mixes bases: with log base 2 (bits) the exponentiation has to be 2^{…}. Exponentiating bits with e overstates the perplexity (3 bits would give e^{3} ≈ 20.1 instead of 2^{3} = 8).")],
       """PPL = exp(H) with H = −(1/N) ∑ ln p(w_{i} | w_{<i}) the average per-token cross-entropy in nats.

Equivalent forms: PPL = 2^{H in bits} = (∏ p(w_{i} | w_{<i}))^{−1/N}. Lower is better, and the value is the “effective number of equally likely choices per token”.""")

single(M, "A word-level language model reports a perplexity of 120 on a test corpus; a character-level model reports 4.5 on the same corpus. What can be concluded?",
       ("Nothing directly: the two numbers are averages over different units and must first be converted to a common one",
        "Each perplexity is an average over its own tokens: few hard predictions for the word model, many easy ones for the character model. A shared unit such as total bits, or bits per character, makes them comparable."),
       [("The character-level model is much better, since a lower perplexity always means the better model",
         "Lower is better only when both numbers measure the same thing. Converting with about 5.5 characters per word, the word model's 120 corresponds to roughly 2.4 per character, so it could even be the better model."),
        ("The two models are about equally good, because perplexity is length-normalised and therefore independent of the tokenisation",
         "Normalising by the number of tokens makes the score depend on how finely the text is split: finer tokens make each prediction easier and the per-token number smaller without the model being better."),
        ("Dividing 120 by the average word length in characters gives the character-level equivalent, which can then be compared with 4.5",
         "Perplexity is an exponentiated mean log-probability, so a per-word value converts to a per-character value with a root, PPL_{char} = PPL_{word}^{1/L} for L characters per word, not by dividing.")],
       """Perplexity depends on what counts as a token. To compare models with different tokenisations, use a unit-free quantity: the total negative log-likelihood of the whole corpus, or bits per character = log_{2}(PPL_{word}) / L, where L is the average number of characters per word (including the space).

Then PPL_{char} = 2^{bits per character} = PPL_{word}^{1/L}.""")

single(M, "In BLEU's modified n-gram precision, candidate n-gram counts are clipped. What is this clipping for?",
       ("It stops a candidate from earning high precision by merely repeating a word found in a reference",
        "Without clipping, `the the the the the the the` would match every one of its words and get precision 7/7; clipping caps the credit at the reference count."),
       [("It makes BLEU recall-oriented by rewarding candidates that cover all of the reference's n-grams",
         "Clipping only limits matches, so it can never add credit for coverage. BLEU stays precision-based; the brevity penalty is what discourages missing content."),
        ("It penalises candidates that are shorter than the reference, so that very short outputs cannot score well",
         "That is the job of the brevity penalty, a separate length-based factor. Clipping acts on n-gram counts, not on sentence length."),
        ("It keeps the geometric mean from collapsing to zero when one of the n-gram precisions is zero",
         "Clipping can only lower the matched counts, so it cannot prevent zeros. Avoiding zero scores is what smoothing methods are for.")],
       """Modified precision p_{n} = ∑ min(Count(g), MaxRefCount(g)) / ∑ Count(g), summed over the candidate's n-grams g.

The clip uses the maximum count in any **single** reference, not the sum over references. Without it, a degenerate output that repeats a frequent word would score perfectly.""")

single(M, "Which of the following is a genuine weakness of BLEU when judging a single translation?",
       ("It counts exact n-gram overlap, so a correct paraphrase in different words can score poorly",
        "Synonyms and legitimate rewordings are not in the reference n-grams, so they get no credit even though the meaning is preserved."),
       [("It ignores word order completely, treating the candidate and the reference as unordered bags of words",
         "Only the unigram precision is order-blind. The 2-, 3- and 4-gram precisions reward matching local word order."),
        ("It can only be computed against exactly one reference translation",
         "BLEU was designed for multiple references: clipping uses the maximum count over all available references, which makes it more forgiving."),
        ("It needs a trained neural network to embed both sentences before they can be compared",
         "BLEU is a purely count-based statistic. Embedding-based metrics such as BERTScore are the ones that need a network.")],
       """BLEU matches surface n-grams, so it cannot recognise synonyms or valid paraphrases, and it correlates with human judgement much better at corpus level than for single sentences.

Metrics such as METEOR (stems and synonyms) or chrF (character n-grams) soften the exact-match problem; human evaluation remains the gold standard.""")

single(M, "A team can run its trained conditional language model over the dev set only in teacher-forced scoring mode, i.e. it computes the probability of each reference token given the reference prefix. It cannot afford to decode any outputs. Which metric can it still report?",
       ("Perplexity on the held-out text",
        "It needs only the probabilities the model assigns to the reference tokens: exp of the mean per-token negative log-likelihood."),
       [("BLEU against a set of reference translations",
         "BLEU compares a generated candidate with the references. With no decoded output there is nothing to score."),
        ("ROUGE-L against a reference summary",
         "ROUGE-L measures the longest common subsequence between a generated summary and the reference; it also needs generated text."),
        ("chrF against a reference translation",
         "chrF compares character n-grams of a generated candidate with the reference, so it too requires decoding.")],
       """Perplexity (like the validation loss) is an **intrinsic** measure computed from the model's probabilities for given text, so it works without generation.

BLEU, ROUGE, METEOR and chrF are **output-based**: they compare generated text with references and therefore need decoding.""")

# ------------------------------------------------------------------ multiple answer
multi(M, "Perplexity is the standard intrinsic measure for language models. Which of the following claims about it hold? Select all that apply.",
      [("It equals exp of the mean per-token negative log-likelihood (natural log), which is the same number as 2 raised to the mean cross-entropy in bits",
        "PPL = exp(H_{nats}) = 2^{H_{bits}}: changing the logarithm base only changes the units of H, not the perplexity."),
       ("If the model assigns probability 0 to even one test token, the perplexity is infinite",
        "−ln 0 = +∞ enters the average, which is why smoothing (or a softmax that never outputs exact zeros) is needed."),
       ("A model that gave every test token probability 1 would reach the best possible perplexity, 1",
        "−ln 1 = 0 and exp(0) = 1. Probabilities cannot exceed 1, so perplexity is never below 1.")],
      [("A low perplexity on the training set shows that the model will also do well on new text",
        "Training perplexity can be pushed down by memorisation. The meaningful number is perplexity on held-out data."),
       ("A perplexity of 1 means the model is guessing uniformly at random",
        "Uniform guessing over V words gives perplexity V. Perplexity 1 means the model is certain and always right on the test text.")],
      """Perplexity = exp(average per-token cross-entropy) = inverse geometric mean of the token probabilities.

Range: 1 (perfect) up to V for a uniform model, and beyond (even ∞) for a model that is worse than uniform. Always evaluate it on held-out data.""")

multi(M, "Consider BLEU as defined by Papineni et al. (2002). Which statements are correct? Select all that apply.",
      [("The score is a brevity penalty multiplied by the geometric mean of the clipped n-gram precisions p_{1}, …, p_{N} (usually N = 4)",
        "BLEU = BP · exp(∑_{n} w_{n} ln p_{n}) with uniform weights w_{n} = 1/N."),
       ("If the candidate shares no 4-gram with the references, unsmoothed BLEU is 0 even when its unigram precision is high",
        "One zero precision makes the geometric mean 0. This is why sentence-level BLEU is usually computed with smoothing."),
       ("The brevity penalty depends only on the candidate and reference lengths, not on which words are used",
        "BP is a function of c and r alone, so a candidate of the right length but with wrong words gets BP = 1."),
       ("With several references, an n-gram's count in the candidate is clipped at the maximum number of times it occurs in any single reference",
        "Clipping against the largest single-reference count is what limits credit for repeated words.")],
      [("The clip is the total number of times the n-gram occurs summed over all references",
        "The cap is the maximum over references, not the sum. Summing would let a candidate collect credit for more repetitions than any single human translation contains."),
       ("The brevity penalty also lowers the score of candidates that are longer than the reference",
        "BP = 1 whenever c > r. Overly long outputs lose out only indirectly, because extra words that match no reference n-gram reduce the precisions.")],
      """BLEU = BP · exp(∑_{n=1}^{N} w_{n} ln p_{n}), BP = 1 if c > r and exp(1 − r/c) otherwise (c = candidate length, r = reference length).

It is precision-based, uses clipped counts, accumulates statistics over the whole test corpus, and is 0 without smoothing as soon as one p_{n} is 0.""")

multi(M, "Select all accurate claims about evaluation metrics.",
      [("ROUGE-L is based on the longest common subsequence of candidate and reference, so matched words must be in order but need not be adjacent",
        "A subsequence keeps the order of words but allows gaps, which rewards sentence-level structure without demanding contiguous matches."),
       ("METEOR can give credit for stem and synonym matches, not just exact word matches",
        "Besides exact matches it aligns words that share a stem or are synonyms, and it weights recall more heavily than precision.")],
      [("chrF compares whole-word n-grams, so it gives no credit to a word that merely shares a stem with the reference word",
        "chrF is a character n-gram F-score: a different inflection still shares many character n-grams with the reference word and earns partial credit."),
       ("Perplexity is the standard way to compare the output quality of two machine translation systems",
        "Perplexity scores how well a model predicts the reference tokens; it says nothing directly about generated translations and is not comparable across vocabularies. MT output is compared with BLEU, chrF or human judgements."),
       ("A higher BLEU score reliably shows which of two systems human judges would prefer, so human evaluation adds little",
        "BLEU correlates only imperfectly with human judgements, especially for single sentences and for small score differences. Human evaluation remains the gold standard, although it is slow and costly.")],
      """ROUGE: recall-oriented overlap, used for summarisation (ROUGE-N: n-grams, ROUGE-L: longest common subsequence).

METEOR: unigram matching with stems and synonyms. chrF: character n-gram F-score. Human judgements stay the gold standard, but they are costly.""")

# ------------------------------------------------------------------ computation
multi(M, "A language model assigns the probabilities 1/2, 1/4 and 1/8 to the three tokens of a held-out sentence (each given its own preceding context). Select all statements that are correct.",
      [(T("The perplexity of the sentence is ${ppl:.0f}.", ppl=ppl),
        T("PPL = (1/2 · 1/4 · 1/8)^{−1/3} = ${prod:.0f}^{1/3} = ${ppl:.0f}: the inverse geometric mean of the token probabilities.", prod=1 / math.prod(P), ppl=ppl)),
       (T("The average cross-entropy is ${bits:.0f} bits per token.", bits=bits),
        T("The values −log_{2} p are 1, 2 and 3 bits; their mean is ${bits:.0f}, and 2^{${bits:.0f}} = ${ppl:.0f} recovers the perplexity.", bits=bits, ppl=ppl)),
       (T("The average cross-entropy is about ${nats:.3f} nats per token (= 2 ln 2).", nats=nats),
        T("The same quantity in nats: (ln 2 + ln 4 + ln 8)/3 = ln 4 ≈ ${nats:.3f}, and e^{${nats:.3f}} = ${ppl:.0f} again. Bits and nats give the same perplexity as long as the exponentiation uses the matching base.", nats=nats, ppl=ppl))],
      [(T("The perplexity is (2 + 4 + 8)/3 ≈ ${am:.2f}.", am=am_inv),
        T("That is the arithmetic mean of the inverse probabilities. Perplexity averages in the log domain (a geometric mean); the arithmetic mean always overshoots it, here ${am:.2f} > ${ppl:.0f}.", am=am_inv, ppl=ppl)),
       (T("The perplexity is 1 / ((1/2 + 1/4 + 1/8)/3) ≈ ${ia:.2f}.", ia=inv_am),
        T("That inverts the arithmetic mean of the probabilities (≈ ${m:.3f}). Averaging must be done on log-probabilities; this shortcut undershoots the true value of ${ppl:.0f}.", m=sum(P) / N, ppl=ppl))],
      T("""PPL = exp(−(1/N) ∑ ln p_{i}) = (∏ p_{i})^{−1/N}.

Here the log-probabilities average to −ln 4 (−2 bits), so PPL = ${ppl:.0f}. Rule of thumb: average the **logs**, never the probabilities or their inverses.""", ppl=ppl),
      calc=True)

num(M, T("A language model is evaluated on a held-out sentence of ${n} tokens. The total negative log-likelihood it assigns to the sentence (natural logarithm) is ${L} nats. What is the sentence's perplexity?", n=N_TOK, L=NLL_TOTAL),
    ppl2,
    T("""1. Per-token cross-entropy: ${L} / ${n} = ${ce:.1f} nats.
2. Perplexity = exp(cross-entropy) = e^{${ce:.1f}} ≈ **${p:.2f}**.

Common slips: not dividing by the number of tokens (e^{${L}} ≈ 2.4 × 10^{17}), or exponentiating with base 2 although the loss is in nats (2^{${ce:.0f}} = ${wrong:.0f}).""", L=NLL_TOTAL, n=N_TOK, ce=NLL_TOTAL / N_TOK, p=ppl2, wrong=2 ** (NLL_TOTAL / N_TOK)),
    tol=0.05, hint="two decimal places", disp=T("${p:.2f}", p=ppl2))

num(M, """BLEU's modified (clipped) unigram precision, for this candidate and these two references:

- Candidate: `the the the the the the the`
- Reference 1: `the cat is on the mat`
- Reference 2: `there is a cat on the mat`

What is the clipped unigram precision p_{1} of the candidate?""",
    p1,
    T("""1. The candidate has ${n} tokens, all `the`.
2. `the` occurs 2 times in reference 1 and once in reference 2, so the clip is the maximum, 2.
3. Clipped count = min(${n}, 2) = ${k}, so p_{1} = ${k}/${n} ≈ **${p:.4f}**.

Without clipping the precision would be ${n}/${n} = 1 for a meaningless sentence. Summing the reference counts (2 + 1 = 3) is a common mistake: the clip uses the maximum over references, not the sum.""", n=len(cand), k=clipped, p=p1),
    tol=0.005, hint="a fraction or a decimal", disp=T("${k}/${n} ≈ ${p:.4f}", k=clipped, n=len(cand), p=p1))

single(M, T("BLEU multiplies the geometric mean of its n-gram precisions by a brevity penalty BP that depends on the candidate length c and the reference length r. A system output has c = ${c} tokens and the reference has r = ${r} tokens. What is BP?", c=c_len, r=r_len),
       (T("${bp:.4f}", bp=bp),
        T("Since c ≤ r, BP = exp(1 − r/c) = exp(1 − ${ratio}) = exp(${et}) ≈ ${bp:.4f}.", ratio=r_len / c_len, et=exp_txt, bp=bp)),
       [(T("${x:.4f}", x=bp_ratio),
         T("Used the plain length ratio c/r = ${x:.1f}. BP = exp(1 − r/c) is smaller (${bp:.4f}), and relative to the plain ratio c/r the penalty gets harsher the shorter the candidate is (c/r = 0.5 gives BP ≈ 0.37).", x=bp_ratio, bp=bp)),
        (T("${x:.4f}", x=bp_swapped),
         T("The exponent −(1 − c/r) = −0.2 uses c/r where r/c belongs. The correct exponent is 1 − r/c = ${et}.", et=exp_txt)),
        (T("${x:.4f}", x=bp_inverted),
         "The exponent 1 − c/r = +0.2 inverts the ratio and yields a value above 1. BP can never exceed 1, so such a result is a red flag."),
        ("1.0000 (no penalty)",
         "Assumed BP only penalises candidates that are too long. In fact BP = 1 applies when c > r; a shorter candidate (c < r) is penalised.")],
       T("""BP = 1 if c > r, and exp(1 − r/c) if c ≤ r.

Here c = ${c}, r = ${r}: BP = exp(1 − ${ratio}) = exp(${et}) ≈ **${bp:.4f}**. BP looks only at lengths, and it exists because precision alone would reward very short outputs.""", c=c_len, r=r_len, ratio=r_len / c_len, et=exp_txt, bp=bp),
       calc=True)
