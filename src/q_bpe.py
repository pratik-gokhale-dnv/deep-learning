from collections import Counter

from qdsl import tf, single, multi, num, T

P = "bpe"
EOW = "</w>"

# ---------------------------------------------------------------- calc (a): first merge on a tiny corpus (all numbers computed here)


def _pair_counts(vocab, with_eow=True, weighted=True):
    c = Counter()
    for word, f in vocab.items():
        for a, b in zip(word, word[1:]):
            if with_eow or EOW not in (a, b):
                c[(a, b)] += f if weighted else 1
    return c


def _unique_top(c):
    (p, n), (_, n2) = c.most_common(2)
    assert n > n2, "tie at the top"
    return p, n


CORPUS = {"bears": 6, "horse": 11, "mice": 10, "rats": 6, "seals": 4}
VOCAB = {tuple(w) + (EOW,): f for w, f in CORPUS.items()}
NAME = {tuple(w) + (EOW,): w for w in CORPUS}
PC = _pair_counts(VOCAB)
FIRST, N_FIRST = _unique_top(PC)
TYPE_PAIR, N_TYPE = _unique_top(_pair_counts(VOCAB, weighted=False))  # 'count each word once' mistake
NOEOW_PAIR, N_NOEOW = _unique_top(_pair_counts(VOCAB, with_eow=False))  # 'ignore </w>' mistake
LETTERS = Counter(ch for w, f in CORPUS.items() for ch in w for _ in range(f))
(_l1, N_L1), (_l2, N_L2), (_l3, N_L3) = LETTERS.most_common(3)
assert {_l1, _l2} == {"s", "e"} and N_L1 == N_L2 > N_L3  # 'two most frequent letters' mistake
assert FIRST == ("e", EOW) and TYPE_PAIR == ("s", EOW) and NOEOW_PAIR == ("r", "s")
assert len({FIRST, TYPE_PAIR, NOEOW_PAIR, ("s", "e")}) == 4
SHOWN = [p for p, n in PC.most_common(4)]
REST = max(n for p, n in PC.items() if p not in SHOWN)
assert SHOWN == [("e", EOW), ("r", "s"), ("s", EOW), ("s", "e")] and REST < PC[("s", "e")]


def _breakdown(pair):
    parts = [(NAME[v], f * sum(1 for x in zip(v, v[1:]) if x == pair)) for v, f in VOCAB.items()]
    return " + ".join("%s %d" % (w, n) for w, n in parts if n)


TABLE = "\n".join(["%-7s %-6s %s" % ("word", "count", "symbols")] +
                  ["%-7s %-6d %s" % (w, f, " ".join(w) + " " + EOW) for w, f in CORPUS.items()])
COUNT_TABLE = "\n".join(["%-9s %-6s %s" % ("pair", "count", "where")] +
                        ["%-9s %-6d %s" % (" ".join(p), PC[p], _breakdown(p)) for p in SHOWN] +
                        ["(every other pair occurs at most %d times)" % REST])

# ---------------------------------------------------------------- calc (b): segmenting an unseen word with an ordered merge list
RULES = [("i", "n"), ("in", "g"), ("e", "a"), ("h", "e"), ("ea", "t"), ("ing", EOW)]
WORD = "heating"


def _merge_all(seq, pair):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(seq[i] + seq[i + 1])
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _encode(word, rules):
    seq = list(word) + [EOW]
    for r in rules:
        seq = _merge_all(seq, r)
    return seq


def _encode_by_position(word, rules):  # mistake: always merge the leftmost listed pair, ignoring rule order
    seq = list(word) + [EOW]
    while True:
        for i in range(len(seq) - 1):
            if (seq[i], seq[i + 1]) in rules:
                seq = seq[:i] + [seq[i] + seq[i + 1]] + seq[i + 2:]
                break
        else:
            return seq


def _encode_one_pass(word, rules):  # mistake: merged symbols are never merged again
    base = set(word) | {EOW}
    seq = list(word) + [EOW]
    for r in rules:
        if r[0] in base and r[1] in base:
            seq = _merge_all(seq, r)
    return seq


def _encode_no_marker(word, rules):  # mistake: rules that involve the end-of-word symbol are skipped
    seq = list(word) + [EOW]
    for r in rules:
        if EOW not in r:
            seq = _merge_all(seq, r)
    return seq


SEG = _encode(WORD, RULES)
SEG_POS = _encode_by_position(WORD, RULES)
SEG_ONE = _encode_one_pass(WORD, RULES)
SEG_NOM = _encode_no_marker(WORD, RULES)
assert SEG == ["h", "eat", "ing" + EOW]
assert SEG_POS == ["he", "a", "t", "ing" + EOW]
assert SEG_ONE == ["h", "ea", "t", "in", "g", EOW]
assert SEG_NOM == ["h", "eat", "ing", EOW]
assert len({tuple(s) for s in (SEG, SEG_POS, SEG_ONE, SEG_NOM)}) == 4


def _show(seq):
    return " ".join("`%s`" % t for t in seq)


RULE_TABLE = "\n".join("%d.  %-4s + %-5s ->  %s" % (i, a, b, a + b) for i, (a, b) in enumerate(RULES, 1))
_seq, _lines = list(WORD) + [EOW], ["start   " + " ".join(list(WORD) + [EOW])]
for _k, _r in enumerate(RULES, 1):
    _new = _merge_all(_seq, _r)
    _lines.append("rule %d  %s%s" % (_k, " ".join(_new), "" if _new != _seq else "   (no change)"))
    _seq = _new
TRACE = "\n".join(_lines)

# ---------------------------------------------------------------- true / false
tf(P, "BPE began as a data-compression method: it repeatedly replaced the most frequent pair of adjacent bytes in a file by a byte value that did not occur in the data. Its use for subword segmentation in neural machine translation is an adaptation of that idea.", True,
   """Gage (1994) described BPE as a compression scheme: find the most frequent byte pair, replace it with an unused byte, repeat, and store the replacement table.

Sennrich et al. (2016) transferred the same greedy idea from bytes to characters of words: the "replacement table" becomes an ordered list of merge rules, and the symbols produced by the merges form the subword vocabulary.""")

tf(P, "A byte-level BPE tokeniser, as used in GPT-2, starts from the 256 possible byte values, so every input string can be encoded and no unknown-word symbol is needed.", True,
   """Any text is a sequence of bytes (for example in UTF-8), and all 256 byte values are in the base vocabulary. Merges only create longer symbols, so the worst case is falling back to individual bytes.

The price is that rare characters, such as an emoji or a character from a rare script, can cost several byte tokens each. A character-level BPE has no such guarantee: a character never seen in training is out of vocabulary.""")

tf(P, "The subword units that BPE produces are linguistic morphemes (stems, prefixes and suffixes), because merging frequent pairs recovers the meaningful parts of words.", False,
   """BPE is purely statistical and contains no linguistic knowledge: it merges whatever pair is most frequent. Sometimes the result looks like a morpheme (a frequent suffix such as `ing</w>`), but many units are arbitrary letter groups that cross morpheme boundaries.

Morphologically motivated segmentation needs a separate tool such as Morfessor or a linguistic analyser.""")

# ---------------------------------------------------------------- single answer
single(P, "Which description matches one iteration of BPE vocabulary learning (Sennrich et al., 2016)?",
       ("Count adjacent symbol pairs weighted by word frequency, then replace every occurrence of the most frequent pair by one new symbol",
        "This is the greedy core: pick the most frequent pair, record it as a merge rule, update the corpus, and repeat until the chosen number of merges has been done."),
       [("Add the most frequent whole word that is not yet in the vocabulary as a new symbol, so that frequent words need no splitting",
         "That extends a word-level vocabulary. BPE never adds words directly; each new symbol is the concatenation of two symbols that already exist."),
        ("Delete the vocabulary symbols whose removal would reduce the likelihood of the corpus the least, so that the vocabulary shrinks at every iteration",
         "That is the pruning step of the unigram language-model tokeniser, which starts from a large vocabulary and removes the least useful symbols (a fraction of them per round) until the target size is reached. BPE starts small and grows."),
        ("Replace every symbol that occurs fewer than a threshold number of times by a special unknown symbol",
         "Frequency cut-offs with an unknown token are the word-level baseline that BPE was designed to avoid: rare strings stay representable as sequences of known symbols.")],
       """BPE training = repeat: (1) count adjacent symbol pairs, weighting each word's pairs by its corpus frequency; (2) merge the most frequent pair everywhere; (3) add the new symbol to the vocabulary and store the rule. The ordered list of rules is the tokeniser.""")

single(P, "A BPE tokeniser starts from 80 base symbols (the characters plus the end-of-word symbol) and is trained for 32,000 merge operations. How large is the resulting symbol inventory, counting the base symbols and every symbol created by a merge?",
       ("32,080: the 80 base symbols plus one new symbol for each of the 32,000 merges",
        "Every merge adds exactly one new symbol and removes none, because the two parts stay available for places where the merged pair does not occur."),
       [("32,000: only the merged symbols count, because merging replaces the characters it combines",
         "The characters remain in the inventory: they are still needed wherever the merged pair does not occur, and they let any word made of known characters be spelled out."),
        ("Fewer than 80: every merge combines two symbols into one, so the inventory shrinks during training",
         "Merging shortens the token sequences of the text, but it enlarges the inventory. This is the usual confusion between tokens per sentence and symbols in the vocabulary."),
        ("It equals the number of distinct word types in the training corpus, as in a word-level vocabulary",
         "That would be the vocabulary of a word-level model. A BPE vocabulary is set by the alphabet and the number of merges, independently of how many word types the corpus contains.")],
       "Vocabulary size = |base alphabet| + number of merges. The number of merges is the hyperparameter that sets the size, here 80 + 32,000 = 32,080. The number of tokens needed for a given text goes down as the merges accumulate; the inventory goes up.")

single(P, "A team retrains the same Transformer translation system after raising the BPE vocabulary from 8,000 to 64,000 symbols (same data, same architecture otherwise). Which effect is the most plausible?",
       ("Sentences are split into fewer tokens on average, but the embedding table and softmax are larger and many tokens occur only rarely",
        "More merges produce longer symbols, so the same text needs fewer tokens; the price is more parameters in the vocabulary-sized layers and a long tail of rare tokens."),
       [("Sentences are split into more tokens on average, since a bigger vocabulary spreads the same text over more distinct symbols",
         "The direction is wrong: each extra merge fuses neighbouring symbols, so the token count per sentence can only stay the same or fall."),
        ("The number of parameters is unchanged, because BPE only changes how the text is split and not the network",
         "The input embedding matrix and the output projection both have one row (or column) per vocabulary symbol, so they grow with the vocabulary."),
        ("The model can now represent words that it could not represent before, because the 8,000-symbol vocabulary left some words out of reach",
         "Even the small vocabulary can spell out any word from its characters. A larger one merely writes frequent strings with fewer symbols.")],
       "Vocabulary size is a trade-off. Large: shorter sequences (cheaper attention, fewer decoding steps) but a bigger embedding/softmax and many rare symbols with poorly trained embeddings. Small: longer sequences and a smaller softmax, but each symbol is seen more often.")

single(P, "A Transformer was trained with a 32,000-symbol BPE vocabulary. At test time the input contains a rare surname that never occurred in the training data, although all of its letters did. What happens to it?",
       ("It is split into known subword symbols, down to single letters in the worst case, so it can still be processed",
        "The learned merges are applied to the unseen word; whatever cannot be merged stays as shorter known symbols, so there is no need for a whole-word unknown token."),
       [("It is replaced by a single unknown-word symbol, because BPE can only represent whole words that were seen during training",
         "That describes a word-level vocabulary. BPE is an open-vocabulary method: unseen words are composed from smaller symbols instead of being mapped to one unknown token."),
        ("It is added to the vocabulary on the fly as a new symbol, and an embedding is initialised for it",
         "The vocabulary and the embedding table are fixed after training, so no new rows are created at test time. The surname is expressed with symbols that already exist."),
        ("It is replaced by the closest known word in embedding space before the sentence is processed",
         "BPE does no nearest-neighbour replacement. It segments the original string deterministically using the merge rules.")],
       "BPE keeps the rare-word problem manageable: a rare or new word is decomposed into frequent subwords, down to single characters. A name like this will typically be split into several pieces, and a model can learn to copy or transliterate such pieces. Only a character that never occurred in training would still be out of vocabulary.")

single(P, "What mainly distinguishes SentencePiece (Kudo and Richardson, 2018) from the word-based BPE preprocessing of Sennrich et al.?",
       ("It is trained and applied directly on raw text, treating whitespace as an ordinary symbol, so no language-specific word splitter is needed",
        "Spaces are encoded as a visible symbol (often ▁) inside the pieces, which makes the tokenisation language-independent and lets the (Unicode-normalised) text be restored exactly. SentencePiece offers both BPE and unigram-LM segmentation."),
       [("It learns the segmentation jointly with the translation model, by back-propagating the translation loss through the choice of subword boundaries",
         "SentencePiece is trained beforehand and separately, like BPE; the translation model then just uses the fixed vocabulary."),
        ("It replaces the discrete subword table by learned continuous embeddings of single characters that are composed into word vectors",
         "SentencePiece still produces a discrete vocabulary of pieces, and the model learns one embedding per piece as usual."),
        ("It produces only whole-word tokens, taken from a dictionary of whitespace-separated words in the training data",
         "Whole-word dictionaries are what subword methods try to move away from. SentencePiece produces subword pieces, and it also works for languages written without spaces (Chinese, Japanese), because it does not rely on whitespace to find words.")],
       "Classic BPE preprocessing pre-tokenises with a language-specific tool, then counts pairs inside words. SentencePiece skips this step by treating the input as a stream of Unicode characters, with whitespace as a symbol; decoding therefore needs no detokeniser.")

single(P, "Sennrich et al. compare independent BPE (one merge list per language) with joint BPE (one merge list learned on the source and target training text together). What is the main advantage of joint BPE for a pair like English–German?",
       ("Names and cognates are segmented more consistently on both sides, which makes it easier to learn the mapping between subword units",
        "With independent BPE a name can be cut differently in the two languages, which hides the correspondence. The Transformer paper's EN–DE experiments, for instance, used one shared BPE vocabulary of about 37,000 tokens."),
       [("Every subword that the decoder can output has been seen in the target-language training text",
         "Sennrich et al. credit independent BPE with this guarantee. With a joint merge list a symbol may come from the source side alone and never occur in the target text."),
        ("It produces shorter token sequences than independent BPE, because the merges are learned from twice as much text",
         "Sennrich et al. describe independent BPE as the more compact option in text and vocabulary size; what joint BPE improves is the consistency between source and target segmentation."),
        ("Source and target sentences end up with the same number of tokens, which simplifies alignment",
         "Different languages express the same content with different numbers of words and subwords; no BPE variant equalises sentence lengths.")],
       "Joint BPE learns one merge list on the concatenated source and target text and applies it to both sides; it works best when the languages share a script. Independent BPE (one list per language) is more compact and ensures that each symbol was seen in its own language's text, at the price of less consistent segmentation of names and cognates.")

# ---------------------------------------------------------------- calculations
single(P, T("""A BPE tokeniser is trained on a corpus with the word counts below. Each word is written as its characters followed by the end-of-word symbol `</w>`, which counts as a symbol of its own.

```
${table}
```

According to the BPE algorithm, which pair of symbols is merged first?""", table=TABLE),
       ("`e` and `</w>`: it has the highest total count once each word's pairs are weighted by that word's frequency",
        T("Correct. The pair occurs once in horse and once in mice: ${bd} = ${n}, ahead of the runner-up (r, s) with ${n2}.", bd=_breakdown(FIRST), n=N_FIRST, n2=N_NOEOW)),
       [("`s` and `</w>`: the plural ending, which closes the largest number of different words (bears, rats and seals)",
         T("This counts each word type once, ignoring the corpus frequencies. (`s`, `</w>`) is indeed found in the most distinct words (${nt}), but weighted by frequency it occurs only ${bd} = ${n} times, fewer than ${best}.", nt=N_TYPE, bd=_breakdown(("s", EOW)), n=PC[("s", EOW)], best=N_FIRST)),
        ("`r` and `s`: the most frequent pair of neighbouring letters inside the words, ignoring the end-of-word symbol",
         T("Dropping `</w>` makes (r, s) the winner with ${bd} = ${n}. But `</w>` is a symbol like any other, so pairs that contain it compete too, and (`e`, `</w>`) reaches ${best}.", bd=_breakdown(("r", "s")), n=N_NOEOW, best=N_FIRST)),
        ("`s` and `e`: they are the two most frequent letters in the corpus",
         T("s and e do have the highest letter counts (${c} each), but BPE merges the most frequent adjacent pair, not the two most frequent letters. The pair (s, e) occurs only ${bd} = ${n} times.", c=N_L1, bd=_breakdown(("s", "e")), n=PC[("s", "e")]))],
       T("""BPE counts every adjacent pair of symbols, adding the frequency of the word once for each occurrence of the pair in it:

```
${counts}
```

The maximum is unique, so (e, `</w>`) is merged and a new symbol `e</w>` enters the vocabulary. The counts are then updated (for example, horse and mice now end in the symbol `e</w>`) and the next merge is chosen the same way.""", counts=COUNT_TABLE),
       calc=True)

single(P, T("""A BPE tokeniser has learned the merge rules below, listed in the order in which they were learned (rule 1 first). The end-of-word symbol `</w>` is an ordinary symbol and takes part in merges.

```
${rules}
```

The word "${word}" did not occur in the training corpus. How is it segmented by the tokeniser?""", rules=RULE_TABLE, word=WORD),
       (_show(SEG),
        "Correct. The rules are applied in learned order, each to the current symbol sequence; rule 4 no longer applies because the e has already been fused into ea, and rules 5 and 6 combine already-merged symbols."),
       [(_show(SEG_POS),
         "This applies rules by position (the leftmost matching pair, h + e) instead of by learned order. Rule 3 (e + a) comes before rule 4, so e joins a first and h + e never matches."),
        (_show(SEG_ONE),
         "Here merged symbols were never merged again. Rules 2, 5 and 6 (in + g, ea + t, ing + `</w>`) combine symbols that earlier rules created, and they must be applied too."),
        (_show(SEG_NOM),
         "The end-of-word symbol was ignored. `</w>` is a symbol like the others, so rule 6 fuses ing with it into the single symbol `ing</w>`.")],
       T("""Apply the rules one after another to the current symbol sequence:

```
${trace}
```

Rule 4 (h + e) cannot fire because rule 3 has already fused the `e` with the `a`: earlier rules take priority over later ones and over left-to-right position. Result: ${seg}. The trailing `</w>` inside `ing</w>` marks that the word ends here.""", trace=TRACE, seg=_show(SEG)),
       calc=True)

# ---------------------------------------------------------------- multiple answer
multi(P, "Sennrich et al. learn the BPE merge list from a table that maps each word to its corpus count. Which statements about this training procedure are correct? Select all that apply.",
      [("Each word contributes its pairs once per occurrence in the corpus, so a pair inside one very frequent word can outrank a pair spread over many rare words",
        "Correct. The counts come from the word-frequency table: a word seen 1,000 times adds 1,000 to each pair it contains (2,000 if the pair occurs twice inside the word)."),
       ("After a pair has been merged, the pair counts have to be updated, because the new symbol forms new neighbouring pairs and some old pairs disappear",
        "Correct. Merging (a, b) takes occurrences away from pairs such as (x, a) and (b, y) and creates the new pairs (x, ab) and (ab, y), so the ranking changes."),
       ("The number of merges is chosen in advance and, together with the base alphabet, determines the size of the final vocabulary",
        "Correct. Training stops after the chosen number of merges; Sennrich et al. note that this is the only hyperparameter of the algorithm.")],
      [("Pairs are also counted across word boundaries, so a frequent word pair such as “of the” can end up as a single symbol",
        "In this algorithm the corpus is a dictionary of words and symbols are only paired inside a word, so a merge never spans two words."),
       ("All merges can be read off the initial pair counts: sort the pairs by frequency once and apply them in that order",
        "Each merge changes the statistics. A pair that ranked high at the start can lose most of its count (for instance when one of its symbols has been absorbed into a new symbol) and new pairs can overtake it, so the counts must be refreshed after every merge.")],
      "Training loop: (1) count adjacent symbol pairs inside words, weighting each word by its corpus frequency; (2) merge the most frequent pair everywhere and store it as the next rule; (3) update the counts and repeat. The ordered rule list is the trained tokeniser. Vocabulary size = size of the initial symbol set + number of merges.")

multi(P, "Why was BPE adopted for neural machine translation instead of a word-level vocabulary of the most frequent words plus an unknown-word symbol? Select all that apply.",
      [("Rare words, names and compounds can be written as sequences of known pieces, so the model can still read and produce them",
        "Correct. This open-vocabulary property is the main motivation: whatever is not a frequent word can still be spelled with smaller symbols, in the worst case single characters."),
       ("Related word forms often share pieces such as a stem or an ending, so what is learned for one form partly carries over to the others",
        "Correct. Overlapping segmentations let inflected and derived forms share embeddings and statistics. BPE has no linguistic knowledge, so this only happens when the forms share frequent substrings.")],
      [("Subword sequences are shorter than word sequences, so the encoder and decoder run for fewer time steps and training is cheaper",
        "Splitting words into pieces can only lengthen a sequence. The open vocabulary is paid for with more tokens per sentence."),
       ("A BPE vocabulary can be built from the alphabet of the language alone, so no training text is needed to obtain the merges",
        "The merges are chosen by counting pairs in a training corpus; without text there is nothing to rank. The alphabet only provides the starting symbols."),
       ("The decoder no longer needs a fixed output vocabulary, because it can create new symbols for unseen words at test time",
        "The output layer is a softmax over a fixed inventory of subword symbols set at training time. Unseen words are handled by combining existing symbols, not by creating new ones.")],
      "A word-level model needs a huge softmax to cover rare words and still has to map the rest to `<unk>`. BPE keeps the inventory at tens of thousands of symbols, yet any word made of known characters can be spelled out. The price is longer sequences (more steps, more attention cost), and the segmentation is statistical rather than linguistic.")

multi(P, "A BPE tokeniser in the style of Sennrich et al. (each word is written as characters plus an end-of-word symbol) has been trained and its ordered list of merge rules is stored. Which statements about segmenting a new text with it, without any randomisation, are correct? Select all that apply.",
      [("A word is split into characters plus the end-of-word symbol, and the rules are then applied in the order in which they were learned",
        "Correct. The rule order encodes priority: a rule learned early (a pair that was frequent in the training corpus) is applied before the later ones."),
       ("A given word always receives the same segmentation, whatever sentence it appears in",
        "Correct. The result is a deterministic function of the word and the merge list; context plays no role at this stage."),
       ("The training corpus is no longer needed: the ordered merge list and the base symbols are enough",
        "Correct. All corpus statistics have been absorbed into the ranking of the rules, so encoding just replays them."),
       ("A word that never occurred in the training corpus is segmented by exactly the same procedure as a word seen in training; there is no special case",
        "Correct. Rules fire wherever their pair occurs in the word, and whatever cannot be merged remains as smaller known symbols.")],
      [("Pair frequencies are recounted on the text being segmented, and the pair that is most frequent in that text is merged first",
        "No counting happens at encoding time. The merge order was fixed during training, and recounting would make a word's segmentation depend on the document it appears in."),
       ("The order of the rules does not matter, because each rule is just a replacement of one pair of symbols by one symbol",
        "Rules overlap on shared symbols. With the rules (e, a) followed by (h, e), the word hea becomes h + ea; with the two rules in the opposite order it becomes he + a. The order is part of the model.")],
      "Encoding is a replay: split into characters (plus `</w>`), then apply rule 1, rule 2, and so on to the current symbol sequence. There is no counting and no corpus; the ordered rule list is the whole model. Unseen words are no special case, since they are spelled out with whatever symbols the rules can build.")

multi(P, "Which statements about subword tokenisers related to BPE are correct? Select all that apply.",
      [("The unigram language-model tokeniser starts from a large candidate vocabulary and prunes the pieces whose removal lowers the data likelihood least",
        "Correct. It works top-down: a big seed vocabulary is pruned to the target size, whereas BPE works bottom-up by merging characters."),
       ("WordPiece, the tokeniser family used by BERT, picks merges by their likelihood gain under a language model, not by raw pair frequency",
        "Correct. Merging two symbols that nearly always occur together gains more likelihood than merging two very frequent symbols that are only often neighbours, so the choice can differ from raw frequency.")],
      [("The unigram language-model tokeniser builds its vocabulary bottom-up by merging the most frequent adjacent pair, exactly as BPE does",
        "That description is BPE. The unigram model starts from a large inventory and prunes it, and it scores whole segmentations with a probabilistic model."),
       ("At test time WordPiece segments a word by replaying its merge list in order, so its encoding step is identical to BPE's",
        "A WordPiece tokeniser keeps a vocabulary and segments greedily by longest match: it takes the longest vocabulary piece at the start of the word and continues with the remainder."),
       ("Standard BPE encoding is probabilistic: the same word may be segmented differently each time it is encoded",
        "With a fixed merge list a word has exactly one segmentation. Randomised segmentations belong to variants such as sampling from a unigram model or BPE-dropout, which are used on purpose to regularise training.")],
      "BPE: bottom-up, merges chosen by frequency, deterministic encoding by replaying the merge list. Unigram LM: top-down pruning of a large vocabulary, probabilistic segmentation (the best one, or a sampled one). WordPiece: bottom-up like BPE, but merges are scored by likelihood gain; encoding is greedy longest-match-first.")

multi(P, "In the BPE algorithm of Sennrich et al., an end-of-word symbol such as `</w>` is appended to every word before pair counting starts. Which statements about this symbol are correct? Select all that apply.",
      [("It lets the algorithm tell a word-final piece such as `est</w>` apart from the same letters inside a word, as at the start of “estimate”",
        "Correct. Without the marker, the suffix est of “newest” and the prefix est of “estimate” would be one and the same symbol, although they play different roles."),
       ("It is an ordinary symbol for pair counting, so it can be merged with the preceding symbol to form word-final symbols",
        "Correct. A pair such as (est, `</w>`) is counted like any other pair and can become the new symbol `est</w>`."),
       ("Because the word boundary is kept inside the symbols, a sequence of pieces can be turned back into the original words without extra information",
        "Correct. A piece ending in `</w>` closes a word and pieces without it continue the same word, so detokenising is just concatenation and splitting at the markers.")],
      [("It is the symbol that the decoder generates at the end of a translation, so it doubles as the end-of-sentence token",
        "These are different symbols. `</w>` closes a single word inside the token stream, whereas an end-of-sentence token such as `</s>` closes the whole sequence."),
       ("It can be left out without changing the merges that are learned, because the spaces in the text already mark the word boundaries",
        "Pairs are counted on each word's symbol sequence. Without the marker, pairs such as (e, `</w>`) cannot occur, so the learned merges change, and the pieces no longer carry any boundary information.")],
      "The marker is appended to every word as an extra symbol: `l o w </w>`. It takes part in counting and merging like any other symbol, so merges such as (er, `</w>`) produce word-final pieces. This keeps word-final and word-internal occurrences of the same letters apart and preserves word boundaries for detokenisation. It has nothing to do with the end-of-sentence token.")

multi(P, "Consider three ways of tokenising the same corpus: whole words, BPE subwords with about 30,000 merges, and single characters. Which statements are correct? Select all that apply.",
      [("On average, the number of tokens per sentence is smallest for words, in between for BPE subwords, and largest for characters",
        "Correct. Every word gives at least one subword and every subword consists of at least one character, so words ≤ subwords ≤ characters."),
       ("The output softmax is much smaller for BPE than for a word-level model that covers nearly all word types, but larger than for characters",
        "Correct. Typical sizes are about a hundred symbols for Latin-script characters, tens of thousands for BPE and hundreds of thousands or more for full word coverage."),
       ("Frequent words end up as single symbols, while rare words are spelled with several pieces, so the number of pieces per word varies",
        "Correct. After enough merges frequent words are single symbols, whereas rare words are only partly merged.")],
      [("Self-attention is cheaper on BPE sequences than on word sequences, because each subword unit is shorter than a word",
        "The cost depends on the number of tokens n (about n² · d per layer), not on how long each token is. Subword sequences are longer than word sequences, so attention gets more expensive."),
       ("As the number of merges grows without limit, BPE tokenisation approaches character-level tokenisation of the text",
        "It is the other way round: zero merges is character-level, and with very many merges the frequent words become single symbols, which is word-level tokenisation."),
       ("A character-level model is just as exposed to unknown words as a word-level model, since any new word is missing from its vocabulary",
        "A new word is still a sequence of known characters, so a character-level model can read it; only characters never seen in training would be unknown.")],
      "By sequence length: words ≤ BPE subwords ≤ characters (strict for typical text). By vocabulary size: characters < BPE < words. The number of merges moves BPE along this scale (zero merges = characters, very many merges = words). Attention cost grows with the number of tokens, while a larger vocabulary costs more in the embedding and softmax layers.")
