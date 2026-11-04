# Deep Learning – exam-prep quiz

An interactive quiz with **200 questions** for revising sequence modelling: RNNs and LSTMs, conditional language models, perplexity and BLEU, knowledge distillation, encoder–decoder models, beam search, attention, the Transformer and byte pair encoding.

**Open [`index.html`](index.html) in any modern browser.** It is a single self-contained file: no server, no internet connection and no dependencies.

## What is inside

| Question type | Count |
|---|---:|
| True / False | 40 |
| Single-answer MCQ | 66 |
| Multiple-answer MCQ | 54 |
| Computation (17 type-in numbers + 23 worked multiple-choice calculations) | 40 |
| **Total** | **200** |

About 20 % of the questions are computations and the rest are conceptual.

| Topic | Questions |
|---|---:|
| RNNs & LSTMs (update rules, gating, vanishing/exploding gradients) | 25 |
| Conditional LMs & teacher/student forcing | 15 |
| Language metrics (perplexity, BLEU) | 15 |
| Knowledge distillation | 12 |
| Encoder–decoder (seq2seq) | 15 |
| Beam search | 14 |
| Neural attention (additive, dot-product, self-, cross-, multi-head) | 25 |
| Transformers (architecture, positional encoding, complexity) | 32 |
| Byte pair encoding | 17 |
| Attention matrix sizes (encoder, decoder, cross-attention) | 18 |
| Backprop through time for a simple RNN | 12 |

## Features

- **See why**: after you check an answer, every option of a single- or multiple-answer question is marked correct or incorrect and comes with its own explanation. True/False questions get a full explanation, and computations a step-by-step worked solution.
- **Partial credit** on multiple-answer questions, with a count of how many correct and incorrect options you chose.
- **Numeric answers** accept decimals, fractions (`3/8`), scientific notation (`1.5e-3`) and thousands separators, within a small tolerance.
- **Show answer** and **Try again** on every question. Optionally turn on instant feedback (Filters → Feedback) to have True/False and single-answer questions graded the moment you pick an option.
- **Filters** by topic, question type and status, plus optional shuffling of question and answer order.
- **Results screen** by topic and by type, with a one-click "Practice the ones I missed".
- **Keyboard shortcuts**: `1`–`8` or `a`–`h` select an option, `Enter` checks or moves on, `n` / `p` go to the next / previous question.
- Dark mode, mobile friendly, screen-reader labels.
- Progress is saved in your browser (`localStorage`), so you can close the page and continue later. **Reset** clears it.

## Scope and sources

The questions follow the exam topic outline (conceptual questions plus the attention-matrix-size and BPTT computations); they were not written from the lecture slides themselves, so check terminology and conventions against your course material. Facts come from the original papers: Bahdanau et al. 2015 and Luong et al. 2015 (attention), Vaswani et al. 2017 (Transformer), Hinton et al. 2015 (distillation), Sennrich et al. 2016 (BPE), Pascanu et al. 2013 (gradient clipping) and Papineni et al. 2002 (BLEU). Where textbooks and libraries differ (for example the GRU gate convention or where layer normalisation sits), the question text says which convention it uses.

## Rebuilding the quiz

`index.html` is generated from the question modules in [`src/`](src). Edit a `src/q_*.py` module (one per topic) and run

```bash
python3 src/build.py
```

The script needs only the Python standard library. It validates every question (answer keys, one rationale per option, balanced markup, no duplicate text) and rewrites `index.html`. Numbers for perplexity and BLEU, knowledge distillation, beam search and BPE merges are computed inside the modules and checked with assertions, so the figures in the questions and in the explanations stay consistent.
