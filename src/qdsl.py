"""Tiny DSL used to author the quiz bank.

Text fields use a mini-markup that build.py converts to HTML:
    _{...}  subscript          ^{...}  superscript (both nest)
    `code`  inline code        **bold**
    blank line = new paragraph; lines starting with "- " or "1. " = lists
    ``` fenced blocks ```      = preformatted text
Use T("... ${name} ${n:,} ...", name=..., n=...) to inject computed values
without clashing with the literal braces used for sub/superscripts.
"""
import re

TOPICS = [
    ("rnn", "RNNs & LSTMs"),
    ("clm", "Conditional LMs & Teacher/Student Forcing"),
    ("metrics", "Language Metrics"),
    ("kd", "Knowledge Distillation"),
    ("encdec", "Encoder–Decoder (Seq2Seq)"),
    ("beam", "Beam Search"),
    ("attn", "Neural Attention"),
    ("transformer", "Transformers"),
    ("bpe", "Byte Pair Encoding"),
    ("sizes", "Attention Matrix Sizes"),
    ("bptt", "Backprop Through Time"),
]

BANK = []

_PH = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def T(s, **kw):
    """Substitute ${name} / ${name:fmt} placeholders (leaves other braces alone)."""
    return _PH.sub(lambda m: format(kw[m.group(1)], m.group(2) or ""), s)


def tf(topic, stmt, answer, exp):
    BANK.append(dict(topic=topic, cat="tf", mode="tf", q=stmt, answer=bool(answer), exp=exp))


def single(topic, q, right, wrong, exp, calc=False):
    """right = (text, why); wrong = [(text, why), ...]"""
    BANK.append(dict(topic=topic, cat="calc" if calc else "single", mode="single",
                     q=q, right=[right], wrong=list(wrong), exp=exp))


def multi(topic, q, right, wrong, exp, calc=False):
    """right = [(text, why), ...]; wrong = [(text, why), ...]"""
    BANK.append(dict(topic=topic, cat="calc" if calc else "multi", mode="multi",
                     q=q, right=list(right), wrong=list(wrong), exp=exp))


def num(topic, q, ans, exp, tol=0.5, rel=0.0, hint="", disp=None):
    """Numeric-entry computation question (always category 'calc').

    Accepted if |x - ans| <= max(tol, rel*|ans|).
    """
    BANK.append(dict(topic=topic, cat="calc", mode="num", q=q, ans=ans, tol=tol,
                     rel=rel, hint=hint, disp=disp, exp=exp))
