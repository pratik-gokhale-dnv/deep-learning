#!/usr/bin/env python3
"""Assemble the quiz bank, validate it, and render index.html from template.html.

Usage (from the repository root):  python3 src/build.py
Environment overrides: QUIZ_OUT (output path), QUIZ_MODULES (comma-separated module subset).
"""
import html.parser
import importlib
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qdsl  # noqa: E402

MODULES = ["q_rnn", "q_clm", "q_metrics", "q_kd", "q_encdec", "q_beam", "q_attn",
           "q_transformer", "q_bpe", "q_sizes", "q_bptt"]
if os.environ.get("QUIZ_MODULES"):  # partial build for checking individual modules
    MODULES = os.environ["QUIZ_MODULES"].split(",")
OUT = os.environ.get("QUIZ_OUT", os.path.join(os.path.dirname(HERE), "index.html"))
TARGET = {"tf": 40, "single": 66, "multi": 54, "calc": 40}
SEED = 20261104


class MarkupError(ValueError):
    pass


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _scripts(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            depth, j = 0, i + 1
            while j < len(s):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= len(s):
                raise MarkupError("unbalanced braces: " + s)
            tag = "sub" if c == "_" else "sup"
            out.append("<%s>%s</%s>" % (tag, _scripts(s[i + 2:j]), tag))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def inline(s):
    parts = s.split("`")
    if len(parts) % 2 == 0:
        raise MarkupError("odd number of backticks: " + s)
    out = []
    for k, p in enumerate(parts):
        if k % 2:
            out.append("<code>%s</code>" % esc(p))
        else:
            t = _scripts(esc(p))
            t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
            t = re.sub(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])", r"<em>\1</em>", t)
            out.append(t)
    return "".join(out)


_UL = re.compile(r"^\s*- ")
_OL = re.compile(r"^\s*\d+\. ")


def block(s):
    lines = s.strip("\n").split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j, buf = i + 1, []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            if j >= len(lines):
                raise MarkupError("unterminated fence: " + s[:80])
            out.append("<pre>%s</pre>" % esc("\n".join(buf)))
            i = j + 1
        elif _UL.match(ln) or _OL.match(ln):
            pat, tag = (_UL, "ul") if _UL.match(ln) else (_OL, "ol")
            items = []
            while i < len(lines) and pat.match(lines[i]):
                items.append(inline(pat.sub("", lines[i], count=1)))
                i += 1
            out.append("<%s>%s</%s>" % (tag, "".join("<li>%s</li>" % x for x in items), tag))
        elif not ln.strip():
            i += 1
        else:
            buf = [ln]
            i += 1
            while (i < len(lines) and lines[i].strip() and not lines[i].startswith("```")
                   and not _UL.match(lines[i]) and not _OL.match(lines[i])):
                buf.append(lines[i])
                i += 1
            out.append("<p>%s</p>" % inline(" ".join(x.strip() for x in buf)))
    return "".join(out)


def balanced_positions(k, rng):
    while True:
        perm = list(range(k))
        rng.shuffle(perm)
        for p in perm:
            yield p


class _Tags(html.parser.HTMLParser):
    VOID = {"br"}

    def __init__(self):
        super().__init__()
        self.stack, self.err = [], None

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.err = "mismatched </%s>" % tag


def html_ok(fragment):
    p = _Tags()
    p.feed(fragment)
    p.close()
    return p.err is None and not p.stack


def build():
    for m in MODULES:
        importlib.import_module(m)
    rng = random.Random(SEED)
    pos_gen = {}
    questions, errors = [], []
    for qid, raw in enumerate(qdsl.BANK, start=1):
        tag = "#%d [%s/%s] %s" % (qid, raw["topic"], raw["mode"], raw["q"][:60].replace("\n", " "))
        try:
            qtext = raw["q"]
            if raw["mode"] == "multi":  # the UI already prints "Select all that apply."
                qtext = re.sub(r"\s*\(?\bSelect all that apply\.?\)?\s*$", "", qtext, flags=re.I)
            q = {"id": qid, "topic": raw["topic"], "cat": raw["cat"], "mode": raw["mode"],
                 "q": block(qtext), "exp": block(raw["exp"])}
            if raw["mode"] == "tf":
                q["opts"] = [{"t": "True", "c": raw["answer"]}, {"t": "False", "c": not raw["answer"]}]
            elif raw["mode"] in ("single", "multi"):
                items = [(inline(t), True, inline(w)) for t, w in raw["right"]] + \
                        [(inline(t), False, inline(w)) for t, w in raw["wrong"]]
                raw_texts = [t for t, _ in raw["right"]] + [t for t, _ in raw["wrong"]]
                if len({t.strip().lower() for t in raw_texts}) != len(raw_texts):
                    errors.append(tag + ": duplicate option text")
                correct = [it for it in items if it[1]]
                wrong = [it for it in items if not it[1]]
                rng.shuffle(wrong)
                k = len(items)
                if raw["mode"] == "single":
                    gen = pos_gen.setdefault(k, balanced_positions(k, rng))
                    p = next(gen)
                    slots = wrong[:]
                    slots.insert(p, correct[0])
                else:
                    slots = items[:]
                    rng.shuffle(slots)
                q["opts"] = [{"t": t, "c": c, "w": w} for t, c, w in slots]
            elif raw["mode"] == "num":
                q.update(ans=raw["ans"], tol=raw["tol"], rel=raw["rel"], hint=raw["hint"],
                         disp=inline(raw["disp"]) if raw["disp"] else None)
                q["opts"] = []
            questions.append(q)
        except Exception as e:  # noqa: BLE001
            errors.append("%s: %s" % (tag, e))
    return questions, errors


FORBIDDEN = [r"all of the above", r"none of the above", r"\boption [A-F]\b", r"\([A-F]\)", r"\bchoice [A-F]\b"]


def validate(questions, errors, warnings):
    seen = Counter(q["q"] for q in questions)
    for q in questions:
        tag = "#%d [%s/%s]" % (q["id"], q["topic"], q["mode"])
        if seen[q["q"]] > 1:
            errors.append(tag + ": duplicate question text")
        if q["topic"] not in dict(qdsl.TOPICS):
            errors.append(tag + ": unknown topic")
        mode, opts = q["mode"], q["opts"]
        ncorrect = sum(1 for o in opts if o["c"])
        if mode == "tf" and (len(opts) != 2 or ncorrect != 1):
            errors.append(tag + ": bad T/F options")
        if mode == "single" and (not 3 <= len(opts) <= 5 or ncorrect != 1):
            errors.append(tag + ": single needs 3-5 options and exactly 1 correct")
        if mode == "multi" and (not 4 <= len(opts) <= 6 or not 2 <= ncorrect <= len(opts) - 1):
            errors.append(tag + ": multi needs 4-6 options and 2..n-1 correct")
        if mode == "num":
            if not isinstance(q["ans"], (int, float)) or q["tol"] < 0 or q["rel"] < 0:
                errors.append(tag + ": bad numeric spec")
        if q["cat"] == "calc" and mode not in ("single", "multi", "num"):
            errors.append(tag + ": bad calc mode")
        if q["cat"] != "calc" and q["cat"] != mode:
            errors.append(tag + ": cat/mode mismatch")
        blob = q["q"] + q["exp"] + "".join(o["t"] + o.get("w", "") for o in opts)
        for pat in FORBIDDEN:
            if re.search(pat, blob, re.I):
                errors.append("%s: forbidden phrase /%s/" % (tag, pat))
        for frag_name, frag in [("q", q["q"]), ("exp", q["exp"])] + \
                [("opt", o["t"]) for o in opts] + [("why", o.get("w", "")) for o in opts]:
            if not html_ok(frag):
                errors.append("%s: unbalanced html in %s" % (tag, frag_name))
        if mode in ("single", "multi"):
            if any(not o["w"].strip() for o in opts):
                errors.append(tag + ": missing option rationale")
        if not q["q"].strip() or not q["exp"].strip():
            errors.append(tag + ": empty question/explanation")
        if len(q["exp"]) > 3500:
            warnings.append(tag + ": long explanation (%d chars)" % len(q["exp"]))


def stats(questions):
    print("Total questions:", len(questions))
    by_cat = Counter(q["cat"] for q in questions)
    print("By category:", dict(by_cat), "(targets %s)" % TARGET)
    print("By mode    :", dict(Counter(q["mode"] for q in questions)))
    print("By topic   :", {t: sum(1 for q in questions if q["topic"] == t) for t, _ in qdsl.TOPICS})
    tf = [q for q in questions if q["mode"] == "tf"]
    print("T/F answers: True=%d False=%d" % (sum(1 for q in tf if q["opts"][0]["c"]), sum(1 for q in tf if not q["opts"][0]["c"])))
    sg = [q for q in questions if q["mode"] == "single"]
    print("single: position of correct option:", dict(sorted(Counter([i for q in sg for i, o in enumerate(q["opts"]) if o["c"]]).items())))
    longest = sum(1 for q in sg if max(q["opts"], key=lambda o: len(o["t"]))["c"])
    print("single: correct option is the longest in %d/%d (%.0f%%)" % (longest, len(sg), 100 * longest / max(1, len(sg))))
    mu = [q for q in questions if q["mode"] == "multi"]
    print("multi: #correct distribution:", dict(sorted(Counter(sum(1 for o in q["opts"] if o["c"]) for q in mu).items())),
          " #options:", dict(sorted(Counter(len(q["opts"]) for q in mu).items())))
    print("num questions:", sum(1 for q in questions if q["mode"] == "num"))


def render(questions):
    topics = [{"id": i, "name": n} for i, n in qdsl.TOPICS]
    data = '{"topics":' + json.dumps(topics, ensure_ascii=False) + ',"questions":[\n' + \
        ",\n".join(json.dumps(q, ensure_ascii=False, separators=(",", ":")) for q in questions) + "\n]}"
    data = data.replace("</", "<\\/")
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        tpl = f.read()
    if "__QUIZ_DATA__" not in tpl:
        raise SystemExit("template placeholder missing")
    out = tpl.replace("__QUIZ_DATA__", data).replace("__NQ__", str(len(questions)))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    return len(out)


if __name__ == "__main__":
    qs, errs = build()
    warns = []
    validate(qs, errs, warns)
    stats(qs)
    for w in warns:
        print("WARN:", w)
    if errs:
        print("\n%d ERROR(S):" % len(errs))
        for e in errs:
            print(" -", e)
        sys.exit(1)
    size = render(qs)
    print("wrote %s (%.1f KB)" % (OUT, size / 1024))
    for k, v in TARGET.items():
        got = sum(1 for q in qs if q["cat"] == k)
        if got != v:
            print("NOTE: category %s has %d (target %d)" % (k, got, v))
