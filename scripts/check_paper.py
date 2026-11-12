"""Paper checks (Task 11).

  uv run python scripts/check_paper.py                  structure: citations, labels, \\val keys, TODOs
  uv run python scripts/check_paper.py --overlap        also word-sequence overlap with the downloaded papers
  uv run python scripts/check_paper.py --draft FILE     overlap of any text file (e.g. a paragraph draft)

The overlap check approximates a similarity report: it marks every run of N consecutive words
(default 8) in your prose that also appears in a paper in literature/pdfs/. Turnitin and similar
tools also search the web and other submissions, so keep this number well below the 15% limit.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER, PDFS = ROOT / "paper", ROOT / "literature" / "pdfs"
DROP = r"cite[a-zA-Z]*|nocite|ref|cref|Cref|autoref|eqref|label|val|todo|input|include|includegraphics|url|href|bibliography|bibliographystyle|graphicspath|begin|end"


def uncomment(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def prose(tex: str) -> str:
    """Plain words of a LaTeX source: comments, math and reference commands removed."""
    t = uncomment(tex)
    t = re.sub(r"\\begin\{(equation|align|figure|table)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]", " ", t, flags=re.S)
    for _ in range(2):
        t = re.sub(r"\\(?:" + DROP + r")\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})+", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", t)
    return re.sub(r"[{}~\\]", " ", t)


def words(text: str) -> list[str]:
    text = re.sub(r"-\s*\n\s*", "", text)          # join words hyphenated across PDF lines
    return re.findall(r"[a-z0-9]+", text.lower())


def sources() -> dict[str, list[str]]:
    from pypdf import PdfReader
    cache = PDFS / "_text"
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for pdf in sorted(PDFS.glob("*.pdf")):
        txt = cache / f"{pdf.stem}.txt"
        if not txt.exists() or txt.stat().st_mtime < pdf.stat().st_mtime:
            try:
                txt.write_text("\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages))
            except Exception as exc:
                print(f"  could not read {pdf.name}: {exc}")
                continue
        out[pdf.stem] = words(txt.read_text())
    return out


def overlap(draft_words: list[str], n: int, top: int = 12) -> float:
    if len(draft_words) < n:
        print(f"overlap: only {len(draft_words)} words of prose so far; nothing to compare")
        return 0.0
    srcs = sources()
    index: dict[tuple, set] = {}
    for k, w in srcs.items():
        for i in range(len(w) - n + 1):
            index.setdefault(tuple(w[i:i + n]), set()).add(k)
    hit, by_src = [set() for _ in draft_words], {}
    for i in range(len(draft_words) - n + 1):
        ks = index.get(tuple(draft_words[i:i + n]))
        if ks:
            for j in range(i, i + n):
                hit[j] |= ks
            for k in ks:
                by_src.setdefault(k, set()).update(range(i, i + n))
    total = sum(bool(h) for h in hit) / len(draft_words)
    print(f"overlap: {100 * total:.1f}% of {len(draft_words)} words are in {n}-word runs found in "
          f"{len(srcs)} downloaded papers")
    for k, pos in sorted(by_src.items(), key=lambda kv: -len(kv[1]))[:top]:
        print(f"  {100 * len(pos) / len(draft_words):5.1f}%  {k}")
    spans, i = [], 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j < len(hit) and hit[j]:
                j += 1
            spans.append((j - i, i, j))
            i = j
        else:
            i += 1
    for length, a, b in sorted(spans, reverse=True)[:top]:
        print(f"  [{length} words, {', '.join(sorted(set.union(*hit[a:b])))}] ... {' '.join(draft_words[a:b])[:160]}")
    return total


def resolve_iffull(text: str, full: bool) -> str:
    """Keep the matching branch of \\iffull{full-paper text}{interim text} (nested braces allowed)."""
    out, i, key = [], 0, "\\iffull"
    while True:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k, args = j + len(key), []
        for _ in range(2):
            while text[k].isspace():
                k += 1
            depth, start = 0, k
            while True:
                depth += text[k] == "{"
                depth -= text[k] == "}"
                k += 1
                if depth == 0:
                    break
            args.append(text[start + 1:k - 1])
        out.append(args[0] if full else args[1])
        i = k


def structure(doc: str = "main") -> int:
    main = (PAPER / f"{doc}.tex").read_text()
    secs = [uncomment((PAPER / "sections" / f"{m}.tex").read_text())
            for m in re.findall(r"\\input\{sections/([^}]+)\}", uncomment(main))]
    text = resolve_iffull(uncomment(main) + "\n" + "\n".join(secs), full=(doc == "main"))
    gens = re.findall(r"\\input\{generated/(tab-[^}]+)\}", text)
    labels_src = text + "\n".join((PAPER / "generated" / f"{g}.tex").read_text() for g in gens
                                  if (PAPER / "generated" / f"{g}.tex").exists())
    errors, warns = [], []
    bib = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "literature" / "refs.bib").read_text()))
    cited = {k.strip() for grp in re.findall(r"\\cite[a-zA-Z]*\{([^}]*)\}", text) for k in grp.split(",")}
    errors += [f"\\cite{{{k}}} is not in literature/refs.bib" for k in sorted(cited - bib)]
    if re.search(r"\\nocite\{\*\}", text):
        warns.append("\\nocite{*} still in main.tex (lists every reference; remove before submission)")
    vals = set(re.findall(r"\\qdset\{([^}]+)\}", (PAPER / "generated" / "numbers.tex").read_text())) \
        if (PAPER / "generated" / "numbers.tex").exists() else set()
    errors += [f"\\val{{{k}}} has no value in generated/numbers.tex" for k in
               sorted(set(re.findall(r"\\val\{([^}]+)\}", text)) - vals)]
    labels = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", labels_src))
    refs = {k for grp in re.findall(r"\\(?:ref|cref|Cref|autoref)\{([^}]*)\}", text) for k in grp.split(",")}
    warns += [f"{k} is never referenced in the text" for k in sorted(labels - refs)]
    errors += [f"\\ref{{{k}}} points to no label" for k in sorted(refs - labels) if k.startswith(("fig:", "tab:"))]
    todos = len(re.findall(r"\\todo\{", text))
    if todos:
        warns.append(f"{todos} TODO marker(s) left")
    comments = sum(line.lstrip().startswith("%") for s in re.findall(r"\\input\{sections/([^}]+)\}", main)
                   for line in (PAPER / "sections" / f"{s}.tex").read_text().splitlines())
    n_words = len(words(prose(text)))
    pdf = PAPER / "build" / f"{doc}.pdf"
    if pdf.exists():
        from pypdf import PdfReader
        pages = len(PdfReader(pdf).pages)
        if pages > 12:
            warns.append(f"{pages} pages; check the page limit of the class or venue")
        print(f"pdf: {pages} page(s) (built {pdf.relative_to(ROOT)})")
    print(f"prose: {n_words} words; cited: {len(cited)} of {len(bib)} references; "
          f"{len(labels)} figure/table labels; {comments} guidance comment line(s) in sections")
    for w in warns:
        print("  warning: " + w)
    for e in errors:
        print("  ERROR: " + e)
    return len(errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--doc", default="main", choices=["main", "interim"], help="which document to check")
    ap.add_argument("--draft", type=Path, help="check the overlap of a plain-text or .tex file")
    ap.add_argument("-n", type=int, default=8, help="words per matching run (default 8)")
    a = ap.parse_args()
    if a.draft:
        raw = a.draft.read_text()
        share = overlap(words(prose(raw) if a.draft.suffix == ".tex" else raw), a.n)
        sys.exit(1 if share >= 0.10 else 0)
    errors = structure(a.doc)
    if a.overlap:
        main_tex = (PAPER / f"{a.doc}.tex").read_text()
        body = "\n".join((PAPER / "sections" / f"{s}.tex").read_text()
                         for s in re.findall(r"\\input\{sections/([^}]+)\}", uncomment(main_tex)))
        abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S)
        share = overlap(words(prose((abstract.group(1) if abstract else "") + "\n" + body)), a.n)
        if share >= 0.10:
            print("  warning: overlap is 10% or more; rewrite the listed passages in your own words")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
