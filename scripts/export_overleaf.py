"""Export the interim report as ONE self-contained .tex file for Overleaf.

  uv run qdrift paper                      # refresh paper/generated first (numbers, references)
  uv run python scripts/export_overleaf.py

The output, paper/overleaf/Interim_Report.tex, needs no other file: every \\val{key} is replaced
by its number from paper/generated/numbers.tex, the full-paper-only branches of \\iffull are
dropped, the comments used by this repository's tooling are removed, and the bibliography is
inlined as a thebibliography environment (the .bbl that IEEEtran.bst produces), so no .bib file
and no BibTeX run is needed. The printed reference list can be switched off: the default
(\\showrefsfalse or \\showrefstrue) is read from paper/interim.tex. The file is then compiled in
an empty folder, with the switch as exported and flipped, to check that it stands alone.
"""

import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "overleaf" / "Interim_Report.tex"
sys.path.insert(0, str(Path(__file__).parent))
from check_paper import resolve_iffull  # noqa: E402  (the same brace-aware helper the checker uses)

HEADER = r"""% =============================================================================
% Interim report (Introduction, Literature Review, Problem Statement), IEEE two-column.
% SINGLE-FILE version for Overleaf, exported on {today} from the repository
% (paper/interim.tex + paper/sections/*.tex) by scripts/export_overleaf.py.
%
% HOW TO USE IN OVERLEAF
%   1. New Project -> Blank Project, then replace the whole content of main.tex with this file.
%   2. Menu -> Compiler: pdfLaTeX (the default), then press Recompile. No other file is needed:
%      IEEEtran is built into Overleaf, all numbers are written in the text, and the
%      references are in the thebibliography block at the end.
%   3. To change a reference, edit its \bibitem entry at the end; to add a citation, add a
%      \bibitem with a new key and use \cite{key} in the text.
%   4. The reference list (the last page) is hidden for now, and the text keeps its citation
%      numbers. To print the list again, change \showrefsfalse to \showrefstrue near the top.
%
% The text was drafted with an AI assistant and has to be revised by the authors in their own
% words (see the Acknowledgment). Delete these comment lines before submitting.
% ============================================================================="""

PREAMBLE = r"""\documentclass[10pt,conference]{IEEEtran}

% ---- SWITCH: the reference list (last page) --------------------------------------------------
% Hidden for now. The text keeps its citation numbers [1], [2], ...; only the printed list is
% dropped. Change \showrefsfalse to \showrefstrue (and recompile) to print the list again.
\newif\ifshowrefs
\showrefs@@SHOW@@
% -----------------------------------------------------------------------------------------------

\IEEEoverridecommandlockouts
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{array}
\usepackage{booktabs}
% hyperref makes citations and DOIs clickable. With the list hidden, a citation could only point
% at the list that is not printed, so the url package and a plain \href are enough.
\ifshowrefs\usepackage[hidelinks]{hyperref}\else\usepackage{url}\providecommand{\href}[2]{#2}\fi
"""

WRAP_BEGIN = r"\ifshowrefs\else\setbox0\vbox\bgroup\fi"   # same two lines as in paper/interim.tex
WRAP_END = r"\ifshowrefs\else\egroup\fi"
SWITCH = r"(?m)^\\showrefs(true|false)\s*$"


def group(text: str, i: int) -> tuple[str, int]:
    """Content of the {...} group that starts at text[i], and the index just after it."""
    assert text[i] == "{", text[i:i + 40]
    depth, j = 0, i
    while True:
        c = text[j]
        if c == "\\":                  # skip escaped characters such as \{ and \}
            j += 2
            continue
        depth += c == "{"
        depth -= c == "}"
        j += 1
        if depth == 0:
            return text[i + 1:j - 1], j


def read_numbers() -> dict[str, str]:
    t = (PAPER / "generated" / "numbers.tex").read_text()
    out = {}
    for m in re.finditer(r"\\qdset\{", t):
        key, j = group(t, m.end() - 1)
        while t[j].isspace():
            j += 1
        out[key], _ = group(t, j)
    return out


def expand_vals(text: str, nums: dict[str, str]) -> str:
    out, i = [], 0
    while True:
        j = text.find("\\val{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        key, i = group(text, j + 4)
        if key not in nums:
            raise SystemExit(f"\\val{{{key}}} has no value in paper/generated/numbers.tex; run `uv run qdrift paper`")
        out.append(nums[key])


def strip_comments(text: str) -> str:
    text = re.sub(r"(?m)^[ \t]*%.*\n", "", text)    # whole-line comments
    text = re.sub(r"(?<!\\)%.*", "", text)           # trailing comments (the "% claim:" tags)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def make_bbl() -> str:
    """The IEEE-formatted bibliography of interim.tex, from a real BibTeX run (tectonic)."""
    tmp = PAPER / "build" / "overleaf-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["tectonic", "--keep-intermediates", "--outdir", str(tmp), "interim.tex"], cwd=PAPER,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("tectonic failed on interim.tex:\n" + r.stderr[-2000:])
    return (tmp / "interim.bbl").read_text()


def build() -> str:
    nums = read_numbers()
    pre = (PAPER / "preamble.tex").read_text()
    k = pre.index("\\newcommand{\\qdauthor}") + len("\\newcommand{\\qdauthor}")
    author, _ = group(pre, k)
    interim = (PAPER / "interim.tex").read_text()
    show = re.search(SWITCH, interim)
    assert show, "paper/interim.tex has no \\showrefstrue or \\showrefsfalse line of its own"
    a, b = interim.index("\\begin{document}") + len("\\begin{document}"), interim.index("\\end{document}")
    body = interim[a:b]
    body = re.sub(r"\\input\{sections/([^}]+)\}", lambda m: (PAPER / "sections" / f"{m.group(1)}.tex").read_text(), body)
    body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "@@BIBLIOGRAPHY@@", body)
    body = body.replace("\\qdauthor", author)
    body = strip_comments(body)
    body = resolve_iffull(body, full=False)
    body = expand_vals(body, nums)
    body = body.replace("@@BIBLIOGRAPHY@@", make_bbl().strip())
    for leftover in ("\\val{", "\\iffull", "\\input{", "\\qdauthor", "\\todo", "\\bibliography{"):
        assert leftover not in body, f"{leftover} is still in the exported body"
    assert WRAP_BEGIN in body and WRAP_END in body, "the hide-the-list lines of paper/interim.tex are missing"
    assert body.index(WRAP_BEGIN) < body.index("\\begin{thebibliography}") < body.index(WRAP_END)
    return "\n".join([HEADER.replace("{today}", date.today().isoformat()),
                      PREAMBLE.replace("@@SHOW@@", show.group(1)), "\\begin{document}", "",
                      body.strip("\n"), "", "\\end{document}", ""])


def check_citations(tex: str) -> None:
    """Every \\cite key has a \\bibitem and every \\bibitem is cited (no compiler needed)."""
    code = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    cited = {k.strip() for m in re.finditer(r"\\cite\{([^}]*)\}", code) for k in m.group(1).split(",")}
    items = set(re.findall(r"^\\bibitem\{([^}]+)\}", code, re.M))
    assert cited <= items, f"cited but there is no \\bibitem: {sorted(cited - items)}"
    assert items <= cited, f"\\bibitem that is never cited: {sorted(items - cited)}"


def compile_alone(tex: str, what: str) -> tuple[int, str, list[str]]:
    """Compile tex alone in an empty folder: (page count, text of the PDF, suspicious log lines)."""
    from pypdf import PdfReader
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "main.tex").write_text(tex)
        r = subprocess.run(["tectonic", "--keep-logs", "main.tex"], cwd=d, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"the exported file ({what}) does not compile on its own:\n" + r.stderr[-3000:])
        pdf = PdfReader(Path(d) / "main.pdf")
        text = " ".join(p.extract_text() or "" for p in pdf.pages)
        log = (Path(d) / "main.log").read_text(errors="ignore")
        bad = [ln for ln in log.splitlines() if re.search(r"undefined|Overfull|Missing|Error|not exist", ln)
               and "Font shape" not in ln and "inputenc" not in ln]
        return len(pdf.pages), text, bad


def check(tex: str) -> None:
    """Compile the exported file alone, as exported and with the reference switch flipped."""
    default = re.search(SWITCH, tex).group(1)
    other = "true" if default == "false" else "false"
    flipped = re.sub(SWITCH, lambda m: "\\showrefs" + other, tex, count=1)
    runs = {default: compile_alone(tex, "as exported"), other: compile_alone(flipped, "switch flipped")}
    n_items = len(re.findall(r"(?m)^\\bibitem\{", tex))
    for mode, (pages, text, bad) in runs.items():
        print(f"compiled alone with \\showrefs{mode}: {pages} pages, log problems: {bad[:5] or 'none'}")
        assert "??" not in text and "[?]" not in text, f"unresolved reference or citation (\\showrefs{mode})"
        assert re.search(rf"\[\s*{n_items}\s*\]", text), f"citation [{n_items}] is missing (\\showrefs{mode})"
        for name in ("Pankaj Baid", "Kartikey Gupta", "Shreshtha Gupta"):
            assert name in text, f"{name} missing from the exported PDF (\\showrefs{mode})"
    hidden, shown = runs["false"], runs["true"]
    assert not re.search(r"doi:\s*10\.", hidden[1]), "the list is printed although the switch hides it"
    assert re.search(r"doi:\s*10\.", shown[1]), "the list is not printed although the switch shows it"
    assert shown[0] >= hidden[0]
    print(f"reference list hidden: {hidden[0]} pages; with \\showrefstrue: {shown[0]} pages")


def main() -> None:
    tex = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    odd = sorted({c for c in tex if ord(c) > 127})
    code = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))   # not the how-to comments
    n_items = len(re.findall(r"^\\bibitem\{", code, re.M))
    n_cites = len(re.findall(r"\\cite\{", code))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(tex.splitlines())} lines, {len(tex) // 1024} KB, "
          f"{n_items} references, {n_cites} \\cite commands")
    print("non-ASCII characters (fine with utf8 + T1):", "".join(odd) or "none")
    check_citations(tex)
    check(tex)


if __name__ == "__main__":
    main()
