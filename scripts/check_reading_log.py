"""Reading log helper (Task 1).

  uv run python scripts/check_reading_log.py --init   add an empty entry for every refs.bib key
  uv run python scripts/check_reading_log.py          report missing fields and bib problems

Write every note in your own words. Quotes: one line at most, in quotation marks, with a page.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB, LOG = ROOT / "literature" / "refs.bib", ROOT / "literature" / "reading-log.md"
FIELDS = ["Status", "Question", "Method", "Hardware and budget", "Key results", "Limitations", "Relevance (RQ1-RQ4)"]

entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", BIB.read_text(), re.S)
keys = [k for k, _ in entries]

if "--init" in sys.argv:
    text = LOG.read_text() if LOG.exists() else "# Reading log\n\nOne entry per paper in refs.bib. Status: unread, skimmed or read.\n"
    for k in keys:
        if f"## {k}\n" not in text:
            text += f"\n## {k}\n" + "".join(f"- {f}: {'unread' if f == 'Status' else ''}\n" for f in FIELDS)
    LOG.write_text(text)
    print(f"{len(keys)} entries in {LOG.relative_to(ROOT)}")
    sys.exit(0)

problems = []
for k, body in entries:
    if "doi =" not in body and "eprint =" not in body:
        problems.append(f"bib {k}: no DOI or arXiv id")
    if "VERIFY" in body:
        problems.append(f"bib {k}: still marked VERIFY")
log = LOG.read_text() if LOG.exists() else ""
sections = dict(re.findall(r"^## (\S+)\n(.*?)(?=^## |\Z)", log, re.S | re.M))
for k in keys:
    body = sections.get(k)
    if body is None:
        problems.append(f"log {k}: missing entry")
        continue
    for f in FIELDS:
        m = re.search(rf"^- {re.escape(f)}:(.*)$", body, re.M)
        if not m or not m.group(1).strip() or (f == "Status" and m.group(1).strip() != "read"):
            problems.append(f"log {k}: '{f}' not filled" if f != "Status" else f"log {k}: not read yet")
read = sum(1 for k in keys if re.search(r"^- Status: read\s*$", sections.get(k, ""), re.M))
print(f"{read}/{len(keys)} papers read; {len(problems)} open item(s)")
for p in problems[:60]:
    print("  " + p)
sys.exit(1 if problems else 0)
