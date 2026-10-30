"""Download the reading list into literature/pdfs/ (git-ignored) and write literature/papers.csv.

Only legal open copies: the arXiv preprint, or the publisher's open-access PDF found via
OpenAlex. Papers without an open copy are listed for manual download (library access).
Every arXiv match is checked against the bib title. With --update-bib, verified arXiv ids
are added to refs.bib and "VERIFY author list" placeholders are replaced with arXiv authors.

Run: uv run python scripts/fetch_papers.py --update-bib
"""

import csv
import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB, OUT, INDEX = ROOT / "literature" / "refs.bib", ROOT / "literature" / "pdfs", ROOT / "literature" / "papers.csv"
UA = {"User-Agent": "qdrift-reading-list/0.1 (student literature review; one request at a time)"}
ATOM = "{http://www.w3.org/2005/Atom}"
ENTRY = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)


CACHE = OUT / "_cache.json"


def get(url: str, timeout: int = 90) -> tuple[bytes, str]:
    """GET with polite pacing; on 429/5xx wait (Retry-After if given) and retry, then give up."""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read(), r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            wait = int(exc.headers.get("Retry-After") or 0) or 30 * 2**attempt
            print(f"  server busy ({exc.code}); waiting {wait} s", flush=True)
            time.sleep(min(wait, 300))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(15)
    raise RuntimeError("unreachable")


def cached(name: str, fn, *args):
    """Memoise network lookups in literature/pdfs/_cache.json so reruns repeat nothing."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    k = f"{name}:{json.dumps(args)}"
    if k not in cache:
        try:
            cache[k] = fn(*args)
        except Exception as exc:
            print(f"  lookup failed ({name}): {str(exc)[:60]}", flush=True)
            return None
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache[k]


def norm(t: str) -> str:
    t = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", t).lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def fields(body: str) -> dict:
    return {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}


def arxiv_meta(ids: list[str]) -> dict:
    out = {}
    for i in range(0, len(ids), 20):
        q = urllib.parse.urlencode({"id_list": ",".join(ids[i:i + 20]), "max_results": 20})
        root = ET.fromstring(get(f"https://export.arxiv.org/api/query?{q}")[0])
        for e in root.findall(f"{ATOM}entry"):
            aid = re.sub(r"v\d+$", "", e.find(f"{ATOM}id").text.rsplit("/abs/", 1)[-1])
            out[aid] = {"title": " ".join(e.find(f"{ATOM}title").text.split()),
                        "authors": [a.find(f"{ATOM}name").text.strip() for a in e.findall(f"{ATOM}author")]}
        time.sleep(3)
    return out


def arxiv_search(title: str) -> str | None:
    words = [w for w in norm(title).split() if len(w) > 3][:10]
    q = "+AND+".join("ti:" + urllib.parse.quote(w) for w in words)
    root = ET.fromstring(get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=5")[0])
    time.sleep(3)
    best = max(((similar(title, e.find(f"{ATOM}title").text), e.find(f"{ATOM}id").text)
                for e in root.findall(f"{ATOM}entry")), default=(0, None))
    return re.sub(r"v\d+$", "", best[1].rsplit("/abs/", 1)[-1]) if best[0] >= 0.9 else None


def openalex(doi: str) -> tuple[str | None, str | None]:
    """(arXiv id, open-access PDF url) for a DOI, from OpenAlex locations."""
    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?select=locations,best_oa_location"
    try:
        w = json.loads(get(url)[0])
    except Exception:
        return None, None
    arxiv = None
    for loc in w.get("locations") or []:
        for u in (loc.get("landing_page_url") or "", loc.get("pdf_url") or ""):
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u)
            if m:
                arxiv = m.group(1)
    best = w.get("best_oa_location") or {}
    return arxiv, best.get("pdf_url")


def download(url: str, path: Path) -> None:
    data, ctype = get(url, timeout=180)
    if not data.startswith(b"%PDF"):
        raise ValueError(f"not a PDF ({ctype or 'unknown type'})")
    path.write_bytes(data)
    time.sleep(3)


def update_bib(text: str, found: dict) -> str:
    def fix(m):
        key, body = m.group(2).strip(), m.group(3)
        info = found.get(key, {})
        if not info.get("arxiv") or info.get("match", 0) < 0.9:
            return m.group(0)
        if "eprint" not in body:
            body = body.rstrip() + f",\n  eprint = {{{info['arxiv']}}}, archivePrefix = {{arXiv}}"
        if "VERIFY" in body and info.get("authors"):
            authors = " and ".join(info["authors"])
            body = re.sub(r"author = \{.*?\},\n", lambda _: f"author = {{{authors}}},\n", body, count=1, flags=re.S)
            body = re.sub(r",?\s*note = \{VERIFY author list\}", "", body)
            body = body.replace("VERIFY authors and venue", "VERIFY venue")
        return f"@{m.group(1)}{{{key},{body}\n}}"
    return ENTRY.sub(fix, text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = [{"key": m.group(2).strip(), **fields(m.group(3))} for m in ENTRY.finditer(BIB.read_text())]
    found = {}
    for e in entries:   # candidate arXiv ids: bib eprint, else OpenAlex by DOI, else title search
        cand, oa = e.get("eprint"), None
        if e.get("doi"):
            r = cached("openalex", openalex, e["doi"]) or (None, None)
            cand, oa = cand or r[0], r[1]
        if not cand:
            cand = cached("search", arxiv_search, e["title"])
        found[e["key"]] = {"arxiv": cand, "oa_pdf": oa}
    ids = sorted({f["arxiv"] for f in found.values() if f["arxiv"]})
    meta = cached("meta", arxiv_meta, ids) or {}
    rows = []
    for e in entries:
        f = found[e["key"]]
        m = meta.get(f["arxiv"] or "")
        f["match"] = similar(e["title"], m["title"]) if m else 0.0
        if f["arxiv"] and f["match"] < 0.9:          # wrong id: try a title search once
            alt = cached("search", arxiv_search, e["title"])
            if alt and alt != f["arxiv"]:
                m = (cached("meta", arxiv_meta, [alt]) or {}).get(alt)
                f.update(arxiv=alt, match=similar(e["title"], m["title"]) if m else 0.0)
        if m:
            f["authors"] = m["authors"]
        path, source, note = OUT / f"{e['key']}.pdf", "", ""
        if path.exists() and path.read_bytes()[:4] == b"%PDF":
            source = "arxiv" if f["arxiv"] and f["match"] >= 0.9 else "publisher-oa"
        else:
            tries = ([("arxiv", f"https://export.arxiv.org/pdf/{f['arxiv']}"),
                      ("arxiv", f"https://arxiv.org/pdf/{f['arxiv']}")] if f["arxiv"] and f["match"] >= 0.9 else [])
            if f.get("oa_pdf"):
                tries.append(("publisher-oa", f["oa_pdf"]))
            for src, url in tries:
                try:
                    download(url, path)
                    source = src
                    break
                except Exception as exc:
                    note = str(exc)[:80]
        landing = f"https://arxiv.org/abs/{f['arxiv']}" if source == "arxiv" else (
            f"https://doi.org/{e['doi']}" if e.get("doi") else "")
        rows.append({"key": e["key"], "title": re.sub(r"[{}]", "", e["title"]), "source": source or "manual",
                     "arxiv": f["arxiv"] if f["match"] >= 0.9 else "", "doi": e.get("doi", ""),
                     "link": landing, "file": f"pdfs/{path.name}" if source else "",
                     "note": "" if source else (note or "no open copy found")})
        print(f"{rows[-1]['source']:12s} {e['key']}")
    with INDEX.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if "--update-bib" in sys.argv:
        BIB.write_text(update_bib(BIB.read_text(), found))
    got = sum(r["source"] != "manual" for r in rows)
    print(f"\n{got}/{len(rows)} downloaded to {OUT.relative_to(ROOT)}; index: {INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
