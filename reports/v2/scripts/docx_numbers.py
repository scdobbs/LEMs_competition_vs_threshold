"""Addendum 10, 5.4: every number in a .docx with its context, for the corrections table.

Reads the document XML directly (no python-docx in the environment). Tracked changes are resolved to the current
text: insertions (w:ins) are kept and deletions (w:del, w:delText, w:moveFrom) dropped. Paragraphs of the body,
tables, footnotes and endnotes are read in document order; each number gets its paragraph index, part and context.

    python reports/v2/scripts/docx_numbers.py reports/inputs/<file>.docx out.csv
"""

from __future__ import annotations

import csv
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PARTS = ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml")
NUM = re.compile(r"(?<![\w.])[-−–]?\d+(?:[.,]\d+)*(?:\s?[×x]\s?10[-−–]?\d+|[eE][-+−]?\d+)?(?:\s?%)?")
SKIP = {f"{W}del", f"{W}moveFrom"}


def _text(el: ET.Element) -> str:
    out: list[str] = []

    def walk(e: ET.Element) -> None:
        if e.tag in SKIP:
            return
        if e.tag == f"{W}t":
            out.append(e.text or "")
        elif e.tag == f"{W}tab":
            out.append("\t")
        elif e.tag in (f"{W}br", f"{W}cr"):
            out.append(" ")
        elif e.tag == f"{W}sym":
            out.append("?")
        for c in e:
            walk(c)

    walk(el)
    return "".join(out)


def paragraphs(path: str) -> list[tuple[str, int, str]]:
    rows = []
    with zipfile.ZipFile(path) as z:
        for part in PARTS:
            if part not in z.namelist():
                continue
            root = ET.fromstring(z.read(part))
            for i, p in enumerate(root.iter(f"{W}p")):
                t = _text(p).strip()
                if t:
                    rows.append((part.split("/")[1].split(".")[0], i, t))
    return rows


def numbers(path: str) -> list[dict[str, object]]:
    out = []
    for part, i, t in paragraphs(path):
        for m in NUM.finditer(t):
            s = m.group(0).strip()
            if re.fullmatch(r"(19|20)\d\d", s):           # years (citations)
                continue
            a, b = max(0, m.start() - 90), min(len(t), m.end() + 90)
            out.append({"part": part, "paragraph": i, "number": s, "context": t[a:b].replace("\n", " ")})
    return out


def main(argv: list[str]) -> None:
    rows = numbers(argv[1])
    with open(argv[2], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["part", "paragraph", "number", "context"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} numbers -> {argv[2]}")


if __name__ == "__main__":
    main(sys.argv)
