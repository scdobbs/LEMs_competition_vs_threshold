"""Addendum 10, 5.4: every number in a .docx with its context, for the corrections table.

Reads the document XML directly (no python-docx in the environment). Tracked changes are resolved to the current
text: insertions (w:ins) are kept and deletions (w:del, w:delText, w:moveFrom) dropped. Paragraphs of the body,
tables, footnotes and endnotes are read in document order, equation text (m:t) included; each number gets its paragraph
index, part and context. A file whose zip central directory is missing (a truncated save) is read from its local file
headers, and each part's CRC is checked (addendum 11: the supplement as delivered); the file itself is never written.

    python reports/v2/scripts/docx_numbers.py reports/inputs/<file>.docx out.csv
"""

from __future__ import annotations

import csv
import re
import struct
import sys
import zipfile
import zlib
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
PARTS = ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
SUP_RUN = re.compile(r"[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+")
# a number: optional sign, digits, and a power of ten written as × 10^-7, ×10−7 (superscript run), 10⁻⁷ or e-7; a bare
# power of ten (10^-3) counts as one number. "^" before a number (a unit exponent) and a letter or digit before it
# (S1, m2, GM1) exclude it.
NUM = re.compile(r"(?<![\w.^])(?:10\^[-−–]?\d+|(?:[-−–](?=\d))?\d+(?:[.,]\d+)*(?:\s?[×x]\s?10\^?[-−–]?\d+|[eE][-+−]?\d+)?)(?:\s?%)?")
SKIP = {f"{W}del", f"{W}moveFrom"}


def _text(el: ET.Element) -> str:
    """Current text of a paragraph. A superscript run is written with a leading "^" (Word stores 10⁵ as "10" plus a
    superscript "5"; in equations an m:sup element), so that powers of ten parse and unit exponents (m², yr⁻¹) are not read as numbers."""
    out: list[str] = []

    def walk(e: ET.Element, sup: bool = False) -> None:
        if e.tag in SKIP:
            return
        if e.tag == f"{W}r":
            va = e.find(f"{W}rPr/{W}vertAlign")
            sup = va is not None and va.get(f"{W}val") == "superscript"
            if sup and "".join(t.text or "" for t in e.iter(f"{W}t")).strip():
                out.append("^")
        if e.tag == f"{M}sup":                     # an equation superscript (OMML)
            out.append("^")
        if e.tag in (f"{W}t", f"{M}t"):
            out.append(e.text or "")
        elif e.tag == f"{W}tab":
            out.append("\t")
        elif e.tag in (f"{W}br", f"{W}cr"):
            out.append(" ")
        elif e.tag == f"{W}sym":
            out.append("?")
        for c in e:
            walk(c, sup)

    walk(el)
    return "".join(out)


def read_parts(path: str) -> dict[str, bytes]:
    """The XML parts of a .docx; from the local file headers when the central directory is missing."""
    try:
        with zipfile.ZipFile(path) as z:
            return {n: z.read(n) for n in z.namelist() if n in PARTS}
    except zipfile.BadZipFile:
        pass
    data, out, i = open(path, "rb").read(), {}, 0
    while (i := data.find(b"PK\x03\x04", i)) >= 0:
        _, flag, meth, _, _, crc, cs, _, nl, el = struct.unpack("<HHHHHIIIHH", data[i + 4:i + 30])
        name, start = data[i + 30:i + 30 + nl].decode(), i + 30 + nl + el
        if name in PARTS:
            if flag & 8 or meth != 8:
                raise ValueError(f"{path}: cannot read {name} without the central directory")
            raw = zlib.decompress(data[start:start + cs], -15)
            if zlib.crc32(raw) != crc:
                raise ValueError(f"{path}: CRC mismatch in {name}")
            out[name] = raw
        i += 4
    if PARTS[0] not in out:
        raise ValueError(f"{path}: no {PARTS[0]}")
    print(f"{path}: zip central directory missing; read {sorted(out)} from local headers, CRC ok", file=sys.stderr)
    return out


def paragraphs(path: str) -> list[tuple[str, int, str]]:
    rows = []
    xml = read_parts(path)
    for part in PARTS:
        if part in xml:
            root = ET.fromstring(xml[part])
            for i, p in enumerate(root.iter(f"{W}p")):
                t = _text(p).strip()
                if t:
                    rows.append((part.split("/")[1].split(".")[0], i, t))
    return rows


def numbers(path: str) -> list[dict[str, object]]:
    out = []
    for part, i, t in paragraphs(path):
        t = SUP_RUN.sub(lambda q: "^" + q.group(0).translate(SUP), t)
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
