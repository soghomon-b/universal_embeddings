#!/usr/bin/env python3
"""
Extract N multilingual "parallel" clusters from Tatoeba (sentences.csv + links.csv).

Given:
  - sentences.csv (tab-separated): sentence_id<TAB>lang<TAB>text
  - links.csv (tab-separated): id1<TAB>id2   (translation graph edges)

Produce:
  - list of length N
  - each element is a list of sentences aligned by the same meaning
  - only for the requested languages, in the same order as list_of_languages

Example:
  langs = ["eng","fra","deu"]  # Tatoeba uses 3-letter ISO 639-3 codes (often)
  result = extract_parallel("sentences.csv", "links.csv", langs, n_sentences=2)

Notes:
  - Tatoeba language codes are typically 3-letter (eng, fra, deu), not en/fr/de.
    If you pass "en", "fr", "de", you should map them first (see map_lang()).
  - Translation relations are a graph: we use connected components.
  - We keep ONE sentence per language per component (first seen).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set
import csv

# ---------- Optional: map 2-letter to common Tatoeba 3-letter codes ----------
_LANG2_TO_TATOEBA = {
    "ar": "ara",
    "zu": "zul",
    "he": "heb",
    "el": "ell",
    "tr": "tur",
    "rw": "kin",
    "ln": "lin",
    "ks_Arab": "kas",
    "as": "asm",
    "ti": "tir",
    "ts": "tso",
    "de": "deu",
    "az": "aze",
    "tt": "tat",
    "et": "est",
    "lg": "lug",
    "ht": "hat",
    "su": "sun",
    "kab": "kab",
    "da": "dan",
    "ta": "tam",
    "kn": "kan",
    "ro": "ron",
    "xh": "xho",
    "tl": "tgl",
    "ig": "ibo",
    "ha": "hau",
    "en": "eng",
    "tn": "tsn",
    "uz": "uzb",
    "gu": "guj",
    "ko": "kor",
    "hu": "hun",
    "ja": "jpn",
    "hi": "hin",
    "hy": "hye",
    "id": "ind",
    "ceb": "ceb",
    "ps": "pus",
    "sr": "srp",
    "mr": "mar",
    "fi": "fin",
    "it": "ita",
    "sa": "san",
    "am": "amh",
    "sn": "sna",
    "ky": "kir",
    "ka": "kat",
    "cy": "cym",
    "yo": "yor",
    "fr": "fra",
    "pl": "pol",
    "hr": "hrv",
    "ur": "urd",
    "bm": "bam",
    "ml": "mal",
    "mk": "mkd",
    "jv": "jav",
    "mi": "mri",
    "ru": "rus",
    "sm": "smo",
    "ee": "ewe",
    "bn": "ben",
    "wo": "wol",
    "te": "tel",
    "tpi": "tpi",
    "zh": "cmn",
    "st": "sot",
    "vi": "vie",
    "mg": "mlg",
    "bg": "bul",
    "uk": "ukr",
    "es": "spa",
    "qu": "que",
    "rn": "run",
}


def map_lang(code: str) -> str:
    code = code.strip().lower()
    return _LANG2_TO_TATOEBA.get(code, code)


# ---------- Union-Find / DSU to compute connected components ----------
class DSU:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def _add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        self._add(x)
        # Path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass(frozen=True)
class Sentence:
    sid: int
    lang: str
    text: str


def _read_sentences(sentences_csv: str, needed_langs: Set[str]) -> Dict[int, Sentence]:
    """
    Read sentences.csv and keep only rows whose lang is in needed_langs.
    Returns dict: sid -> Sentence
    """
    kept: Dict[int, Sentence] = {}

    # Tatoeba "sentences.csv" is TSV with no header: id<TAB>lang<TAB>text
    with open(sentences_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                sid = int(row[0])
            except ValueError:
                continue
            lang = row[1].strip()
            if lang not in needed_langs:
                continue
            text = row[2].strip()
            kept[sid] = Sentence(sid=sid, lang=lang, text=text)

    return kept


def _build_dsu_over_relevant_nodes(links_csv: str, relevant_nodes: Set[int]) -> DSU:
    """
    Build DSU using only edges that touch relevant_nodes.
    For an edge (a,b):
      - if a in relevant_nodes OR b in relevant_nodes, we union(a,b)
    This is enough to recover the full component for relevant sentences
    (because DSU will include newly seen nodes as they appear in unions).
    """
    dsu = DSU()

    with open(links_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                a = int(row[0])
                b = int(row[1])
            except ValueError:
                continue

            if (a in relevant_nodes) or (b in relevant_nodes):
                dsu.union(a, b)

    return dsu


from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import csv


def extract_parallel_maxcover(
    sentences_csv: str,
    links_csv: str,
    list_of_languages: List[str],
    n_sentences: int,
    *,
    min_langs: int = 2,
    fill_missing: Optional[str] = None,  # None or "" etc.
) -> List[List[Optional[str]]]:
    """
    Like extract_parallel, but each returned cluster does NOT need all languages.
    Instead, we return clusters that cover as many requested languages as possible.

    Returns: list (length <= n_sentences) of lists (length == len(list_of_languages))
      - Each inner list is aligned by meaning (connected component).
      - Missing languages are filled with `fill_missing` (default None).
      - Clusters are ordered by descending coverage among requested languages.
    """
    if n_sentences <= 0:
        return []

    langs = [map_lang(x) for x in list_of_languages]
    needed_langs = set(langs)

    # 1) Read sentences only for requested languages
    sid_to_sent = _read_sentences(sentences_csv, needed_langs)
    if not sid_to_sent:
        return []

    relevant_nodes = set(sid_to_sent.keys())

    # 2) DSU over edges touching relevant nodes
    dsu = _build_dsu_over_relevant_nodes(links_csv, relevant_nodes)

    # 3) Build component -> lang -> text (keep first text per lang)
    comp: Dict[int, Dict[str, str]] = {}
    for sid, sent in sid_to_sent.items():
        root = dsu.find(sid)
        d = comp.setdefault(root, {})
        d.setdefault(sent.lang, sent.text)

    # 4) Score components by coverage
    scored: List[Tuple[int, int]] = []  # (coverage, root)
    for root, lang_to_text in comp.items():
        coverage = sum(1 for l in langs if l in lang_to_text)
        if coverage >= min_langs:
            scored.append((coverage, root))

    # No components with at least min_langs
    if not scored:
        return []

    # Sort best coverage first
    scored.sort(key=lambda x: x[0], reverse=True)

    # 5) Produce up to n_sentences results
    results: List[List[Optional[str]]] = []
    for coverage, root in scored:
        lang_to_text = comp[root]
        row = [lang_to_text.get(l, fill_missing) for l in langs]
        results.append(row)
        if len(results) >= n_sentences:
            break

    return results


# ------------------ Example usage ------------------
if __name__ == "__main__":
    languages = [
        "cy",
        "en",
        "hi",
        "id",
        "it",
        "ja",
        "mr",
        "ms",
        "ne",
        "ro",
        "sn",
        "sw",
        "xh",
        "zh",
        "zu",
    ]
    k = 20

    SENTENCES = "../data/eval/sentences.csv"
    LINKS = "../data/eval/links.csv"

    out = extract_parallel_maxcover(
        SENTENCES,
        LINKS,
        languages,
        n_sentences=k,
        min_langs=5,  # require at least 5 of your requested languages
        fill_missing=None,  # or "" if you prefer
    )

    print("Returned clusters:", len(out))

    for i, row in enumerate(out, 1):
        print(
            f"\n=== Cluster {i} (coverage {sum(x is not None for x in row)}/{len(row)}) ==="
        )
        for lang, text in zip([map_lang(x) for x in languages], row):
            if text is not None:
                print(f"[{lang}] {text}")
