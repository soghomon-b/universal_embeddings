"""
Download + parse OPUS NLLB-v1 resources from download_links, sample 5k pairs per language pair,
and write merged TSV.

Dependencies:
  pip install requests beautifulsoup4 translate-toolkit

Notes:
- Handles:
  * download.php?f=NLLB/v1/moses/<src>-<tgt>.txt.zip
  * download.php?f=NLLB/v1/tmx/<src>-<tgt>.tmx.gz
- Outputs:
  out_dir/
    per_pair/<src>-<tgt>.tsv
    merged.tsv
"""

from __future__ import annotations

import gzip
import io
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import requests
from translate.storage.tmx import tmxfile  # translate-toolkit


# ---- paste your dict here (or import it) ----
download_links = download_links = {
    ("ak", "ee"): "download.php?f=NLLB/v1/moses/ak-ee.txt.zip",
    ("ak", "tw"): "download.php?f=NLLB/v1/moses/ak-tw.txt.zip",
    ("am", "om"): "download.php?f=NLLB/v1/moses/am-om.txt.zip",
    ("am", "ti"): "download.php?f=NLLB/v1/moses/am-ti.txt.zip",
    ("ar", "en"): "download.php?f=NLLB/v1/moses/ar-en.txt.zip",
    ("ar", "fa"): "download.php?f=NLLB/v1/moses/ar-fa.txt.zip",
    ("ar", "fr"): "download.php?f=NLLB/v1/moses/ar-fr.txt.zip",
    ("ar", "he"): "download.php?f=NLLB/v1/moses/ar-he.txt.zip",
    ("as", "bn"): "download.php?f=NLLB/v1/moses/as-bn.txt.zip",
    ("az", "tr"): "download.php?f=NLLB/v1/moses/az-tr.txt.zip",
    ("bg", "mk"): "download.php?f=NLLB/v1/moses/bg-mk.txt.zip",
    ("bm", "dyu"): "download.php?f=NLLB/v1/moses/bm-dyu.txt.zip",
    ("bm", "fr"): "download.php?f=NLLB/v1/moses/bm-fr.txt.zip",
    ("ceb", "tl"): "download.php?f=NLLB/v1/moses/ceb-tl.txt.zip",
    ("cs", "sk"): "download.php?f=NLLB/v1/moses/cs-sk.txt.zip",
    ("cy", "en"): "download.php?f=NLLB/v1/moses/cy-en.txt.zip",
    ("da", "de"): "download.php?f=NLLB/v1/moses/da-de.txt.zip",
    ("de", "en"): "download.php?f=NLLB/v1/moses/de-en.txt.zip",
    ("de", "nl"): "download.php?f=NLLB/v1/moses/de-nl.txt.zip",
    ("el", "tr"): "download.php?f=NLLB/v1/moses/el-tr.txt.zip",
    ("en", "es"): "download.php?f=NLLB/v1/moses/en-es.txt.zip",
    ("en", "fj"): "download.php?f=NLLB/v1/moses/en-fj.txt.zip",
    ("en", "fr"): "download.php?f=NLLB/v1/moses/en-fr.txt.zip",
    ("en", "hi"): "download.php?f=NLLB/v1/moses/en-hi.txt.zip",
    ("en", "ja"): "download.php?f=NLLB/v1/moses/en-ja.txt.zip",
    ("en", "mi"): "download.php?f=NLLB/v1/moses/en-mi.txt.zip",
    ("en", "pt"): "download.php?f=NLLB/v1/moses/en-pt.txt.zip",
    ("en", "ru"): "download.php?f=NLLB/v1/moses/en-ru.txt.zip",
    ("en", "sm"): "download.php?f=NLLB/v1/moses/en-sm.txt.zip",
    ("en", "sw"): "download.php?f=NLLB/v1/moses/en-sw.txt.zip",
    ("en", "tl"): "download.php?f=NLLB/v1/moses/en-tl.txt.zip",
    ("en", "tpi"): "download.php?f=NLLB/v1/moses/en-tpi.txt.zip",
    ("en", "zh"): "download.php?f=NLLB/v1/moses/en-zh.txt.zip",
    ("es", "pt"): "download.php?f=NLLB/v1/moses/es-pt.txt.zip",
    ("es", "qu"): "download.php?f=NLLB/v1/tmx/qu-es.tmx.gz",
    ("et", "fi"): "download.php?f=NLLB/v1/moses/et-fi.txt.zip",
    ("fa", "ur"): "download.php?f=NLLB/v1/moses/fa-ur.txt.zip",
    ("ff", "wo"): "download.php?f=NLLB/v1/moses/ff-wo.txt.zip",
    ("fi", "ru"): "download.php?f=NLLB/v1/moses/fi-ru.txt.zip",
    ("fr", "ht"): "download.php?f=NLLB/v1/moses/fr-ht.txt.zip",
    ("fr", "kab"): "download.php?f=NLLB/v1/moses/fr-kab.txt.zip",
    ("fr", "ln"): "download.php?f=NLLB/v1/moses/fr-ln.txt.zip",
    ("fr", "mg"): "download.php?f=NLLB/v1/moses/fr-mg.txt.zip",
    ("fr", "wo"): "download.php?f=NLLB/v1/moses/fr-wo.txt.zip",
    ("gu", "hi"): "download.php?f=NLLB/v1/moses/gu-hi.txt.zip",
    ("ha", "ig"): "download.php?f=NLLB/v1/moses/ha-ig.txt.zip",
    ("hi", "mr"): "download.php?f=NLLB/v1/moses/hi-mr.txt.zip",
    ("hi", "ne"): "download.php?f=NLLB/v1/moses/hi-ne.txt.zip",
    ("hi", "sa"): "download.php?f=NLLB/v1/moses/hi-sa.txt.zip",
    ("hr", "sr"): "download.php?f=NLLB/v1/moses/hr-sr.txt.zip",
    ("hu", "ro"): "download.php?f=NLLB/v1/moses/hu-ro.txt.zip",
    ("hy", "ka"): "download.php?f=NLLB/v1/moses/hy-ka.txt.zip",
    ("hy", "ru"): "download.php?f=NLLB/v1/moses/hy-ru.txt.zip",
    ("id", "jv"): "download.php?f=NLLB/v1/moses/id-jv.txt.zip",
    ("id", "ms"): "download.php?f=NLLB/v1/moses/id-ms.txt.zip",
    ("id", "su"): "download.php?f=NLLB/v1/moses/id-su.txt.zip",
    ("ig", "yo"): "download.php?f=NLLB/v1/moses/ig-yo.txt.zip",
    ("it", "ro"): "download.php?f=NLLB/v1/moses/it-ro.txt.zip",
    ("ja", "ko"): "download.php?f=NLLB/v1/moses/ja-ko.txt.zip",
    ("ja", "vi"): "download.php?f=NLLB/v1/moses/ja-vi.txt.zip",
    ("ja", "zh"): "download.php?f=NLLB/v1/moses/ja-zh.txt.zip",
    ("jv", "su"): "download.php?f=NLLB/v1/moses/jv-su.txt.zip",
    ("kn", "ml"): "download.php?f=NLLB/v1/moses/kn-ml.txt.zip",
    ("ko", "zh"): "download.php?f=NLLB/v1/moses/ko-zh.txt.zip",
    ("kr_Arab", "kr_Latn"): "download.php?f=NLLB/v1/moses/kr_Arab-kr_Latn.txt.zip",
    ("ks_Arab", "ks_Deva"): "download.php?f=NLLB/v1/moses/ks_Arab-ks_Deva.txt.zip",
    ("ku_Arab", "ku_Latn"): "download.php?f=NLLB/v1/moses/ku_Arab-ku_Latn.txt.zip",
    ("ky", "uz"): "download.php?f=NLLB/v1/moses/ky-uz.txt.zip",
    ("lg", "sw"): "download.php?f=NLLB/v1/moses/lg-sw.txt.zip",
    ("ml", "ta"): "download.php?f=NLLB/v1/moses/ml-ta.txt.zip",
    ("no", "sv"): "download.php?f=NLLB/v1/moses/no-sv.txt.zip",
    ("nso", "st"): "download.php?f=NLLB/v1/moses/nso-st.txt.zip",
    ("pl", "uk"): "download.php?f=NLLB/v1/moses/pl-uk.txt.zip",
    ("prs", "ps"): "download.php?f=NLLB/v1/moses/prs-ps.txt.zip",
    ("rn", "rw"): "download.php?f=NLLB/v1/tmx/rw-rn.tmx.gz",
    ("ru", "tt"): "download.php?f=NLLB/v1/moses/ru-tt.txt.zip",
    ("rw", "sw"): "download.php?f=NLLB/v1/moses/rw-sw.txt.zip",
    ("sn", "zu"): "download.php?f=NLLB/v1/moses/sn-zu.txt.zip",
    ("ta", "te"): "download.php?f=NLLB/v1/moses/ta-te.txt.zip",
    ("tn", "ts"): "download.php?f=NLLB/v1/moses/tn-ts.txt.zip",
    ("tr", "uz"): "download.php?f=NLLB/v1/moses/tr-uz.txt.zip",
    ("vi", "zh"): "download.php?f=NLLB/v1/moses/vi-zh.txt.zip",
    ("xh", "zu"): "download.php?f=NLLB/v1/moses/xh-zu.txt.zip",
}


BASE = "https://opus.nlpl.eu/legacy/"
OUT_DIR = "nllb_sampled"
SAMPLES_PER_PAIR = 5000
SEED = 42

# Simple length band (tokens ~= whitespace split)
MIN_TOK = 5
MAX_TOK = 100
MAX_LEN_RATIO = 2.5  # len(src)/len(tgt) must be within [1/MAX_LEN_RATIO, MAX_LEN_RATIO]


def mkdirp(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def norm_line(s: str) -> str:
    return " ".join(s.strip().split())


def tok_len(s: str) -> int:
    return len(s.split())


def pass_filters(src: str, tgt: str) -> bool:
    src = norm_line(src)
    tgt = norm_line(tgt)
    if not src or not tgt:
        return False
    ls = tok_len(src)
    lt = tok_len(tgt)
    if ls < MIN_TOK or lt < MIN_TOK:
        return False
    if ls > MAX_TOK or lt > MAX_TOK:
        return False
    ratio = (ls / lt) if lt else 999.0
    if ratio > MAX_LEN_RATIO or ratio < (1.0 / MAX_LEN_RATIO):
        return False
    return True


def download_bytes(url: str, timeout: int = 120) -> bytes:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except:
        print(f"No URL found: {url}")
        return ""


def iter_pairs_from_moses_zip(blob: bytes) -> Iterator[Tuple[str, str]]:
    """
    MOSES zip usually contains exactly two text files (one per language).
    We'll pick the two *largest* text members and zip them line-by-line.
    """
    with zipfile.ZipFile(io.BytesIO(blob), "r") as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        # keep only plausible text files
        members = [
            m
            for m in members
            if m.filename.lower().endswith(
                (".txt", ".gz", ".en", ".fr", ".ar", ".ru", ".zh", ".hi")
            )
            or True
        ]
        # Prefer largest two members
        members = sorted(members, key=lambda m: m.file_size, reverse=True)
        if len(members) < 2:
            raise RuntimeError("MOSES zip did not contain at least 2 files.")

        m1, m2 = members[0], members[1]

        def read_lines(member: zipfile.ZipInfo) -> List[str]:
            data = z.read(member)
            # Some zips contain gz inside; handle.
            if member.filename.endswith(".gz"):
                data = gzip.decompress(data)
            text = data.decode("utf-8", errors="replace")
            return text.splitlines()

        l1 = read_lines(m1)
        l2 = read_lines(m2)

        n = min(len(l1), len(l2))
        for i in range(n):
            yield l1[i], l2[i]


def iter_pairs_from_tmx_gz(
    blob: bytes, src: str, tgt: str
) -> Iterator[Tuple[str, str]]:
    """
    Parse .tmx.gz using translate-toolkit.
    """
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as gf:
        raw = gf.read()
    # translate-toolkit wants a binary file-like
    with io.BytesIO(raw) as f:
        tmx = tmxfile(f, src, tgt)
        for unit in tmx.unit_iter():
            yield unit.source or "", unit.target or ""


def sample_pairs(
    pairs: Iterable[Tuple[str, str]], k: int, seed: int
) -> List[Tuple[str, str]]:
    """
    Reservoir sample after filtering, so we don't store everything in memory.
    """
    rng = random.Random(seed)
    reservoir: List[Tuple[str, str]] = []
    n_seen = 0
    for s, t in pairs:
        s = norm_line(s)
        t = norm_line(t)
        if not pass_filters(s, t):
            continue

        n_seen += 1
        if len(reservoir) < k:
            reservoir.append((s, t))
        else:
            j = rng.randrange(n_seen)
            if j < k:
                reservoir[j] = (s, t)

    rng.shuffle(reservoir)
    return reservoir


def write_tsv(path: str, rows: List[Tuple[str, str]], src: str, tgt: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s, t in rows:
            f.write(f"{src}\t{tgt}\t{s}\t{t}\n")


def main() -> None:
    random.seed(SEED)
    mkdirp(OUT_DIR)
    per_pair_dir = os.path.join(OUT_DIR, "per_pair")
    mkdirp(per_pair_dir)

    merged_path = os.path.join(OUT_DIR, "merged.tsv")
    merged_out = open(merged_path, "w", encoding="utf-8", newline="\n")

    total_written = 0
    for (a, b), rel in sorted(download_links.items()):
        url = BASE + rel
        print(f"\n== {a}-{b} ==\nDownloading: {url}")

        blob = download_bytes(url)
        if blob == "":
            continue

        if rel.endswith(".txt.zip"):
            raw_pairs = iter_pairs_from_moses_zip(blob)
        elif rel.endswith(".tmx.gz"):
            # NOTE: the URL filename might be reversed (e.g., qu-es) even if key is ("es","qu")
            # We pass the key order (a,b) to tmxfile; if it errors, swap.
            try:
                raw_pairs = iter_pairs_from_tmx_gz(blob, a, b)
                # quick probe
                _ = next(iter(raw_pairs))
                # re-create iterator after probing
                raw_pairs = iter_pairs_from_tmx_gz(blob, a, b)
            except Exception:
                raw_pairs = iter_pairs_from_tmx_gz(blob, b, a)
        else:
            print(f"Skipping unknown format: {rel}")
            continue

        sampled = sample_pairs(raw_pairs, SAMPLES_PER_PAIR, seed=SEED)
        print(
            f"Sampled {len(sampled)} pairs after filtering (target {SAMPLES_PER_PAIR})."
        )

        per_path = os.path.join(per_pair_dir, f"{a}-{b}.tsv")
        write_tsv(per_path, sampled, a, b)
        print(f"Wrote: {per_path}")

        for s, t in sampled:
            merged_out.write(f"{a}\t{b}\t{s}\t{t}\n")
        total_written += len(sampled)

    merged_out.close()
    print(f"\nDone. Wrote merged TSV with {total_written} total pairs:")
    print(f"  {merged_path}")


if __name__ == "__main__":
    print(len(download_links))
    main()
