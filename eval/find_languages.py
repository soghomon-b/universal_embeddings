from __future__ import annotations
import csv
from typing import Dict, List, Set
import pycountry

# Exceptions where ISO639-1 doesn't map cleanly to what you likely want in Tatoeba
# - Tatoeba uses ISO 639-3 codes, and for "Chinese" you often want a specific variety:
#   cmn = Mandarin (very common in Tatoeba). If you want another variety, change this.
SPECIAL_CASES = {
    "zh": "cmn",  # instead of zho; Tatoeba commonly uses cmn/yue/wuu/... not generic 'zho'
    "ku_Arab": "kub",
    "ks_Arab": "ksb",
    "kr_Arab": "krb",
}


def load_tatoeba_lang_set(sentences_csv: str) -> Set[str]:
    """Return the set of language codes present in sentences.csv."""
    langs: Set[str] = set()
    with open(sentences_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if len(row) >= 2:
                langs.add(row[1].strip())
    return langs


def iso639_1_to_3(alpha2: str) -> str | None:
    """Convert 2-letter ISO639-1 -> 3-letter ISO639-3/2 (pycountry's alpha_3)."""
    try:
        rec = pycountry.languages.get(alpha_2=alpha2)
        return getattr(rec, "alpha_3", None) if rec else None
    except Exception:
        return None


def normalize_requested_langs(
    requested: List[str],
    available_tatoeba_langs: Set[str],
) -> List[str]:
    """
    Normalize user-requested language codes to Tatoeba codes present in sentences.csv.
    Instead of raising an error, prints languages that cannot be resolved.
    """
    out: List[str] = []
    missing: List[str] = []

    for code in requested:
        code = code.strip()
        base = code.split("_", 1)[0].lower()

        # 1) Special cases (check both full code and base)
        if code in SPECIAL_CASES:
            cand = SPECIAL_CASES[code]
            if cand in available_tatoeba_langs:
                out.append(cand)
                continue

        if base in SPECIAL_CASES:
            cand = SPECIAL_CASES[base]
            if cand in available_tatoeba_langs:
                out.append(cand)
                continue

        # 2) Already present exactly
        if code in available_tatoeba_langs:
            out.append(code)
            continue

        # 3) Base present
        if base in available_tatoeba_langs:
            out.append(base)
            continue

        # 4) ISO639-1 -> ISO639-3
        if len(base) == 2:
            alpha3 = iso639_1_to_3(base)
            if alpha3 and alpha3 in available_tatoeba_langs:
                out.append(alpha3)
                continue

        # If nothing worked → mark missing
        missing.append(code)

    if missing:
        print("⚠️ The following requested languages were NOT found in sentences.csv:")
        for m in missing:
            print("   -", m)

    return out


# Example:
langs = languages = [
    "tw",
    "ar",
    "zu",
    "he",
    "ku_Arab",
    "el",
    "tr",
    "rw",
    "ln",
    "ks_Arab",
    "as",
    "ti",
    "om",
    "ts",
    "de",
    "fa",
    "az",
    "tt",
    "et",
    "lg",
    "ht",
    "su",
    "kab",
    "da",
    "ne",
    "ta",
    "kn",
    "dyu",
    "ro",
    "xh",
    "tl",
    "ig",
    "ha",
    "en",
    "tn",
    "uz",
    "gu",
    "ko",
    "hu",
    "ja",
    "hi",
    "kr_Arab",
    "ms",
    "hy",
    "id",
    "ceb",
    "ps",
    "sw",
    "ak",
    "sr",
    "mr",
    "fi",
    "it",
    "sa",
    "am",
    "sn",
    "ky",
    "ka",
    "cy",
    "yo",
    "fr",
    "pl",
    "hr",
    "ur",
    "ff",
    "bm",
    "ml",
    "mk",
    "prs",
    "jv",
    "mi",
    "ru",
    "sm",
    "ee",
    "bn",
    "wo",
    "te",
    "tpi",
    "nso",
    "zh",
    "st",
    "vi",
    "mg",
    "bg",
    "uk",
    "es",
    "qu",
    "rn",
]
available = load_tatoeba_lang_set("../data/eval/sentences.csv")
langs = normalize_requested_langs(langs, available)
print(langs)
