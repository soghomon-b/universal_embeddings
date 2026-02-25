# Two histograms:
# 1) Number of languages per broad family (e.g., Indo-European)
# 2) Number of distinct sub-families per broad family (e.g., Semitic under Afro-Asiatic)
#
# Works with a CSV/TSV/Sheets-export OR with two pasted columns (Language + FamilyString).

import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------
# Option A: read from a file you exported from Google Sheets
# -----------------------
# Put your exported file path here:
# df = pd.read_csv("languages.csv")         # CSV
# df = pd.read_csv("languages.tsv", sep="\t")  # TSV
#
# Expect columns: "Language" and "Family"
#
# Option B: paste two columns directly below (Language<TAB>Family)
# -----------------------
PASTE_TWO_COLS_TSV = r"""
Twi	Niger-Congo (Kwa)
Arabic	Afro-Asiatic (Semitic)
Zulu	Niger-Congo (Bantu)
Hebrew	Afro-Asiatic (Semitic)
Kurdish (Arabic script)	Indo-European (Indo-Iranian, Iranian)
Greek	Indo-European (Hellenic)
Turkish	Turkic
Kinyarwanda	Niger-Congo (Bantu)
Lingala	Niger-Congo (Bantu)
Kashmiri (Arabic script)	Indo-European (Indo-Iranian, Indo-Aryan)
Assamese	Indo-European (Indo-Iranian, Indo-Aryan)
Tigrinya	Afro-Asiatic (Semitic)
Oromo	Afro-Asiatic (Cushitic)
Tsonga	Niger-Congo (Bantu)
German	Indo-European (Germanic)
Persian (Farsi)	Indo-European (Indo-Iranian, Iranian)
Azerbaijani	Turkic
Tatar	Turkic
Estonian	Uralic (Finnic)
Luganda	Niger-Congo (Bantu)
Haitian Creole	Creole (French-based)
Sundanese	Austronesian (Malayo-Polynesian)
Kabyle	Afro-Asiatic (Berber)
Danish	Indo-European (Germanic)
Nepali	Indo-European (Indo-Iranian, Indo-Aryan)
Tamil	Dravidian
Kannada	Dravidian
Dyula	Niger-Congo (Mande)
Romanian	Indo-European (Romance)
Xhosa	Niger-Congo (Bantu)
Tagalog (Filipino)	Austronesian (Malayo-Polynesian)
Igbo	Niger-Congo (Atlantic-Congo, Volta-Niger)
Hausa	Afro-Asiatic (Chadic)
English	Indo-European (Germanic)
Tswana	Niger-Congo (Bantu)
Uzbek	Turkic
Gujarati	Indo-European (Indo-Iranian, Indo-Aryan)
Korean	Koreanic
Hungarian	Uralic (Ugric)
Japanese	Japonic
Hindi	Indo-European (Indo-Iranian, Indo-Aryan)
Kanuri (Arabic script)	Nilo-Saharan (Saharan)
Malay	Austronesian (Malayo-Polynesian)
Armenian	Indo-European (Armenian)
Indonesian	Austronesian (Malayo-Polynesian)
Cebuano	Austronesian (Malayo-Polynesian)
Pashto	Indo-European (Indo-Iranian, Iranian)
Swahili	Niger-Congo (Bantu)
Akan	Niger-Congo (Kwa)
Serbian	Indo-European (Slavic, South Slavic)
Marathi	Indo-European (Indo-Iranian, Indo-Aryan)
Finnish	Uralic (Finnic)
Italian	Indo-European (Romance)
Sanskrit	Indo-European (Indo-Iranian, Indo-Aryan)
Amharic	Afro-Asiatic (Semitic)
Shona	Niger-Congo (Bantu)
Kyrgyz	Turkic
Georgian	Kartvelian
Welsh	Indo-European (Celtic)
Yoruba	Niger-Congo (Atlantic-Congo, Volta-Niger)
French	Indo-European (Romance)
Polish	Indo-European (Slavic, West Slavic)
Croatian	Indo-European (Slavic, South Slavic)
Urdu	Indo-European (Indo-Iranian, Indo-Aryan)
Fulah	Niger-Congo (Atlantic-Congo)
Bambara	Niger-Congo (Mande)
Malayalam	Dravidian
Macedonian	Indo-European (Slavic, South Slavic)
Dari	Indo-European (Indo-Iranian, Iranian)
Javanese	Austronesian (Malayo-Polynesian)
Māori	Austronesian (Polynesian)
Russian	Indo-European (Slavic, East Slavic)
Samoan	Austronesian (Polynesian)
Ewe	Niger-Congo (Kwa)
Bengali	Indo-European (Indo-Iranian, Indo-Aryan)
Wolof	Niger-Congo (Atlantic)
Telugu	Dravidian
Tok Pisin	Creole (English-based)
Northern Sotho	Niger-Congo (Bantu)
Chinese	Sino-Tibetan (Sinitic)
Southern Sotho	Niger-Congo (Bantu)
Vietnamese	Austroasiatic
Malagasy	Austronesian (Malayo-Polynesian)
Bulgarian	Indo-European (Slavic, South Slavic)
Ukrainian	Indo-European (Slavic, East Slavic)
Spanish	Indo-European (Romance)
Quechua	Quechuan
Kirundi	Niger-Congo (Bantu)
""".strip()

# Build df from pasted TSV (Option B)
if PASTE_TWO_COLS_TSV:
    rows = []
    for line in PASTE_TWO_COLS_TSV.splitlines():
        if not line.strip():
            continue
        lang, fam = line.split("\t", 1)
        rows.append({"Language": lang.strip(), "Family": fam.strip()})
    df = pd.DataFrame(rows)


# -----------------------
# Parsing helpers
# -----------------------
def parse_family(fam_str: str):
    """
    Input examples:
      "Indo-European (Indo-Iranian, Indo-Aryan)"
      "Afro-Asiatic (Semitic)"
      "Turkic"
      "Creole (French-based)"

    Returns:
      broad: "Indo-European"
      subfamilies: list like ["Indo-Iranian", "Indo-Aryan"] or [] if none
    """
    s = fam_str.strip()
    m = re.match(r"^([^(]+?)(?:\s*\(([^)]+)\))?\s*$", s)
    if not m:
        return s, []
    broad = m.group(1).strip()
    inside = m.group(2)
    if not inside:
        return broad, []
    subs = [p.strip() for p in inside.split(",") if p.strip()]
    return broad, subs


# Parse columns
broad_list = []
sub_list = []
for fam in df["Family"].astype(str):
    broad, subs = parse_family(fam)
    broad_list.append(broad)
    sub_list.append(subs)

df["BroadFamily"] = broad_list
df["SubFamilies"] = sub_list

# -----------------------
# Histogram 1: # languages per broad family
# -----------------------
lang_counts = df["BroadFamily"].value_counts().sort_values(ascending=False)

plt.figure()
plt.bar(lang_counts.index, lang_counts.values)
plt.xticks(rotation=60, ha="right")
plt.ylabel("# of languages")
plt.title("Languages per broad family")
plt.tight_layout()
plt.show()

# -----------------------
# Histogram 2: # distinct sub-families per broad family
# (If a language has multiple subfamilies, we count each subfamily toward the set.)
# -----------------------
broad_to_subs = defaultdict(set)
for broad, subs in zip(df["BroadFamily"], df["SubFamilies"]):
    for sub in subs:
        broad_to_subs[broad].add(sub)

sub_counts = pd.Series({b: len(sset) for b, sset in broad_to_subs.items()}).sort_values(
    ascending=False
)

# Include broad families that had zero subfamilies explicitly
for b in df["BroadFamily"].unique():
    sub_counts.loc[b] = sub_counts.get(b, 0)

sub_counts = sub_counts.sort_values(ascending=False)

plt.figure()
plt.bar(sub_counts.index, sub_counts.values)
plt.xticks(rotation=60, ha="right")
plt.ylabel("# of distinct sub-families")
plt.title("Distinct sub-families per broad family")
plt.tight_layout()
plt.show()
