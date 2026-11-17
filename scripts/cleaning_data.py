from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import re
import unicodedata
from datetime import datetime

import pandas as pd

df = pd.read_csv(ROOT / "data" / "accidente.csv")

# ------- curatare ------------------------------------
# Remove romanian characters.
# NFKD decomposition strips every diacritic, including both the comma-below
# (ș, ț) and the cedilla (ş, ţ) variants used interchangeably on the site.
def remove_diacritics(val):
    if isinstance(val, str):
        val = unicodedata.normalize("NFKD", val)
        val = "".join(ch for ch in val if not unicodedata.combining(ch))
    return val


df = df.map(remove_diacritics)

# Romanian counties (without diacritics). Multi-word names are listed so that
# e.g. "SATU MARE" is not truncated to "SATU".
COUNTIES = [
    "ALBA", "ARAD", "ARGES", "BACAU", "BIHOR", "BISTRITA-NASAUD", "BOTOSANI",
    "BRAILA", "BRASOV", "BUZAU", "CALARASI", "CARAS-SEVERIN", "CLUJ",
    "CONSTANTA", "COVASNA", "DAMBOVITA", "DOLJ", "GALATI", "GIURGIU", "GORJ",
    "HARGHITA", "HUNEDOARA", "IALOMITA", "IASI", "ILFOV", "MARAMURES",
    "MEHEDINTI", "MURES", "NEAMT", "OLT", "PRAHOVA", "SALAJ", "SATU MARE",
    "SIBIU", "SUCEAVA", "TELEORMAN", "TIMIS", "TULCEA", "VALCEA", "VASLUI",
    "VRANCEA", "BUCURESTI",
]
county_alt = "|".join(sorted((re.escape(c) for c in COUNTIES), key=len, reverse=True))
COUNTY_REGEX = rf"(?:JUDET[UI]L|MUNICIPIUL)\s+({county_alt})\b"

# Typos found in the source titles
COUNTY_TYPOS = {
    "TELELORMAN": "TELEORMAN", "SATU MATE": "SATU MARE", "DAMBOVIITA": "DAMBOVITA",
    "MEHENDINTI": "MEHEDINTI", "VACEA": "VALCEA",
}


def normalize_title(title):
    if not isinstance(title, str):
        return title
    title = title.upper()
    # "CARAS - SEVERIN" / "CARAS SEVERIN" -> "CARAS-SEVERIN"
    title = re.sub(r"\b(CARAS|BISTRITA)\s*(?:-\s*|\s+)(SEVERIN|NASAUD)\b", r"\1-\2", title)
    for wrong, right in COUNTY_TYPOS.items():
        title = re.sub(rf"\b{wrong}\b", right, title)
    return title

# Wrapped differences larger than this are treated as data errors
# (e.g. resumption announced 5 minutes *before* the report -> 1435 min).
MAX_WRAPPED_MINUTES = 12 * 60


# Function to calculate the difference in minutes from 2 (HH:MM) string formats
def minute_difference(start, end):
    try:
        fmt = "%H:%M"
        t1 = datetime.strptime(start, fmt)
        t2 = datetime.strptime(end, fmt)
    except (TypeError, ValueError):
        return None
    delta = (t2 - t1).total_seconds() / 60
    if delta < 0:
        delta += 24 * 60
        if delta > MAX_WRAPPED_MINUTES:
            return None
    return int(delta)


# Clean values that should not be null
df = df[~df["Ora"].isna()]
df = df[~df["Descriere"].isna()]
# The standard start of the description is "Centrul INFOTRAFIC". Other contents indicate bad scraped data.
df = df[df["Descriere"].str.contains("Centrul INFOTRAFIC")]
df = df.drop_duplicates(subset="Descriere", keep="first")
# Create a new column with the region where the incidents took place
df["Judet"] = df["Titlu"].map(normalize_title).str.extract(COUNTY_REGEX, expand=False)
df["Judet"] = df["Judet"].fillna("Nementionat")

# Create a new column with the roads where the incidents took place
df["Drum"] = df["Titlu"].str.extract(r'\b((?:DEX|D[NJCE]{1,2}|A)\s?\d+[A-Z]?|DNCB|DN\s?CB|CENTURA\s+[A-Z]+)', expand=False)
df["Drum"] = df["Drum"].str.replace(r'([A-Z]+)\s?(\d+)([A-Z]?)', r'\1 \2\3', regex=True)
df["Drum"] = df["Drum"].str.replace("DN CB", "DNCB", regex=False)
df["Drum"] = df["Drum"].str.replace("CENTURA BUCURESTI", "DNCB", regex=False)

# ------- traffic resumption ----------------------------
# Three accepted forms, in order of priority:
#   "in 30 (de) minute"     -> duration in minutes
#   "in 2 ore" / "o ora"    -> duration in hours
#   "dupa ora 16.30"        -> clock time (HH:MM); requires "ora"/"orei" in
#                              front so that dates such as "30.01.2024" are not
#                              mistaken for a time.
NUM_WORDS = {"o": 1, "una": 1, "un": 1, "doua": 2, "trei": 3, "patru": 4, "cinci": 5}
RE_MINUTES = re.compile(r"\b(\d{1,3})\s+(?:de\s+)?minute\b", re.I)
RE_HOURS = re.compile(r"\b(\d{1,2}|o|una|un|doua|trei|patru|cinci)\s+(?:de\s+)?or[ae]\b(?!\s*\d)", re.I)
RE_CLOCK = re.compile(r"\bor(?:a|ei)\s+([01]?\d|2[0-3])[:.]([0-5]\d)\b", re.I)


def parse_resumption(text):
    """Return ('min', minutes) or ('clock', 'HH:MM') or None."""
    if not isinstance(text, str):
        return None
    m = RE_MINUTES.search(text)
    if m:
        return ("min", int(m.group(1)))
    m = RE_HOURS.search(text)
    if m:
        tok = m.group(1).lower()
        hours = int(tok) if tok.isdigit() else NUM_WORDS[tok]
        return ("min", hours * 60)
    m = RE_CLOCK.search(text)
    if m:
        return ("clock", f"{int(m.group(1)):02d}:{m.group(2)}")
    return None


#  cleaning date, time + transform columns in a timestamp
df["Data"] = df["Data"].str.replace("Data: ", "", regex=False)
df["Ora"] = df["Ora"].str.replace("Ora: ", "", regex=False).str.strip()


def resumption_minutes(row):
    # prefer the dedicated "Reluare" paragraph, fall back on the description
    parsed = parse_resumption(row["Reluare"]) or parse_resumption(row["Descriere"])
    if parsed is None:
        return None
    kind, value = parsed
    if kind == "min":
        return value
    return minute_difference(row["Ora"], value)


df["Reluare"] = pd.to_numeric(df.apply(resumption_minutes, axis=1), errors="coerce").astype("Int64")

months_translate = {
    "Ianuarie": "January", "Februarie": "February", "Martie": "March",
    "Aprilie": "April", "Mai": "May", "Iunie": "June",
    "Iulie": "July", "August": "August", "Septembrie": "September",
    "Octombrie": "October", "Noiembrie": "November", "Decembrie": "December"
}

# Replace romanian month with english month in string
months_regex = r"\b(" + "|".join(months_translate) + r")\b"
data_en = df["Data"].str.replace(months_regex, lambda x: months_translate[x.group(0)], regex=True)
# Transform "Data" field in date field
date_type_data = pd.to_datetime(data_en, format=r'%d %B %Y', errors="coerce")
# create timestamp
data_ora = pd.to_datetime(date_type_data.dt.strftime(r"%Y-%m-%d") + ' ' + df["Ora"], format="%Y-%m-%d %H:%M", errors='coerce')
df["data_ora"] = data_ora

#Write the clean version in csv
df.columns = ["nr_crt", "titlu", "data", "ora", "descriere", "reluare", "judet", "drum", "data_ora"]
coloane = ["data_ora", "judet", "drum", "descriere", "reluare"]
df[coloane].to_csv(ROOT / "data" / "accidente_clean.csv", index=False, encoding="utf-8")
