from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import re

import numpy as np
import pandas as pd

# Load the cleaned data
df = pd.read_csv(ROOT / "data" / "accidente_clean.csv")

numere_text = {
    "o": 1, "un": 1, "una": 1, "unu": 1, "unei": 1, "unui": 1,
    "unul": 1, "unuia": 1, "uneia": 1, "doi": 2,
    "doua": 2, "trei": 3, "patru": 4, "cinci": 5,
    "sase": 6, "sapte": 7, "opt": 8, "noua": 9, "zece": 10
}
# Number token with word boundaries on BOTH sides, so that "o" does not match
# the start of "ora" and "un" does not match "unde".
regex_numere = r"\b(\d{1,2}|un(?:ui|ei|a|u|ul|uia)?|uneia|o|doua|doi|trei|patru|cinci|sase|sapte|opt|noua|zece)\b"

# ------- vehicles ----------------------------------------------------------
regex_auto = (
    r"\s+(autoturism(?:e)?|autocamion|autocamioane|camioane|motocicleta|motociclete|"
    r"autoutilitara|autoutilitare|autotren(?:uri)?|autovehicul(?:e)?|microbuz(?:e)?|"
    r"autobuz(?:e)?|tren(?:uri)?|ansamblu rutier|ansambluri rutiere|"
    r"autocisterna|autocisterne|cisterna|cisterne|tiruri|tir|vehicul(?:e)?|"
    r"biciclet[ae]|scuter(?:e)?|basculant[ae]|excavator|buldozer|ambulant[ae]|"
    r"autocar(?:e)?|dub[ae]|ATV|tractor(?:are)?)\b"
)
regex_auto_full = re.compile(regex_numere + regex_auto, re.IGNORECASE)


def extract_vehicles(text):
    matches = regex_auto_full.findall(str(text))  # list of (nr, vehicle)
    results = [(veh.lower(), nr.lower()) for nr, veh in matches[:3]]  # max 3 tuples
    while len(results) < 3:
        results.append((np.nan, np.nan))
    return results


results = df["descriere"].apply(extract_vehicles)
df[["veh_1", "nr_veh_1", "veh_2", "nr_veh_2", "veh_3", "nr_veh_3"]] = pd.DataFrame(
    [(v1, n1, v2, n2, v3, n3) for (v1, n1), (v2, n2), (v3, n3) in results],
    index=df.index,
)

# ------- wounded / deaths --------------------------------------------------
regex_deaths = (
    r"\b(dece(?:s|sul|se|dat\w*)|decedat\w*|moartea|mortal\w*|murit|"
    r"si-a\s+pierdut\s+viata|si-au\s+pierdut\s+viata|fara\s+viata|lipsit\w*\s+de\s+viata)\b"
)
regex_wounded = (
    r"\b(ranit\w*|ranire\w*|leziuni|rani|la\s+spital|ingrijiri|"
    r"incarcerat\w*|descarcerat\w*|victim[ae])\b"
)
# Nouns that a victim count refers to. A number only counts as a victim count
# when it is followed (within two words) by one of these, or directly by the
# keyword itself - this skips road numbers ("DN 7"), kilometres, hours, ages.
regex_persons = (
    r"(?:persoan[ae]|victim[ae]|ranit\w*|decedat\w*|pasager\w*|conducator\w*|sofer\w*|"
    r"copi[il]|barbat\w*|femei\w*|tiner\w*|minor\w*|biciclist\w*|pieton\w*|motociclist\w*|"
    r"ocupant\w*|calator\w*|participant\w*|dintre)"
)
regex_count = re.compile(regex_numere + r"(?:\s+\w+){0,2}?\s+" + regex_persons)
regex_negation = re.compile(r"\b(nu|fara)\b")


def convert_to_number(val):
    """Convert digits or Romanian number words into int (0 when unknown)."""
    val = str(val).lower()
    if val.isdigit():
        return int(val)
    return numere_text.get(val, 0)


def extract_wounded_deaths(text, regex):
    """Return the number of victims of the type described by `regex`.

    The text is split in sentence fragments; the first fragment that mentions
    the keyword and is not negated ("nu au rezultat victime", "fara victime")
    decides the count. If no explicit count is found next to a person noun,
    a mention of the keyword counts as 1.
    """
    text = str(text).lower()
    # split in clauses: punctuation, and "... si cu ranirea ..." / "... iar ..."
    for sen in re.split(r"[.?!,;]|\bsi cu\b|\biar\b", text):
        if not re.search(regex, sen):
            continue
        # "fara viata" (lifeless) is a death expression, not a negation
        if regex_negation.search(re.sub(r"fara\s+viata", "", sen)):
            continue
        match = regex_count.search(sen)
        if match:
            return convert_to_number(match.group(1)) or 1
        return 1
    return 0


df["nr_raniti"] = df["descriere"].apply(lambda x: extract_wounded_deaths(x, regex_wounded))
df["nr_morti"] = df["descriere"].apply(lambda x: extract_wounded_deaths(x, regex_deaths))

for col in ["nr_veh_1", "nr_veh_2", "nr_veh_3"]:
    df[col] = df[col].fillna("0").astype(str).apply(convert_to_number)
df["total_nr_veh"] = df["nr_veh_1"] + df["nr_veh_2"] + df["nr_veh_3"]

df.to_csv(ROOT / "data" / "accidente_processed.csv", index=False)
