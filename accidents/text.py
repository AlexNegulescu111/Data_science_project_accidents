"""Pure text-processing functions: no I/O, no global state, easy to test."""
import re
import unicodedata
from datetime import datetime

import numpy as np

from accidents.config import (COUNTIES, COUNTY_TYPOS, MAX_WRAPPED_MINUTES,
                              NUMBER_WORDS)

# ---------------------------------------------------------------------------
# Diacritics
# ---------------------------------------------------------------------------

def remove_diacritics(val):
    """Strip every diacritic.

    NFKD decomposition handles both the comma-below (ș, ț) and the cedilla
    (ş, ţ) variants that the source site uses interchangeably.
    Non-string values are returned unchanged.
    """
    if not isinstance(val, str):
        return val
    val = unicodedata.normalize("NFKD", val)
    return "".join(ch for ch in val if not unicodedata.combining(ch))


# ---------------------------------------------------------------------------
# County / road from the title
# ---------------------------------------------------------------------------

_county_alt = "|".join(sorted((re.escape(c) for c in COUNTIES), key=len, reverse=True))
RE_COUNTY = re.compile(rf"(?:JUDET[UI]L|MUNICIPIUL)\s+({_county_alt})\b")
RE_ROAD = re.compile(r"\b((?:DEX|D[NJCE]{1,2}|A)\s?\d+[A-Z]?|DNCB|DN\s?CB|CENTURA\s+[A-Z]+)")


def normalize_title(title):
    """Upper-case, join split county names and fix known typos."""
    if not isinstance(title, str):
        return title
    title = remove_diacritics(title).upper()
    # "CARAS - SEVERIN" / "CARAS SEVERIN" -> "CARAS-SEVERIN"
    title = re.sub(r"\b(CARAS|BISTRITA)\s*(?:-\s*|\s+)(SEVERIN|NASAUD)\b", r"\1-\2", title)
    for wrong, right in COUNTY_TYPOS.items():
        title = re.sub(rf"\b{wrong}\b", right, title)
    return title


def extract_county(title):
    """Return the county named in a title, or None."""
    if not isinstance(title, str):
        return None
    m = RE_COUNTY.search(normalize_title(title))
    return m.group(1) if m else None


def extract_road(title):
    """Return the road named in a title in a canonical form ("DN 7", "A 1")."""
    if not isinstance(title, str):
        return None
    m = RE_ROAD.search(remove_diacritics(title))
    if not m:
        return None
    road = re.sub(r"([A-Z]+)\s?(\d+)([A-Z]?)", r"\1 \2\3", m.group(1))
    road = road.replace("DN CB", "DNCB").replace("CENTURA BUCURESTI", "DNCB")
    return road


# ---------------------------------------------------------------------------
# Traffic resumption
# ---------------------------------------------------------------------------
# Three accepted forms, in order of priority:
#   "in 30 (de) minute"     -> duration in minutes
#   "in 2 ore" / "o ora"    -> duration in hours
#   "dupa ora 16.30"        -> clock time; requires "ora"/"orei" in front so
#                              that dates such as "30.01.2024" are not read as
#                              a time.
_HOUR_WORDS = {"o": 1, "una": 1, "un": 1, "doua": 2, "trei": 3, "patru": 4, "cinci": 5}
RE_MINUTES = re.compile(r"\b(\d{1,3})\s+(?:de\s+)?minute\b", re.I)
RE_HOURS = re.compile(r"\b(\d{1,2}|o|una|un|doua|trei|patru|cinci)\s+(?:de\s+)?or[ae]\b(?!\s*\d)", re.I)
RE_CLOCK = re.compile(r"\bor(?:a|ei)\s+([01]?\d|2[0-3])[:.]([0-5]\d)\b", re.I)


def parse_resumption(text):
    """Return ("min", minutes), ("clock", "HH:MM") or None."""
    if not isinstance(text, str):
        return None
    text = remove_diacritics(text)
    m = RE_MINUTES.search(text)
    if m:
        return ("min", int(m.group(1)))
    m = RE_HOURS.search(text)
    if m:
        tok = m.group(1).lower()
        hours = int(tok) if tok.isdigit() else _HOUR_WORDS[tok]
        return ("min", hours * 60)
    m = RE_CLOCK.search(text)
    if m:
        return ("clock", f"{int(m.group(1)):02d}:{m.group(2)}")
    return None


def minute_difference(start, end):
    """Minutes from `start` to `end` (both "HH:MM"); None when invalid.

    Crossing midnight is allowed, but a wrapped difference larger than
    MAX_WRAPPED_MINUTES is treated as a data error.
    """
    try:
        t1 = datetime.strptime(start, "%H:%M")
        t2 = datetime.strptime(end, "%H:%M")
    except (TypeError, ValueError):
        return None
    delta = (t2 - t1).total_seconds() / 60
    if delta < 0:
        delta += 24 * 60
        if delta > MAX_WRAPPED_MINUTES:
            return None
    return int(delta)


def resumption_minutes(report_time, resumption_text, description=None):
    """Minutes until traffic resumes, from the "Reluare" paragraph or,
    failing that, from the description. None when nothing usable is found."""
    parsed = parse_resumption(resumption_text) or parse_resumption(description)
    if parsed is None:
        return None
    kind, value = parsed
    if kind == "min":
        return value
    return minute_difference(report_time, value)


# ---------------------------------------------------------------------------
# Numbers, vehicles, victims
# ---------------------------------------------------------------------------
# Number token with word boundaries on BOTH sides, so that "o" does not match
# the start of "ora" and "un" does not match "unde".
REGEX_NUMBERS = r"\b(\d{1,2}|un(?:ui|ei|a|u|ul|uia)?|uneia|o|doua|doi|trei|patru|cinci|sase|sapte|opt|noua|zece)\b"

REGEX_VEHICLES = (
    r"\s+(autoturism(?:e)?|autocamion|autocamioane|camioane|motocicleta|motociclete|"
    r"autoutilitara|autoutilitare|autotren(?:uri)?|autovehicul(?:e)?|microbuz(?:e)?|"
    r"autobuz(?:e)?|tren(?:uri)?|ansamblu rutier|ansambluri rutiere|"
    r"autocisterna|autocisterne|cisterna|cisterne|tiruri|tir|vehicul(?:e)?|"
    r"biciclet[ae]|scuter(?:e)?|basculant[ae]|excavator|buldozer|ambulant[ae]|"
    r"autocar(?:e)?|dub[ae]|ATV|tractor(?:are)?)\b"
)
RE_VEHICLE_COUNT = re.compile(REGEX_NUMBERS + REGEX_VEHICLES, re.IGNORECASE)

REGEX_DEATHS = (
    r"\b(dece(?:s|sul|se|dat\w*)|decedat\w*|moartea|mortal\w*|murit|"
    r"si-a\s+pierdut\s+viata|si-au\s+pierdut\s+viata|fara\s+viata|lipsit\w*\s+de\s+viata)\b"
)
REGEX_WOUNDED = (
    r"\b(ranit\w*|ranire\w*|leziuni|rani|la\s+spital|ingrijiri|"
    r"incarcerat\w*|descarcerat\w*|victim[ae])\b"
)
# Nouns that a victim count refers to. A number only counts as a victim count
# when it is followed (within two words) by one of these - this skips road
# numbers ("DN 7"), kilometres, hours and ages.
REGEX_PERSONS = (
    r"(?:persoan[ae]|victim[ae]|ranit\w*|decedat\w*|pasager\w*|conducator\w*|sofer\w*|"
    r"copi[il]|barbat\w*|femei\w*|tiner\w*|minor\w*|biciclist\w*|pieton\w*|motociclist\w*|"
    r"ocupant\w*|calator\w*|participant\w*|dintre)"
)
RE_VICTIM_COUNT = re.compile(REGEX_NUMBERS + r"(?:\s+\w+){0,2}?\s+" + REGEX_PERSONS)
RE_NEGATION = re.compile(r"\b(nu|fara)\b")
# Clause boundaries: punctuation, "... si cu ranirea ..." and "... iar ..."
RE_CLAUSE_SPLIT = re.compile(r"[.?!,;]|\bsi cu\b|\biar\b")


def convert_to_number(val):
    """Digits or Romanian number words -> int (0 when unknown)."""
    val = str(val).lower()
    if val.isdigit():
        return int(val)
    return NUMBER_WORDS.get(val, 0)


def extract_vehicles(text, max_vehicles=3):
    """Return `max_vehicles` (vehicle, count_token) pairs, padded with NaN."""
    matches = RE_VEHICLE_COUNT.findall(remove_diacritics(str(text)))
    results = [(veh.lower(), nr.lower()) for nr, veh in matches[:max_vehicles]]
    while len(results) < max_vehicles:
        results.append((np.nan, np.nan))
    return results


def extract_victims(text, regex):
    """Number of victims of the type described by `regex`.

    The text is split in clauses; the first clause that mentions the keyword
    and is not negated ("nu au rezultat victime", "fara victime") decides the
    count. Without an explicit count next to a person noun, a mention counts
    as 1.
    """
    text = remove_diacritics(str(text)).lower()
    for clause in RE_CLAUSE_SPLIT.split(text):
        if not re.search(regex, clause):
            continue
        # "fara viata" (lifeless) is a death expression, not a negation
        if RE_NEGATION.search(re.sub(r"fara\s+viata", "", clause)):
            continue
        match = RE_VICTIM_COUNT.search(clause)
        if match:
            return convert_to_number(match.group(1)) or 1
        return 1
    return 0


def extract_deaths(text):
    return extract_victims(text, REGEX_DEATHS)


def extract_wounded(text):
    return extract_victims(text, REGEX_WOUNDED)
