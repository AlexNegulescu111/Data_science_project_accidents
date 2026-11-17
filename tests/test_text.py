"""Tests for the text-extraction rules. Most sentences are taken verbatim
from the scraped data, including the cases the first version got wrong."""
import numpy as np
import pytest

from accidents.text import (convert_to_number, extract_county, extract_deaths,
                            extract_road, extract_vehicles, extract_wounded,
                            minute_difference, parse_resumption,
                            remove_diacritics, resumption_minutes)


# --- diacritics --------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Poliției Române", "Politiei Romane"),      # comma-below ț
    ("Poliţiei Române", "Politiei Romane"),      # cedilla ţ (was not stripped)
    ("JUDEȚUL CONSTANȚA", "JUDETUL CONSTANTA"),
    ("JUDEŢUL BRAŞOV", "JUDETUL BRASOV"),
    ("ăâîșțĂÂÎȘȚ", "aaistAAIST"),
])
def test_remove_diacritics(raw, expected):
    assert remove_diacritics(raw) == expected


def test_remove_diacritics_keeps_non_strings():
    assert remove_diacritics(None) is None
    assert remove_diacritics(3) == 3


# --- county / road -----------------------------------------------------------

@pytest.mark.parametrize("title, county", [
    ("JUDEȚUL SIBIU: TRAFIC ÎNGREUNAT PE AUTOSTRADA A1, DIN CAUZA UNUI ACCIDENT", "SIBIU"),
    ("JUDEŢUL CONSTANŢA: TRAFIC BLOCAT PE DN 39", "CONSTANTA"),   # cedilla
    ("JUDEȚUL SATU MARE: TRAFIC OPRIT PE DN 19A", "SATU MARE"),   # two words
    ("JUDETUL CARAS - SEVERIN: TRAFIC INGREUNAT PE DN 6", "CARAS-SEVERIN"),
    ("JUDETUL CARAS SEVERIN - TRAFIC OPRIT PE DN6", "CARAS-SEVERIN"),
    ("JUDETUL BISTRITA-NASAUD: TRAFIC BLOCAT PE DN 17", "BISTRITA-NASAUD"),
    ("JUDETIL SUCEAVA: TRAFIC OPRIT PE DN 2P", "SUCEAVA"),         # typo in source
    ("JUDETUL TELELORMAN: TRAFIC ALTERNATIV PE DN 6", "TELEORMAN"),
    ("JUDETUL VACEA: TRAFIC INGREUNAT PE DN 7", "VALCEA"),
])
def test_extract_county(title, county):
    assert extract_county(title) == county


@pytest.mark.parametrize("title", [
    "TRAFIC RESTRICTIONAT PE A1 BUCURESTI-PITESTI",   # motorway, no county
    None,
])
def test_extract_county_missing(title):
    assert extract_county(title) is None


@pytest.mark.parametrize("title, road", [
    ("JUDEȚUL SIBIU: TRAFIC ÎNGREUNAT PE AUTOSTRADA A1", "A 1"),
    ("JUDETUL SATU MARE: TRAFIC OPRIT PE DN 19A, DIN CAUZA", "DN 19A"),
    ("JUDETUL CARAS SEVERIN - TRAFIC OPRIT PE DN6 DIN CAUZA", "DN 6"),
    ("TRAFIC BLOCAT PE DN CB", "DNCB"),
    ("TRAFIC OPRIT PE DJ 102", "DJ 102"),
])
def test_extract_road(title, road):
    assert extract_road(title) == road


# --- resumption --------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Se estimeaza reluarea traficului in 30 de minute.", ("min", 30)),
    ("Se estimeaza reluarea circulatiei in 30 minute.", ("min", 30)),
    ("Se estimeaza reluarea traficului in aproximativ o ora.", ("min", 60)),
    ("Se estimeaza reluarea traficului in conditii normale intr-o ora.", ("min", 60)),
    ("Se estimeaza reluarea normala a traficului in aproximativ 2 ore.", ("min", 120)),
    ("Se estimează reluarea traficului în condiții normale după ora 16.30.", ("clock", "16:30")),
    ("Se estimeaza reluarea traficului in jurul orei 9:05.", ("clock", "09:05")),
    # a date must not be read as a clock time
    ("Traficul este deviat prin DN58 Anina – Resita pana la data de 30.01.2024.", None),
    (None, None),
])
def test_parse_resumption(text, expected):
    assert parse_resumption(text) == expected


@pytest.mark.parametrize("start, end, expected", [
    ("15:50", "16:30", 40),
    ("23:30", "01:00", 90),      # crosses midnight
    ("16:35", "16:30", None),    # 1435 min after wrapping -> data error
    ("bad", "16:30", None),
])
def test_minute_difference(start, end, expected):
    assert minute_difference(start, end) == expected


def test_resumption_falls_back_on_description():
    desc = "Centrul INFOTRAFIC ... Se estimeaza reluarea circulatiei in 45 de minute."
    assert resumption_minutes("10:00", None, desc) == 45
    assert resumption_minutes("10:00", "Se estimeaza reluarea dupa ora 11.15.", desc) == 75


# --- numbers / vehicles ------------------------------------------------------

@pytest.mark.parametrize("token, value", [("3", 3), ("doua", 2), ("unei", 1), ("Unuia", 1), ("ceva", 0)])
def test_convert_to_number(token, value):
    assert convert_to_number(token) == value


def test_extract_vehicles_ansambluri_rutiere():
    # "ansambluri rutiere" was broken by a missing "|" in the regex
    text = ("traficul este oprit temporar pe DN 6 din cauza unui accident in care "
            "sunt implicate doua ansambluri rutiere.")
    assert extract_vehicles(text)[0] == ("ansambluri rutiere", "doua")


def test_extract_vehicles_pads_with_nan():
    vehicles = extract_vehicles("un autoturism si un ansamblu rutier au fost implicate")
    assert vehicles[:2] == [("autoturism", "un"), ("ansamblu rutier", "un")]
    assert all(np.isnan(v) and np.isnan(n) for v, n in vehicles[2:])


def test_extract_vehicles_autocisterne():
    assert extract_vehicles("au fost implicate doua autocisterne")[0] == ("autocisterne", "doua")


# --- victims -----------------------------------------------------------------

@pytest.mark.parametrize("text, deaths", [
    # cases the first version reported as 0 deaths
    ("Din nefericire, unul dintre minori a decedat.", 1),
    ("Pasagerul de pe scaunul din fata a decedat.", 1),
    ("Pietonul a decedat.", 1),
    ("Accidentul s-a soldat cu decesul unuia dintre conducatorii auto si cu ranirea altor doua persoane.", 1),
    # explicit counts
    ("Impactul s-a soldat cu decesul celor doi participanti la trafic.", 2),
    ("Doua persoane (din acest autoturism) au decedat.", 2),
    ("In urma impactului 2 dintre acestea au decedat.", 2),
    ("O persoana a decedat si alte patru sunt ranite.", 1),
    ("Un barbat de 45 de ani si-a pierdut viata.", 1),       # age is not a count
    # no death
    ("Nu au rezultat victime.", 0),
    ("Un autoturism a intrat in coliziune cu un microbuz pe DN 7.", 0),
])
def test_extract_deaths(text, deaths):
    assert extract_deaths(text) == deaths


@pytest.mark.parametrize("text, wounded", [
    ("In urma impactului a rezultat ranirea unei persoane.", 1),
    ("In urma impactului au rezultat 3 victime.", 3),
    ("Impactul s-a soldat cu ranirea a doua persoane.", 2),
    ("Trei persoane au suferit leziuni usoare.", 3),
    ("Din primele informatii au rezultat 9 victime, care urmeaza a fi investigate medical.", 9),
    ("In urma careia 8 persoane au necesitat transportul la spital, constiente.", 8),
    # road number / kilometre must not be read as a victim count
    ("Pe DN 7, la kilometrul 162, o persoana a fost ranita.", 1),
    # negation, also when "nu" is only a substring of another word ("continua")
    ("Nu au rezultat victime.", 0),
    ("Fara victime.", 0),
    ("Circulatia continua pe un fir. Un sofer a fost ranit.", 1),
])
def test_extract_wounded(text, wounded):
    assert extract_wounded(text) == wounded
