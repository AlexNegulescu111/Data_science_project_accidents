"""Paths and constants shared by the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw_reports.csv"
CLEAN_CSV = DATA_DIR / "accidents_clean.csv"
PROCESSED_CSV = DATA_DIR / "accidents_processed.csv"

VISUALS_DIR = ROOT / "visuals"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

# Label used when a title does not name a county (mostly motorway reports)
UNKNOWN_COUNTY = "UNKNOWN"

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

# Typos found in the source titles
COUNTY_TYPOS = {
    "TELELORMAN": "TELEORMAN", "SATU MATE": "SATU MARE", "DAMBOVIITA": "DAMBOVITA",
    "MEHENDINTI": "MEHEDINTI", "VACEA": "VALCEA",
}

MONTHS_RO_EN = {
    "Ianuarie": "January", "Februarie": "February", "Martie": "March",
    "Aprilie": "April", "Mai": "May", "Iunie": "June",
    "Iulie": "July", "August": "August", "Septembrie": "September",
    "Octombrie": "October", "Noiembrie": "November", "Decembrie": "December",
}

NUMBER_WORDS = {
    "o": 1, "un": 1, "una": 1, "unu": 1, "unei": 1, "unui": 1,
    "unul": 1, "unuia": 1, "uneia": 1, "doi": 2,
    "doua": 2, "trei": 3, "patru": 4, "cinci": 5,
    "sase": 6, "sapte": 7, "opt": 8, "noua": 9, "zece": 10,
}

# Wrapped clock differences larger than this are treated as data errors
# (e.g. resumption announced 5 minutes *before* the report -> 1435 min).
MAX_WRAPPED_MINUTES = 12 * 60

# Vehicle names normalised to one singular form (used in the analysis)
VEHICLE_SINGULAR = {
    "autoturisme": "autoturism",
    "autocamioane": "autocamion",
    "camioane": "autocamion",
    "autovehicule": "autovehicul",
    "autoutilitare": "autoutilitara",
    "microbuze": "microbuz",
    "autobuze": "autobuz",
    "motociclete": "motocicleta",
    "tiruri": "tir",
    "vehicule": "autovehicul",
    "vehicul": "autovehicul",
    "ansambluri rutiere": "ansamblu rutier",
    "autocisterne": "autocisterna",
    "cisterne": "cisterna",
    "trenuri": "tren",
    "autotrenuri": "autotren",
    "biciclete": "bicicleta",
    "scutere": "scuter",
    "basculante": "basculanta",
    "ambulante": "ambulanta",
    "autocare": "autocar",
    "dube": "duba",
    "tractoare": "tractor",
}

GEO_COUNTIES = DATA_DIR / "geo" / "romania_counties.geojson"

# English display labels for the (Romanian) vehicle types found in the text
VEHICLE_LABELS_EN = {
    "autoturism": "car",
    "autoutilitara": "van",
    "autocamion": "truck",
    "autotren": "truck with trailer",
    "ansamblu rutier": "articulated lorry",
    "tir": "TIR lorry",
    "autocisterna": "tanker truck",
    "cisterna": "tanker",
    "microbuz": "minibus",
    "autobuz": "bus",
    "autocar": "coach",
    "motocicleta": "motorcycle",
    "bicicleta": "bicycle",
    "scuter": "scooter",
    "tractor": "tractor",
    "autovehicul": "vehicle (unspecified)",
    "tren": "train",
    "basculanta": "dump truck",
    "duba": "van (duba)",
    "ambulanta": "ambulance",
    "ATV": "ATV",
}
