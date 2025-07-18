from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import numpy as np
from pandasgui import show
import re

# Încarcă datele curățate
df = pd.read_csv(ROOT / "data" / accidente_clean.csv")

regex_numere = r"(?i)\b(\d+|un(?:ui)?|o|unei|doua|doi|trei|patru|cinci|sase|sapte|opt|noua|zece)"
regex_auto = (r"\s(autoturism(?:e)?|autocamion|autocamioane|camioane|motocicleta|motociclete|autoutilitara|autoutilitare|"
r"autotren|autovehicul(?:e)?|microbuz(?:e)?|autobuz(?:e)?|tren(?:uri)?|ansamblu rutier|ansambluri rutiere"
r"autocisterna(?:e)?|cisterna(?:e)?|tiruri|tir|vehicul(?:e)?|bicicleta|scuter|"
r"basculanta|excavator|buldozer|ambulanta|autocar|duba|ATV|tractor)\b")
regex_wounded = r"(ranit[a]?|ranirea|leziuni|rani|la spital|ingrijiri|incarcerat[a]?|descarcerare|victima|victime)"
regex_deaths = r"(deces[ul]?|decedat|decedata|moartea|mortal|murit)"

regex_auto_full = re.compile(regex_numere+regex_auto)

def extract_vehicles(text):
    text = str(text)
    matches = re.findall(regex_auto_full, text)  # obține direct listă de (nr, vehicul)

    results = [(veh.lower(), nr.lower()) for nr, veh in matches[:3]]  # max 3 tuple
    while len(results) < 3:
        results.append((np.nan, np.nan))
    return results
results = df["descriere"].apply(extract_vehicles)
df[["veh_1", "nr_veh_1", "veh_2", "nr_veh_2", "veh_3", "nr_veh_3"]] = pd.DataFrame(
    [(v1, n1, v2, n2, v3, n3) for (v1, n1), (v2, n2), (v3, n3) in results]
)
# Function to extract deaths and wounded from text
def extract_wounded_deaths(text, regex):
    text = str(text).lower()
    sentence = re.split(r"[.?!,;]", text)
    for sen in sentence:
        if re.search(regex, sen):
            if "nu" in sen or "fara" in sen:
                return 0
            match = re.search(regex_numere, sen)
            if match:
                return convert_to_number(match.group(1))
            else:
                return 1
    return 0


numere_text = {
    "o": 1, "un": 1, "una": 1, "unu": 1, "unei": 1, "unui": 1, "doi": 2,
    "doua": 2, "trei": 3, "patru": 4, "cinci": 5,
    "sase": 6, "sapte": 7, "opt": 8, "noua": 9, "zece": 10
}
# function to convert text values meaning numbers into int 
def convert_to_number(val):
    val = val.lower()
    if val.isdigit():
        return int(val)
    return numere_text.get(val, 0) 
# extract number of wounded and detahs
df["nr_raniti"] = df["descriere"].apply(lambda x: extract_wounded_deaths(x, regex_wounded))
df["nr_morti"] = df["descriere"].apply(lambda x: extract_wounded_deaths(x, regex_deaths))
df.loc[df["nr_morti"] >= 10, "nr_morti"] = 1 # extraction result >= 10 is the age of the person

df["nr_veh_1"] = df["nr_veh_1"].fillna("0").astype(str)
df["nr_veh_1"] = df["nr_veh_1"].apply(convert_to_number)

df["nr_veh_2"] = df["nr_veh_2"].fillna("0").astype(str)
df["nr_veh_2"] = df["nr_veh_2"].apply(convert_to_number)

df["nr_veh_3"] = df["nr_veh_3"].fillna("0").astype(str)
df["nr_veh_3"] = df["nr_veh_3"].apply(convert_to_number)
df["total_nr_veh"] = df["nr_veh_1"] + df["nr_veh_2"] + df["nr_veh_3"]

df.to_csv(ROOT / "data" / accidente_processed.csv")
