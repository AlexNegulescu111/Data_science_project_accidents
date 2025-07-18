from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
from datetime import datetime
from pandasgui import show
df = pd.read_csv(ROOT / "data" / accidente.csv")

# ------- curatare ------------------------------------
# Remove romanian characters
diacritice = {
    "ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t",
    "Ă": "A", "Â": "A", "Î": "I", "Ș": "S", "Ț": "T"
}

# Function to remove romanian characters
def remove_diacritics(val):
    if isinstance(val, str):
        for k, v in diacritice.items():
            val = val.replace(k, v)
    return val


# Apply on remove_diacritics DataFrame
df = df.applymap(remove_diacritics)

#Function to calculate the difference in minutes from 2 (HH:MM) string formats
def minute_difference(start, end):
    try:
        fmt = "%H:%M"
        t1 = datetime.strptime(start, fmt)
        t2 = datetime.strptime(end, fmt)
        delta = (t2 - t1).total_seconds()/60
        if delta <0:
            delta+=24*60
        return int(delta)
    except:
        return None


# Clean values that should not be null
df = df[~df["Ora"].isna()] 
df = df[~df["Descriere"].isna()]
# The standard start of the description is "Centrul INFOTRAFIC". Other contents indicate bad scraped data.
df = df[df["Descriere"].str.contains("Centrul INFOTRAFIC")]
df = df.drop_duplicates(subset="Descriere", keep="first")
# Create a new column with the region where the incidents took place
df["Judet"] = df["Titlu"].str.extract(r'JUDE.{3}\s([A-Z]+(?:-(?!\s)[A-Z]+)?)', expand=False)
df["Judet"] = df["Judet"].fillna("Nementionat")

# Create a new column with the roads where the incidents took place
df["Drum"] = df["Titlu"].str.extract(r'\b((?:DEX|D[NJCE]{1,2}|A)\s?\d+[A-Z]?|DNCB|DN\s?CB|CENTURA\s+[A-Z]+)', expand=False)
df["Drum"] = df["Drum"].str.replace(r'([A-Z]+)\s?(\d+)([A-Z]?)', r'\1 \2\3', regex=True)
df["Drum"] = df["Drum"].str.replace("DN CB", "DNCB", regex=False)
df["Drum"] = df["Drum"].str.replace("CENTURA BUCURESTI", "DNCB", regex=False)

#extract the duration or the hour of the traffic ressumption from Descriere
regex = r'\b(\d{1,2})(?=\s(?:de\s)?minute)|\b(\d{1,2}[:.]\d{2})'
mask = df["Reluare"].isna()
extrase = df.loc[mask, "Descriere"].str.extract(regex, expand=False)
df.loc[mask, "Reluare"] = extrase[0].combine_first(extrase[1])

# Clean all the data in Reluare.
df["Reluare"] = df["Reluare"].str.extract(regex, expand = False).apply(lambda x: x[0] if pd.notnull(x[0]) else x[1], axis = 1)

# Transform the hour of traffic ressumption in minutes
df["Reluare"] = df["Reluare"].str.replace(".", ":", regex=False)


#  cleaning date, time + transform columns in a timestamp 
df["Data"] = df["Data"].str.replace("Data: ", "", regex=False)
df["Ora"] = df["Ora"].str.replace("Ora: ", "", regex=False)

months_translate = {
    "Ianuarie": "January", "Februarie": "February", "Martie": "March",
    "Aprilie": "April", "Mai": "May", "Iunie": "June",
    "Iulie": "July", "August": "August", "Septembrie": "September",
    "Octombrie": "October", "Noiembrie": "November", "Decembrie": "December"
}
# Modify Reluare column so that it shows only the interval traffic was blocked
mask = df["Reluare"].str.contains(":", regex=False, na=False)
df.loc[mask, "Reluare"] = df.loc[mask].apply(lambda x: minute_difference(x["Ora"], x["Reluare"]), axis = 1)
df["Reluare"] = pd.to_numeric(df["Reluare"], errors="coerce").astype("Int64")

# Replace romanian month with english month in string
data_en = df["Data"].str.replace("|".join(months_translate.keys()), lambda x: months_translate[x.group(0)], regex=True)
# Transform "Data" field in date field
date_type_data = pd.to_datetime(data_en, format= r'%d %B %Y', errors="coerce")
# create timestamp
data_ora = pd.to_datetime(date_type_data.dt.strftime(r"%Y %m %d") + ' ' + df["Ora"], errors='coerce')
df["data_ora"] = data_ora

#Write the clean version in csv
df.columns = ["nr_crt", "titlu", "data", "ora","descriere","reluare", "judet", "drum", "data_ora"]
coloane = ["data_ora", "judet", "drum", "descriere", "reluare"]
df[coloane].to_csv(ROOT / "data" / accidente_clean.csv", index = False, encoding = "utf-8")