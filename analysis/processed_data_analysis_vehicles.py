from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv(ROOT / "data" / "accidente_processed.csv")
# 2. Clean data
df = df[~df["judet"].str.contains("nementionat", case=False)]
df = df[~df["drum"].str.contains("nan", case=False, na=False)]
df = df.dropna(subset="veh_1")

plural_to_singular = {
    "autoturisme": "autoturism",
    "autocamioane": "autocamion",
    "autovehicule": "autovehicul",
    "autoutilitare": "autoutilitara",
    "microbuze": "microbuz",
    "motociclete": "motocicleta",
    "tiruri": "tir",
    "vehicule": "autovehicul",
    "vehicul": "autovehicul"
}

df[["veh_1", "veh_2", "veh_3"]] = df[["veh_1", "veh_2", "veh_3"]].replace(plural_to_singular)
df_deaths = df[df["nr_morti"] > 0]
df_wounded = df[df["nr_raniti"] > 0]

# ---------------------------------------------------------------------------------------------------------
# process the vehicles involved in dangerous accidents

df_dangerous = pd.concat([df_deaths, df_wounded[df_wounded["nr_raniti"]>=2]]).drop_duplicates()

cars_involved_1 = df_dangerous.groupby("veh_1")["nr_veh_1"].sum().sort_values(ascending=False)
cars_involved_2 = df_dangerous.groupby("veh_2")["nr_veh_2"].sum().sort_values(ascending=False)
cars_involved_3 = df_dangerous.groupby("veh_3")["nr_veh_3"].sum().sort_values(ascending=False)
cars_involved = cars_involved_1.add(cars_involved_2, fill_value=0).add(cars_involved_3, fill_value=0)
cars_involved = cars_involved.sort_values(ascending=False)
# The distribution of the vehicles involved in dangerous accidents
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=cars_involved.values, y=cars_involved.index, color="red")
for i, v in enumerate(cars_involved.values):
    ax.text(v+50, i, str(int(v)), ha="center")
plt.title("The distribution of the vehicles involved in serious accidents")
plt.xlabel("Nr. of vehicle")
plt.ylabel("Type of vehicles")
plt.grid(True, axis="x", linestyle="dashed")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------------------------
# process the vehicles involved in mortal accidents

# 1. Total Sum
s1 = df.groupby("veh_1")["nr_morti"].sum()
s2 = df.groupby("veh_2")["nr_morti"].sum()
s3 = df.groupby("veh_3")["nr_morti"].sum()
total_sum = s1.add(s2, fill_value=0).add(s3, fill_value=0)

# 2. Total count
c1 = df["veh_1"].value_counts()
c2 = df["veh_2"].value_counts()
c3 = df["veh_3"].value_counts()
total_count = c1.add(c2, fill_value=0).add(c3, fill_value=0)

# 3. Mean
avg_per_vehicle = (total_sum / total_count).sort_values(ascending=False)

mask = total_count > 10
avg_filtered = avg_per_vehicle[mask]
total_filtered = total_count[mask]

# The distribution of the vehicles involved in dangerous accidents
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=avg_filtered.values, y=avg_filtered.index, color="gray")
for i, key in enumerate(avg_filtered.index):
    count = int(total_filtered.get(key, 0))
    mean = avg_filtered[key]
    ax.text(mean + 0.01, i, f"{mean*100:.2f}% - ({count} acc.)", ha="left")
plt.title("The mean nr. of deaths per vehicle")
plt.xlabel("Mean nr. of deaths")
plt.ylabel("Type of vehicles")
plt.grid(True, axis="x", linestyle="dashed")
plt.tight_layout()
plt.show()