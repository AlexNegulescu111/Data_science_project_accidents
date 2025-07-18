from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(ROOT / "data" / "accidente_clean.csv")
print(df.info())

# 1. Vizualize data - Distribution of traffic ressumption(reluare)
plt.figure(figsize=(10, 5))
sns.histplot(df["reluare"].dropna(), bins = 30, kde=True, color="blue")
plt.title("Traffic ressumtion distribution")
plt.xlabel("Minutes")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 1. Vizualize data - Traffic ressumption (reluare) - boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df["reluare"], vert=True )
plt.yscale("log")
plt.title("Traffic ressumption values (log scale)")
plt.ylabel("Minutes(log)")
plt.grid(True)
plt.show()

# 2. Manage outlayers via IQR
reluare = df["reluare"].dropna()
q1 = np.quantile(reluare, 0.25)
q3 = np.quantile(reluare, 0.75)
iqr = q3 - q1
low_val = q1 - 1.5*iqr
high_val = q3 + 1.5*iqr
if low_val<0:
    low_val = 0
print(f"Values caught between {low_val} - {high_val}")
no_outlayers = (reluare>=low_val)&(reluare<=high_val)
print(f"No outlayers: {no_outlayers.sum()/len(reluare)*100:.2f} % of the data")

# 3. Prepare the data for vizualizing
df["data_ora"] = pd.to_datetime(df["data_ora"])
df["hour"] = df["data_ora"].dt.hour
df["day"] = df["data_ora"].dt.day_name()
df["month"] = df["data_ora"].dt.month_name()
df["year"] = df["data_ora"].dt.year

# accidents per hour
plt.figure(figsize=(10, 5))
sns.countplot(x="hour", data=df, palette="Blues_d")
plt.title("Distribution of accidents per hour")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

#accident per day
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(10, 5))
sns.countplot(x="day", data=df, palette="Blues_d", order=day_order)
plt.title("Distribution of accidents per day")
plt.xlabel("Day")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

#accident per month
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
plt.figure(figsize=(10, 5))
sns.countplot(x="month", data=df, palette="Blues_d", order=month_order)
plt.title("Distribution of accidents per month")
plt.xlabel("Month")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# accident per year
plt.figure(figsize=(10, 5))
sns.countplot(x="year", data=df, palette="Blues_d")
plt.title("Distribution of accidents per year")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# heatmap day-hour accidents
pivot = df.groupby(["day", "hour"]).size().unstack().fillna(0)
pivot = pivot.reindex(day_order)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="YlOrRd")
plt.title("Heatmap: Accidents per day and hour")
plt.xlabel("Hour")
plt.ylabel("Day")
plt.tight_layout()
plt.show()

# top 15 counties by nr. of accidents
mask = df[~df["judet"].str.contains("nementionat", case=False, na=False)]
top_15_counties = mask["judet"].value_counts().head(15)
sns.barplot(x=top_15_counties.values, y=top_15_counties.index,  palette="Blues_d")
plt.title("15 most dangerous counties")
plt.xlabel("Accidents")
plt.ylabel("County")
plt.grid(True)
plt.tight_layout()
plt.show()

# top 15 roads by nr. of accidents
top_15_roads = df["drum"].dropna().value_counts().head(15)
sns.barplot(x=top_15_roads.values, y=top_15_roads.index,  palette="Blues_d")
plt.title("15 most dangerous roads")
plt.xlabel("Accidents")
plt.ylabel("Roads")
plt.grid(True)
plt.tight_layout()
plt.show()

# mean time interval of the traffic resumption in the most affected counties (no outlayers data)
no_outlayers_data = df[(df["reluare"] <= high_val) &
                       (~df["judet"].str.contains("nementionat", case=False, na=False)) &
                       (df["judet"].isin(top_15_counties.index))]
mean_traffic_ressumption = no_outlayers_data.groupby("judet")["reluare"].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=mean_traffic_ressumption.values, y=mean_traffic_ressumption.index, palette="Blues_d")
plt.title("Mean time interval of the traffic resumption in the most affected counties (no outlayers data)")
plt.xlabel("The mean of traffic ressumption (Min.)")
plt.ylabel("County")
plt.tight_layout()
plt.show()