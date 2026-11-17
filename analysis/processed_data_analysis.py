from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv(ROOT / "data" / "accidente_processed.csv")

# 2. Clean data
df = df[["nr_morti", "nr_raniti", "reluare", "judet", "drum"]].dropna()
df = df[~df["judet"].str.contains("nementionat", case=False)]
df_deaths = df[df["nr_morti"] > 0]
df_wounded = df[df["nr_raniti"] > 0]

#calculate the percentages of nr of accidents with deaths and wounded per road
deaths_accidents_road = df_deaths["drum"].value_counts()/len(df_deaths)
wounded_accidents_road = df_wounded["drum"].value_counts()/len(df_wounded)

#calculate the percentages of nr of accidents with deaths and wounded per county
deaths_accidents_county = df_deaths["judet"].value_counts()/len(df_deaths)
wounded_accidents_county = df_wounded["judet"].value_counts()/len(df_wounded)

#calculate the percentages of accidents per road and county
accidents_road = df["drum"].value_counts()/len(df)
accidents_county = df["judet"].value_counts()/len(df)

#calculate the percentage of wounded and deaths relative to the total number of deaths and wounded
#----for roads
deaths_total_percentage_road = df_deaths.groupby("drum")["nr_morti"].agg("sum")/df_deaths["nr_morti"].sum()
wounded_total_percentage_road = df_wounded.groupby("drum")["nr_raniti"].agg("sum")/df_wounded["nr_raniti"].sum()

#----for county
deaths_total_percentage_county = df_deaths.groupby("judet")["nr_morti"].agg("sum")/df_deaths["nr_morti"].sum()
wounded_total_percentage_county = df_wounded.groupby("judet")["nr_raniti"].agg("sum")/df_wounded["nr_raniti"].sum()

# Combine the components into one table per road/county. A road with no
# fatal accidents must get 0 (not NaN) for the "deaths" components, otherwise
# the sum below becomes NaN and the road silently disappears from the ranking.
def risk_score(acc_deaths, acc_wounded, acc_all, tot_deaths, tot_wounded):
    parts = pd.concat(
        [acc_deaths, acc_wounded, acc_all, tot_deaths, tot_wounded], axis=1
    ).fillna(0)
    parts.columns = ["acc_deaths", "acc_wounded", "acc_all", "tot_deaths", "tot_wounded"]
    risk = (
        parts["acc_deaths"] * 0.4 +
        parts["acc_wounded"] * 0.2 +
        parts["acc_all"] * 0.2 +
        parts["tot_deaths"] * 0.1 +
        parts["tot_wounded"] * 0.1)
    return risk, (risk - risk.mean()) / risk.std()


road_risk, road_risk_zscore = risk_score(
    deaths_accidents_road, wounded_accidents_road, accidents_road,
    deaths_total_percentage_road, wounded_total_percentage_road)
county_risk, county_risk_zscore = risk_score(
    deaths_accidents_county, wounded_accidents_county, accidents_county,
    deaths_total_percentage_county, wounded_total_percentage_county)


comparison_df_road = pd.DataFrame({
    "Real_risk": road_risk,
    "Z-score": road_risk_zscore
}).sort_values("Real_risk", ascending=False).head(15)

comparison_df_county = pd.DataFrame({
    "Real_risk": county_risk,
    "Z-score": county_risk_zscore
}).sort_values("Real_risk", ascending=False).head(15)

# Vizualize top 15 most dangerous roads
comparison_df_road[["Real_risk", "Z-score"]].plot(
    kind="bar", 
    figsize=(12, 6), 
    width=0.8,
    colormap="Set2"
)

plt.title("Top 15 most dangerous roads ")
plt.ylabel("risk value - real risk/z-score")
plt.xlabel("Road")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Method")
plt.tight_layout()
plt.show()

# Vizualize top 15 most dangerous counties
comparison_df_county[["Real_risk", "Z-score"]].plot(
    kind="bar", 
    figsize=(12, 6), 
    width=0.8,
    colormap="Set2"
)

plt.title("Top 15 most dangerous counties ")
plt.ylabel("risk value - real risk/z-score")
plt.xlabel("County")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Method")
plt.tight_layout()
plt.show()

# vizualize the nr. of victims based on traffic resumption intervals
df_dangerous = pd.concat([df_deaths, df_wounded[df_wounded["nr_raniti"]>=2]]).drop_duplicates()
df_dangerous["victime"] = df_dangerous["nr_morti"] + df_dangerous["nr_raniti"]
df_dangerous["reluare_bin"] = pd.cut(df_dangerous["reluare"], bins=[0,10,20,30,60,90,180,360,9999], include_lowest=True)
df_dangerous.groupby("reluare_bin")["victime"].sum().plot(kind="bar", title="Dangerous accidents - nr. of victims based on traffic resumption intervals")
plt.xlabel("Traffic resumption intervals")
plt.ylabel("Nr. of victims")
plt.tight_layout()
plt.show()
