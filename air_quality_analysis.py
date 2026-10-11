import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("DATA/DATASET.csv")

print("\n========== AIR QUALITY DATA ==========")
print(df)

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

print("\n========== LAST FIVE ROWS ==========")
print(df.tail())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== CITY COUNTS ==========")
print(df["City"].value_counts())

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "Temperature",
    "Humidity"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.drop_duplicates()

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

df["City"] = df["City"].astype("string").str.strip()

df.to_csv("DATA/DATASET.csv", index=False)

print("\n========== CLEANED DATA ==========")
print(df.head())

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

print("\n========== DATASET SHAPE AFTER CLEANING ==========")
print(df.shape)

print("\n========== POLLUTANT AVERAGES ==========")

print(df[["PM2.5", "PM10", "NO2", "SO2"]].mean())

print("\n========== POLLUTANT MEDIANS ==========")

print(df[["PM2.5", "PM10", "NO2", "SO2"]].median())

print("\n========== POLLUTANT STANDARD DEVIATION ==========")

print(df[["PM2.5", "PM10", "NO2", "SO2"]].std())

print("\n========== CITY-WISE AVERAGES ==========")

city_analysis = df.groupby("City")[
    ["PM2.5", "PM10", "NO2", "SO2"]
].mean()

print(city_analysis)

print("\n========== CITIES SORTED BY PM2.5 ==========")

print(city_analysis.sort_values("PM2.5", ascending=False))

print("\n========== RECORDS WITH PM2.5 ABOVE 100 ==========")

print(df[df["PM2.5"] > 100])

print("\n========== POLLUTANT CORRELATION ==========")

print(df[["PM2.5", "PM10", "NO2", "SO2"]].corr())

os.makedirs("OUTPUT", exist_ok=True)

city_analysis["PM2.5"].plot(kind="bar", figsize=(9, 5))

plt.title("Average PM2.5 Levels by City")
plt.xlabel("City")
plt.ylabel("PM2.5 Level")
plt.tight_layout()
plt.savefig("OUTPUT/pm25_by_city.png")
plt.show()

plt.figure(figsize=(9, 5))

plt.hist(df["PM2.5"], bins=10, edgecolor="black")

plt.title("Distribution of PM2.5 Levels")
plt.xlabel("PM2.5 Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("OUTPUT/pm25_distribution.png")
plt.show()

plt.figure(figsize=(9, 5))

sns.heatmap(
    df[["PM2.5", "PM10", "NO2", "SO2"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Pollutant Correlation Heatmap")
plt.tight_layout()
plt.savefig("OUTPUT/pollutant_correlation.png")
plt.show()

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="PM2.5",
    y="PM10",
    hue="City"
)

plt.title("PM2.5 vs PM10")
plt.tight_layout()
plt.savefig("OUTPUT/pm25_vs_pm10.png")
plt.show()

print("Graphs saved successfully in the OUTPUT folder!")

df["Date"] = pd.to_datetime(df["Date"])

os.makedirs("OUTPUT", exist_ok=True)

plt.figure(figsize=(10, 5))

for city in df["City"].unique():
    city_data = df[df["City"] == city]
    plt.plot(city_data["Date"], city_data["PM2.5"], label=city)

plt.title("PM2.5 Levels Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 Level")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("OUTPUT/line_chart.png")
plt.show()


city_counts = df["City"].value_counts()

plt.figure(figsize=(8, 5))
plt.pie(
    city_counts.values,
    labels=city_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Distribution of Records by City")
plt.tight_layout()
plt.savefig("OUTPUT/pie_chart.png")
plt.show()


plt.figure(figsize=(9, 5))

sns.boxplot(data=df, x="City", y="PM2.5")

plt.title("PM2.5 Distribution by City")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("OUTPUT/box_plot.png")
plt.show()


plt.figure(figsize=(9, 5))

sns.violinplot(data=df, x="City", y="PM2.5")

plt.title("PM2.5 Distribution by City - Violin Plot")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("OUTPUT/violin_plot.png")
plt.show()


plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="City")

plt.title("Number of Observations per City")
plt.xlabel("City")
plt.ylabel("Number of Observations")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("OUTPUT/count_plot.png")
plt.show()


plt.figure(figsize=(10, 7))

sns.pairplot(
    df,
    vars=["PM2.5", "PM10", "NO2", "SO2"],
    hue="City"
)

plt.savefig("OUTPUT/pair_plot.png")
plt.show()
