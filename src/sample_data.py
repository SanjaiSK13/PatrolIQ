import pandas as pd
import numpy as np
import os

print("=" * 60)
print("PatrolIQ — Data Sampling Pipeline")
print("=" * 60)

RAW_PATH = "data/crimes_raw.csv"
OUTPUT_PATH = "data/crimes_sample.parquet"
SAMPLE_SIZE = 500_000
RANDOM_SEED = 42

# inspect the file without loading it all
print("\n[1/6] Counting rows in raw file (this takes ~30 seconds)...")
row_count = sum(1 for _ in open(RAW_PATH, encoding="utf-8")) - 1
print(f"      Total records found: {row_count:,}")

# load in chunks
print("\n[2/6] Loading dataset in chunks...")

USECOLS = [
    "ID", "Case Number", "Date", "Block", "IUCR",
    "Primary Type", "Description", "Location Description",
    "Arrest", "Domestic", "Beat", "District", "Ward",
    "Community Area", "FBI Code", "X Coordinate",
    "Y Coordinate", "Year", "Latitude", "Longitude"
]

chunks = []
chunk_size = 100_000

for chunk in pd.read_csv(
    RAW_PATH,
    usecols=USECOLS,
    chunksize=chunk_size,
    low_memory=False,
    encoding="utf-8"
):
    chunks.append(chunk)
    loaded = sum(len(c) for c in chunks)
    print(f"      Loaded {loaded:,} rows...", end="\r")

df = pd.concat(chunks, ignore_index=True)
print(f"\n      Full dataset shape: {df.shape}")

# basic cleaning before sampling
print("\n[3/6] Cleaning data...")

df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

before = len(df)
df = df.dropna(subset=["Latitude", "Longitude", "Date", "Primary Type"])
df = df[df["Latitude"].between(41.6, 42.1)]
df = df[df["Longitude"].between(-87.95, -87.5)]
df = df[df["Year"] >= 2015]
after = len(df)

print(f"      Rows after cleaning: {after:,} (removed {before - after:,})")

# sample 500K (or all if less available)
print(f"\n[4/6] Sampling {SAMPLE_SIZE:,} records...")

if len(df) >= SAMPLE_SIZE:
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
else:
    df_sample = df.copy()
    print(f"      Warning: only {len(df):,} records available after cleaning. Using all.")

df_sample = df_sample.reset_index(drop=True)
print(f"      Sample shape: {df_sample.shape}")

# feature engineering
print("\n[5/6] Engineering features...")

df_sample["Hour"] = df_sample["Date"].dt.hour
df_sample["Day_of_Week"] = df_sample["Date"].dt.day_name()
df_sample["Month"] = df_sample["Date"].dt.month
df_sample["Is_Weekend"] = df_sample["Date"].dt.dayofweek >= 5

def get_season(month):
    if month in [12, 1, 2]:  return "Winter"
    elif month in [3, 4, 5]:  return "Spring"
    elif month in [6, 7, 8]:  return "Summer"
    else:                      return "Fall"

df_sample["Season"] = df_sample["Month"].apply(get_season)

severity_map = {
    "HOMICIDE": 10, "CRIM SEXUAL ASSAULT": 9, "KIDNAPPING": 9,
    "ARSON": 8, "ROBBERY": 8, "ASSAULT": 7, "BATTERY": 7,
    "BURGLARY": 6, "MOTOR VEHICLE THEFT": 6, "WEAPONS VIOLATION": 6,
    "STALKING": 5, "INTIMIDATION": 5, "OFFENSE INVOLVING CHILDREN": 5,
    "THEFT": 4, "DECEPTIVE PRACTICE": 4, "SEX OFFENSE": 4,
    "CRIMINAL DAMAGE": 3, "NARCOTICS": 3, "CRIMINAL TRESPASS": 2,
    "GAMBLING": 2, "LIQUOR LAW VIOLATION": 2, "PUBLIC PEACE VIOLATION": 2,
    "INTERFERENCE WITH PUBLIC OFFICER": 1, "OBSCENITY": 1,
}
df_sample["Crime_Severity_Score"] = df_sample["Primary Type"].map(severity_map).fillna(3)

df_sample["Arrest"] = df_sample["Arrest"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)
df_sample["Domestic"] = df_sample["Domestic"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)

# save as parquet
print("\n[6/6] Saving sample as parquet...")
df_sample.to_parquet(OUTPUT_PATH, index=False)
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"      Saved to: {OUTPUT_PATH}")
print(f"      File size: {size_mb:.1f} MB")

print("\n" + "=" * 60)
print("SAMPLING COMPLETE — Summary")
print("=" * 60)
print(f"  Records:       {len(df_sample):,}")
print(f"  Columns:       {df_sample.shape[1]}")
print(f"  Date range:    {df_sample['Date'].min().date()} → {df_sample['Date'].max().date()}")
print(f"  Crime types:   {df_sample['Primary Type'].nunique()}")
print(f"  Districts:     {df_sample['District'].nunique()}")
print(f"  Arrest rate:   {df_sample['Arrest'].mean()*100:.1f}%")
print(f"  Domestic rate: {df_sample['Domestic'].mean()*100:.1f}%")
print("\n  Top 5 crime types:")
top5 = df_sample["Primary Type"].value_counts().head()
for crime, count in top5.items():
    print(f"    {crime:<30} {count:>7,}")
print("=" * 60)
print("Step 2 complete. Ready for Step 3 (Preprocessing & EDA).")