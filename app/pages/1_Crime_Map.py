import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(page_title="Crime Map", layout="wide")
st.title("Geographic Crime Analysis")
st.markdown("Interactive crime heatmap with K-Means cluster boundaries")

@st.cache_data
def load_data():
    return pd.read_parquet("./data/crimes_clustered.parquet")

@st.cache_data
def load_cluster_summary():
    return pd.read_parquet("./data/kmeans_cluster_summary.parquet")

df      = load_data()
summary = load_cluster_summary()

st.sidebar.header("Map filters")
crime_types = ["All"] + sorted(df["Primary Type"].unique().tolist())
selected_crime = st.sidebar.selectbox("Crime type", crime_types)

selected_year = st.sidebar.selectbox(
    "Year",
    ["All"] + sorted(df["Date"].dt.year.unique().tolist(), reverse=True)
)

map_type = st.sidebar.radio(
    "Map type",
    ["Heatmap", "Cluster zones", "Marker cluster"]
)

sample_size = st.sidebar.slider(
    "Points to display", 5000, 50000, 20000, step=5000
)

df_filtered = df.copy()
if selected_crime != "All":
    df_filtered = df_filtered[df_filtered["Primary Type"] == selected_crime]
if selected_year != "All":
    df_filtered = df_filtered[df_filtered["Date"].dt.year == int(selected_year)]

df_map = df_filtered.dropna(
    subset=["Latitude","Longitude"]
).sample(min(sample_size, len(df_filtered)), random_state=42)

col1, col2, col3 = st.columns(3)
col1.metric("Filtered records", f"{len(df_filtered):,}")
col2.metric("Arrest rate",
            f"{df_filtered['Arrest'].mean()*100:.1f}%")
col3.metric("Avg severity",
            f"{df_filtered['Crime_Severity_Score'].mean():.2f}")

m = folium.Map(
    location=[41.85, -87.65],
    zoom_start=11,
    tiles="CartoDB dark_matter"
)

if map_type == "Heatmap":
    heat_data = df_map[["Latitude","Longitude",
                         "Crime_Severity_Score"]].values.tolist()
    HeatMap(
        heat_data, radius=8, blur=10,
        gradient={0.2:"blue", 0.5:"yellow", 0.8:"orange", 1.0:"red"}
    ).add_to(m)

elif map_type == "Cluster zones":
    cluster_colors = [
        "red","blue","green","purple","orange",
        "darkred","lightblue","darkgreen","cadetblue","darkpurple"
    ]
    for _, row in df_map.iterrows():
        cid = int(row["KMeans_Cluster"])
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=2,
            color=cluster_colors[cid % len(cluster_colors)],
            fill=True, fill_opacity=0.5, weight=0
        ).add_to(m)

    for cid, row in summary.iterrows():
        folium.Marker(
            location=[row["Lat_Center"], row["Lon_Center"]],
            popup=folium.Popup(
                f"<b>Cluster {cid}</b><br>"
                f"Crimes: {row['Crime_Count']:,}<br>"
                f"Top crime: {row['Top_Crime']}<br>"
                f"Arrest rate: {row['Arrest_Rate']*100:.1f}%",
                max_width=200
            ),
            icon=folium.Icon(color="white", icon="info-sign")
        ).add_to(m)

elif map_type == "Marker cluster":
    mc = MarkerCluster().add_to(m)
    for _, row in df_map.head(5000).iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=3,
            color="orange", fill=True, fill_opacity=0.6
        ).add_to(mc)

st_folium(m, width=None, height=550)

st.divider()
st.subheader("K-Means cluster summary")
summary_display = summary.copy()
summary_display["Arrest_Rate"] = (
    summary_display["Arrest_Rate"] * 100
).round(1).astype(str) + "%"
summary_display["Avg_Severity"] = summary_display["Avg_Severity"].round(2)
st.dataframe(summary_display, use_container_width=True)
