import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np

st.set_page_config(page_title="Clustering Analysis", layout="wide")
st.title("Clustering Analysis")
st.markdown("K-Means · DBSCAN · Hierarchical — geographic crime hotspot detection")

@st.cache_data
def load_data():
    return pd.read_parquet("data/crimes_clustered.parquet")

@st.cache_data
def load_comparison():
    return pd.read_parquet("data/clustering_comparison.parquet")

@st.cache_data
def load_dbscan():
    return pd.read_parquet("data/dbscan_results.parquet")

@st.cache_data
def load_hierarchical():
    return pd.read_parquet("data/hierarchical_results.parquet")

df         = load_data()
comparison = load_comparison()
dbscan_df  = load_dbscan()
hier_df    = load_hierarchical()

algo = st.sidebar.radio(
    "Select algorithm",
    ["K-Means", "DBSCAN", "Hierarchical", "Comparison"]
)

if algo == "K-Means":
    st.subheader("K-Means geographic clustering")
    k_sample = df.sample(30000, random_state=42)
    fig = px.scatter(
        k_sample, x="Longitude", y="Latitude",
        color=k_sample["KMeans_Cluster"].astype(str),
        title="K-Means crime hotspot zones",
        opacity=0.4, height=550,
        color_discrete_sequence=px.colors.qualitative.T10
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(legend_title="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster statistics")
    summary = load_data().groupby("KMeans_Cluster").agg(
        Crime_Count=("ID","count"),
        Avg_Severity=("Crime_Severity_Score","mean"),
        Arrest_Rate=("Arrest","mean"),
        Top_Crime=("Primary Type", lambda x: x.value_counts().index[0])
    ).round(3)
    summary["Arrest_Rate"] = (summary["Arrest_Rate"]*100).round(1)
    st.dataframe(summary, use_container_width=True)

elif algo == "DBSCAN":
    st.subheader("DBSCAN density-based clustering")
    noise    = dbscan_df[dbscan_df["DBSCAN_Cluster"] == -1]
    clusters = dbscan_df[dbscan_df["DBSCAN_Cluster"] != -1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters found",
                dbscan_df["DBSCAN_Cluster"].nunique() - 1)
    col2.metric("Noise points",  f"{len(noise):,}")
    col3.metric("Noise %",
                f"{len(noise)/len(dbscan_df)*100:.1f}%")

    fig = px.scatter(
        clusters.sample(min(30000, len(clusters)), random_state=42),
        x="Longitude", y="Latitude",
        color=clusters.sample(
            min(30000,len(clusters)), random_state=42
        )["DBSCAN_Cluster"].astype(str),
        title="DBSCAN clusters (noise removed)",
        opacity=0.5, height=500
    )
    fig.update_traces(marker=dict(size=2))
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"DBSCAN removed {len(noise):,} noise points "
            f"({len(noise)/len(dbscan_df)*100:.1f}%) — "
            f"these are isolated incidents not part of any hotspot.")

elif algo == "Hierarchical":
    st.subheader("Hierarchical clustering")
    fig = px.scatter(
        hier_df,
        x="Longitude", y="Latitude",
        color=hier_df["Hier_Cluster"].astype(str),
        title="Hierarchical geographic zones",
        opacity=0.5, height=550,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(legend_title="Zone")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Zone breakdown")
    zone_counts = hier_df["Hier_Cluster"].value_counts().reset_index()
    zone_counts.columns = ["Zone","Crime Count"]
    fig2 = px.bar(
        zone_counts, x="Zone", y="Crime Count",
        color="Crime Count", color_continuous_scale="Purples"
    )
    st.plotly_chart(fig2, use_container_width=True)

elif algo == "Comparison":
    st.subheader("Algorithm performance comparison")
    st.dataframe(comparison, use_container_width=True)

    fig = px.bar(
        comparison, x="Algorithm", y="Silhouette Score",
        color="Algorithm", title="Silhouette score comparison",
        color_discrete_sequence=["#7F77DD","#1D9E75","#D85A30"],
        height=400, text="Silhouette Score"
    )
    fig.add_hline(
        y=0.5, line_dash="dash",
        annotation_text="Target threshold (0.5)"
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Algorithm characteristics")
    char_df = pd.DataFrame({
        "Algorithm":   ["K-Means","DBSCAN","Hierarchical"],
        "Best for":    ["Patrol zone planning",
                        "Natural hotspot detection",
                        "Zone hierarchy analysis"],
        "Handles noise":  ["No","Yes","No"],
        "Cluster shape":  ["Spherical","Arbitrary","Hierarchical"],
        "Speed":          ["Fast","Medium","Slow"],
    })
    st.dataframe(char_df, use_container_width=True)
