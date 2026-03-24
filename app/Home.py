import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="PatrolIQ",
    page_icon="shield",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_parquet("data/crimes_clustered.parquet")

df = load_data()

st.title("PatrolIQ — Smart Safety Analytics Platform")
st.markdown("**Chicago Crime Intelligence Dashboard** | 500,000 records analyzed")
st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records",    f"{len(df):,}")
col2.metric("Crime Types",      f"{df['Primary Type'].nunique()}")
col3.metric("Districts",        f"{df['District'].nunique()}")
col4.metric("Arrest Rate",      f"{df['Arrest'].mean()*100:.1f}%")
col5.metric("Domestic Rate",    f"{df['Domestic'].mean()*100:.1f}%")

st.divider()

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Crime type distribution")
    top_crimes = df["Primary Type"].value_counts().head(12).reset_index()
    top_crimes.columns = ["Crime Type", "Count"]
    fig = px.bar(
        top_crimes, x="Count", y="Crime Type",
        orientation="h", color="Count",
        color_continuous_scale="Purples",
        height=420
    )
    fig.update_layout(showlegend=False,
                      coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Hourly crime pattern")
    hourly = df.groupby("Hour").size().reset_index(name="Count")
    fig2 = px.line(
        hourly, x="Hour", y="Count",
        markers=True, height=420,
        color_discrete_sequence=["#D85A30"]
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Crime severity heatmap — hour vs day of week")
day_order = ["Monday","Tuesday","Wednesday",
             "Thursday","Friday","Saturday","Sunday"]
pivot = df.pivot_table(
    index="Day_of_Week", columns="Hour",
    values="Crime_Severity_Score", aggfunc="mean"
).reindex([d for d in day_order if d in df["Day_of_Week"].unique()])

fig3 = px.imshow(
    pivot, color_continuous_scale="YlOrRd",
    aspect="auto", height=280
)
fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption("Data source: Chicago Data Portal | "
           "Model: Unsupervised ML — K-Means, DBSCAN, "
           "Hierarchical, PCA, t-SNE")