import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Temporal Patterns", layout="wide")
st.title("Temporal Crime Pattern Analysis")
st.markdown("When do crimes happen? Hourly, daily, monthly and seasonal breakdowns.")

@st.cache_data
def load_data():
    return pd.read_parquet("data/crimes_clustered.parquet")

@st.cache_data
def load_temporal_profiles():
    return pd.read_parquet("data/temporal_cluster_profiles.parquet")

df       = load_data()
profiles = load_temporal_profiles()

st.sidebar.header("Filters")
crime_filter = st.sidebar.multiselect(
    "Filter by crime type",
    options=sorted(df["Primary Type"].unique()),
    default=[]
)
season_filter = st.sidebar.multiselect(
    "Filter by season",
    options=["Winter","Spring","Summer","Fall"],
    default=[]
)

df_f = df.copy()
if crime_filter:
    df_f = df_f[df_f["Primary Type"].isin(crime_filter)]
if season_filter:
    df_f = df_f[df_f["Season"].isin(season_filter)]

tab1, tab2, tab3, tab4 = st.tabs([
    "Hourly", "Daily & Weekly", "Monthly & Seasonal", "Temporal Clusters"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        hourly = df_f.groupby("Hour").size().reset_index(name="Count")
        fig = px.line(
            hourly, x="Hour", y="Count",
            title="Crime frequency by hour",
            markers=True,
            color_discrete_sequence=["#D85A30"]
        )
        fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hourly_sev = df_f.groupby("Hour")["Crime_Severity_Score"].mean().reset_index()
        fig2 = px.bar(
            hourly_sev, x="Hour", y="Crime_Severity_Score",
            title="Avg crime severity by hour",
            color="Crime_Severity_Score",
            color_continuous_scale="RdYlGn_r"
        )
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    day_order = ["Monday","Tuesday","Wednesday",
                 "Thursday","Friday","Saturday","Sunday"]
    col1, col2 = st.columns(2)
    with col1:
        daily = df_f["Day_of_Week"].value_counts().reindex(day_order).reset_index()
        daily.columns = ["Day","Count"]
        fig3 = px.bar(
            daily, x="Day", y="Count",
            title="Crime frequency by day of week",
            color="Count", color_continuous_scale="Purples"
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        pivot_hw = df_f.pivot_table(
            index="Day_of_Week", columns="Hour",
            values="ID", aggfunc="count"
        ).reindex([d for d in day_order if d in df_f["Day_of_Week"].unique()])
        fig4 = px.imshow(
            pivot_hw,
            title="Crime heatmap — hour vs day",
            color_continuous_scale="YlOrRd",
            aspect="auto"
        )
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        monthly = df_f.groupby("Month").size().reset_index(name="Count")
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",
                       6:"Jun",7:"Jul",8:"Aug",9:"Sep",
                       10:"Oct",11:"Nov",12:"Dec"}
        monthly["Month_Name"] = monthly["Month"].map(month_names)
        fig5 = px.bar(
            monthly, x="Month_Name", y="Count",
            title="Crimes by month",
            color="Count", color_continuous_scale="Blues"
        )
        fig5.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        seasonal = df_f["Season"].value_counts().reset_index()
        seasonal.columns = ["Season","Count"]
        fig6 = px.pie(
            seasonal, names="Season", values="Count",
            title="Crime distribution by season",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig6, use_container_width=True)

with tab4:
    st.subheader("Temporal cluster profiles")
    st.dataframe(profiles, use_container_width=True)

    cluster_hourly = df_f.groupby(
        ["Temporal_Cluster","Hour"]
    ).size().reset_index(name="Count")

    fig7 = px.line(
        cluster_hourly, x="Hour", y="Count",
        color="Temporal_Cluster",
        title="Crime patterns by temporal cluster",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig7.update_layout(xaxis=dict(tickmode="linear", dtick=2))
    st.plotly_chart(fig7, use_container_width=True)
