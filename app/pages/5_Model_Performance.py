import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance & Experiment Tracking")
st.markdown("Clustering metrics, dimensionality reduction results and model comparison")

@st.cache_data
def load_comparison():
    return pd.read_parquet("data/clustering_comparison.parquet")

comparison = load_comparison()

tab1, tab2, tab3 = st.tabs([
    "Clustering metrics",
    "Dimensionality reduction",
    "MLflow experiments"
])

with tab1:
    st.subheader("Geographic clustering — algorithm comparison")
    st.dataframe(comparison, use_container_width=True)

    fig = go.Figure()
    algos  = comparison["Algorithm"].tolist()
    scores = comparison["Silhouette Score"].tolist()
    colors = ["#7F77DD", "#1D9E75", "#D85A30"]

    for algo, score, color in zip(algos, scores, colors):
        if score:
            fig.add_trace(go.Bar(
                name=algo, x=[algo], y=[score],
                marker_color=color,
                text=[f"{score:.4f}"],
                textposition="outside"
            ))

    fig.add_hline(
        y=0.5, line_dash="dash", line_color="black",
        annotation_text="Target: 0.5"
    )
    fig.update_layout(
        title="Silhouette score by algorithm",
        yaxis_title="Silhouette score",
        showlegend=False, height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evaluation metrics explained")
    metrics_df = pd.DataFrame({
        "Metric": [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Inertia"
        ],
        "Range": ["−1 to +1", "0 to ∞", "0 to ∞"],
        "Good value": ["> 0.5", "< 1.0", "As low as possible"],
        "Meaning": [
            "How well separated clusters are",
            "Ratio of within to between cluster distance",
            "Sum of squared distances to cluster center"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True)

with tab2:
    st.subheader("PCA — explained variance")

    try:
        pca_model = pickle.load(open("src/models/pca_model.pkl", "rb"))
        var = pca_model.explained_variance_ratio_

        var_df = pd.DataFrame({
            "Component":    [f"PC{i+1}" for i in range(len(var))],
            "Variance %":   (var * 100).round(2),
            "Cumulative %": (var.cumsum() * 100).round(2)
        })
        st.dataframe(var_df, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=var_df["Component"],
            y=var_df["Variance %"],
            name="Individual",
            marker_color="#7F77DD"
        ))
        fig2.add_trace(go.Scatter(
            x=var_df["Component"],
            y=var_df["Cumulative %"],
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#D85A30", width=2),
            yaxis="y2"
        ))
        fig2.update_layout(
            title="PCA scree plot",
            yaxis=dict(title="Individual variance %"),
            yaxis2=dict(
                title="Cumulative %",
                overlaying="y",
                side="right"
            ),
            height=420,
            legend=dict(x=0.7, y=0.3)
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load PCA model: {e}")

    st.subheader("t-SNE parameters used")
    tsne_params = pd.DataFrame({
        "Parameter": [
            "Perplexity", "Iterations",
            "Learning rate", "Sample size"
        ],
        "Value": ["40", "1000", "200", "15,000"],
        "Effect": [
            "Controls local vs global structure balance",
            "More iterations = better convergence",
            "Controls step size in optimization",
            "Stratified sample across all crime types"
        ]
    })
    st.dataframe(tsne_params, use_container_width=True)

with tab3:
    st.subheader("MLflow experiment tracking")
    st.info(
        "MLflow runs are tracked locally. "
        "To view the full interactive UI, run this command "
        "in your terminal and visit http://localhost:5000\n\n"
        "`mlflow ui`"
    )

    st.subheader("Experiments logged")
    experiments_df = pd.DataFrame({
        "Experiment": [
            "PatrolIQ_Clustering",
            "PatrolIQ_Clustering",
            "PatrolIQ_Clustering",
            "PatrolIQ_Clustering",
            "PatrolIQ_DimReduction",
            "PatrolIQ_DimReduction"
        ],
        "Run name": [
            "KMeans_Geographic",
            "DBSCAN_Geographic",
            "Hierarchical_Geographic",
            "KMeans_Temporal",
            "PCA_CrimeFeatures",
            "tSNE_CrimeFeatures"
        ],
        "Key metrics": [
            "silhouette, davies_bouldin, inertia",
            "n_clusters, noise_pct, silhouette",
            "silhouette, davies_bouldin",
            "silhouette, inertia",
            "pc1_variance, pc2_variance, total_variance",
            "kl_divergence"
        ],
        "Status": [
            "FINISHED", "FINISHED", "FINISHED",
            "FINISHED", "FINISHED", "FINISHED"
        ]
    })
    st.dataframe(experiments_df, use_container_width=True)

    st.subheader("Model files saved")
    model_files = [f for f in os.listdir("src/models") if f.endswith(".pkl")]
    model_df = pd.DataFrame({
        "Model file": sorted(model_files),
        "Purpose": [
            "DBSCAN geographic clustering",
            "Hierarchical geographic clustering",
            "K-Means geographic clustering",
            "K-Means temporal clustering",
            "Label encoder — crime type",
            "Label encoder — location",
            "PCA dimensionality reduction",
            "StandardScaler — full features",
            "StandardScaler — geographic",
            "StandardScaler — temporal"
        ][:len(model_files)]
    })
    st.dataframe(model_df, use_container_width=True)