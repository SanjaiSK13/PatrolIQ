import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import pickle

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance & MLflow Tracking")
st.markdown("Experiment tracking, metric comparison and model registry")

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
    colors = ["#7F77DD","#1D9E75","#D85A30"]

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
        "Metric": ["Silhouette Score","Davies-Bouldin Index","Inertia"],
        "Range":  ["−1 to +1","0 to ∞","0 to ∞"],
        "Good value": ["> 0.5","< 1.0","As low as possible"],
        "Meaning": [
            "How well separated clusters are",
            "Ratio of within-cluster to between-cluster distance",
            "Sum of squared distances to cluster center"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True)

with tab2:
    st.subheader("PCA — explained variance")
    pca_model = pickle.load(open("src/models/pca_model.pkl","rb"))
    var       = pca_model.explained_variance_ratio_

    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(var))],
        "Variance %": (var * 100).round(2),
        "Cumulative %": (var.cumsum() * 100).round(2)
    })
    st.dataframe(var_df, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=var_df["Component"],
        y=var_df["Variance %"],
        name="Individual", marker_color="#7F77DD"
    ))
    fig2.add_trace(go.Scatter(
        x=var_df["Component"],
        y=var_df["Cumulative %"],
        name="Cumulative", mode="lines+markers",
        line=dict(color="#D85A30", width=2),
        yaxis="y2"
    ))
    fig2.update_layout(
        title="PCA scree plot",
        yaxis=dict(title="Individual variance %"),
        yaxis2=dict(title="Cumulative %",
                    overlaying="y", side="right"),
        height=420, legend=dict(x=0.7, y=0.3)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("t-SNE parameters")
    tsne_params = pd.DataFrame({
        "Parameter":   ["Perplexity","Iterations",
                        "Learning rate","Sample size"],
        "Value":       ["40","1000","200","15,000"],
        "Effect":      [
            "Controls local vs global structure balance",
            "More iterations = better convergence",
            "Controls step size in optimization",
            "Stratified sample across all crime types"
        ]
    })
    st.dataframe(tsne_params, use_container_width=True)

with tab3:
    st.subheader("MLflow experiment runs")
    st.info("To view the full interactive MLflow UI, "
            "run `mlflow ui` in your terminal and "
            "visit http://localhost:5000")

    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"]
            )
            if not runs:
                continue

            st.markdown(f"**Experiment: {exp.name}**")
            run_data = []
            for run in runs:
                run_data.append({
                    "Run name":  run.data.tags.get(
                        "mlflow.runName","unnamed"),
                    "Status":    run.info.status,
                    **{k: round(v, 4) for k, v
                       in run.data.metrics.items()},
                })
            st.dataframe(
                pd.DataFrame(run_data),
                use_container_width=True
            )
    except Exception as e:
        st.warning(f"Could not load MLflow data: {e}")
