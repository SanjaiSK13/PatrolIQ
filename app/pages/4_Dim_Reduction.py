import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path

# --- PATH CONFIGURATION ---
# Get the absolute path to the project root (PatrolIQ)
# Assumes this script is in PatrolIQ/app/pages/
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "src" / "models"

st.set_page_config(page_title="Dimensionality Reduction", layout="wide")
st.title("Dimensionality Reduction")
st.markdown("PCA and t-SNE visualizations of high-dimensional crime patterns")

@st.cache_data
def load_pca():
    path = DATA_DIR / "pca_results.parquet"
    if not path.exists():
        st.error(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data
def load_tsne():
    path = DATA_DIR / "tsne_results.parquet"
    if not path.exists():
        st.error(f"Missing file: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)

pca_df  = load_pca()
tsne_df = load_tsne()

tab1, tab2 = st.tabs(["PCA", "t-SNE"])

with tab1:
    st.subheader("Principal Component Analysis")

    # Load PCA Model
    model_path = MODEL_DIR / "pca_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            pca_model = pickle.load(f)
        
        var = pca_model.explained_variance_ratio_
        col1, col2, col3 = st.columns(3)
        col1.metric("PC1 variance", f"{var[0]*100:.1f}%")
        col2.metric("PC2 variance", f"{var[1]*100:.1f}%")
        col3.metric("Total (3 PCs)", f"{sum(var[:3])*100:.1f}%")
    else:
        st.warning("PCA model file not found in src/models/")

    color_by = st.selectbox(
        "Color points by",
        ["KMeans_Cluster", "Crime_Severity_Score", "Hour_Label"],
        key="pca_color"
    )

    # Sampling for performance
    sample_size = min(30000, len(pca_df))
    sample_pca = pca_df.sample(sample_size, random_state=42) if not pca_df.empty else pca_df

    if not sample_pca.empty:
        # 2D PCA Plot - Added render_mode="svg" to fix WebGL errors
        fig = px.scatter(
            sample_pca, x="PC1", y="PC2",
            color=sample_pca[color_by].astype(str) if color_by == "KMeans_Cluster" else color_by,
            title="PCA 2D projection",
            opacity=0.4, height=520,
            color_discrete_sequence=px.colors.qualitative.T10 if color_by == "KMeans_Cluster" else None,
            color_continuous_scale="RdYlGn_r" if color_by == "Crime_Severity_Score" else "Viridis",
            render_mode="svg" # <-- FIXES WEBGL ERROR
        )
        fig.update_traces(marker=dict(size=2))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("3D PCA projection")
        # 3D plots ALWAYS require WebGL. If this fails, use a standalone browser (Chrome/Edge).
        sample_3d = pca_df.sample(min(15000, len(pca_df)), random_state=42)
        fig3d = px.scatter_3d(
            sample_3d, x="PC1", y="PC2", z="PC3",
            color=sample_3d["KMeans_Cluster"].astype(str),
            opacity=0.5, height=600,
            color_discrete_sequence=px.colors.qualitative.T10,
            title="PCA 3D projection — colored by cluster"
        )
        fig3d.update_traces(marker=dict(size=2))
        st.plotly_chart(fig3d, use_container_width=True)

with tab2:
    st.subheader("t-SNE visualization")
    if not tsne_df.empty:
        st.caption(f"Based on {len(tsne_df):,} stratified sample")

        color_tsne = st.selectbox(
            "Color points by",
            ["Primary_Type", "KMeans_Cluster", "Hour", "Severity", "Season"],
            key="tsne_color"
        )

        # 2D t-SNE Plot
        is_discrete = color_tsne in ["Primary_Type", "Season", "KMeans_Cluster"]
        
        fig_t = px.scatter(
            tsne_df, x="tsne_x", y="tsne_y",
            color=tsne_df[color_tsne].astype(str) if is_discrete else color_tsne,
            title=f"t-SNE — colored by {color_tsne}",
            opacity=0.6, height=650,
            color_discrete_sequence=px.colors.qualitative.T10 if is_discrete else None,
            color_continuous_scale="Viridis" if not is_discrete else None,
            render_mode="svg" # <-- FIXES WEBGL ERROR
        )
        fig_t.update_traces(marker=dict(size=3))
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Run the t-SNE processing script first to generate data.")