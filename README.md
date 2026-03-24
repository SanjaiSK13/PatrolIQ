# PatrolIQ — Smart Safety Analytics Platform

An enterprise-grade **Geospatial AI & Urban Analytics platform** that transforms 500,000 raw Chicago crime records into **actionable tactical intelligence** for law enforcement, urban planners, and public safety agencies.

The platform identifies **hidden spatial patterns**, **severity-weighted hotspots**, and **temporal crime clusters** using **unsupervised machine learning** and advanced interactive visualization — deployed live on Streamlit Cloud.

---

## Live Demo

**[PatrolIQ — Live App](https://your-app-url.streamlit.app)**

---

## Overview

PatrolIQ bridges the gap between traditional crime dashboards and intelligent urban analytics systems. Instead of merely reporting historical incidents, the platform answers the questions law enforcement asks every day:

- *Where should we patrol tonight?*
- *Which neighborhoods need more resources?*
- *When do most crimes occur?*

By applying K-Means, DBSCAN, Hierarchical Clustering, PCA, and t-SNE to 500,000 Chicago crime records, the platform surfaces patterns invisible to the human eye — and presents them through an interactive, production-ready dashboard.

---

## Key Capabilities

- 500,000 crime records processed from the Chicago Data Portal (7.8M source)
- Geographic crime hotspot detection using 3 clustering algorithms
- Temporal pattern analysis — hourly, daily, monthly, seasonal
- Dimensionality reduction for high-dimensional crime pattern visualization
- MLflow experiment tracking for all ML runs
- Interactive folium heatmaps with severity weighting
- Fully deployed on Streamlit Cloud with CI/CD via GitHub

---

## Tech Stack

### Core
- Python 3.10
- Streamlit 1.31.0
- Pandas, NumPy, SciPy

### Machine Learning
- Scikit-learn — K-Means, DBSCAN, Hierarchical Clustering, PCA
- Scikit-learn — t-SNE (manifold)
- MLflow — experiment tracking and model registry

### Visualization
- Plotly — interactive charts and 3D scatter plots
- Folium + streamlit-folium — interactive crime heatmaps
- Seaborn, Matplotlib — EDA plots

### Deployment
- Streamlit Cloud
- GitHub CI/CD

---

## System Architecture

```
Chicago Crime Dataset (7.8M Records)
            ↓
    Download & Sample (500K Records)
            ↓
    Data Cleaning & Validation
            ↓
    Feature Engineering (22 → 26 features)
    Hour · Day · Season · Severity Score
            ↓
        ┌───────────────────────┐
        │   ML Intelligence     │
        │                       │
        │  Geographic           │
        │  ├─ K-Means           │
        │  ├─ DBSCAN            │
        │  └─ Hierarchical      │
        │                       │
        │  Temporal             │
        │  └─ K-Means           │
        │                       │
        │  Dimensionality       │
        │  ├─ PCA               │
        │  └─ t-SNE             │
        └───────────────────────┘
            ↓
    MLflow Experiment Tracking
            ↓
    Streamlit Multi-Page App
            ↓
    Streamlit Cloud Deployment
```

---

## Project Structure

```
PatrolIQ/
│   .gitignore
│   requirements.txt
│   README.md
│
├── app/
│   │   Home.py                      ← Main dashboard entry point
│   │   utils.py                     ← Shared path helpers
│   │
│   └── pages/
│           1_Crime_Map.py           ← Interactive folium heatmap
│           2_Temporal.py            ← Hourly / daily / seasonal analysis
│           3_Clustering.py          ← K-Means, DBSCAN, Hierarchical
│           4_Dim_Reduction.py       ← PCA and t-SNE visualizations
│           5_Model_Performance.py   ← Metrics and MLflow summary
│
├── data/
│       crimes_raw.csv               ← Full download (not in git, 1.7GB)
│       crimes_sample.parquet        ← 500K sampled records
│       crimes_clean.parquet         ← Cleaned + engineered features
│       crimes_clustered.parquet     ← With cluster labels added
│       clustering_comparison.parquet
│       dbscan_results.parquet
│       hierarchical_results.parquet
│       kmeans_cluster_summary.parquet
│       temporal_cluster_profiles.parquet
│       pca_results.parquet
│       tsne_results.parquet
│
├── notebooks/
│       01_EDA.ipynb                 ← Exploratory data analysis
│       02_Clustering.ipynb          ← All clustering algorithms
│       03_Dimensionality_Reduction.ipynb ← PCA and t-SNE
│
│   └── plots/                       ← All saved visualizations
│       ├── (10 EDA plots)
│       └── clustering/ (6 plots)
│       └── dimred/ (5 plots)
│
├── src/
│       sample_data.py               ← Download, sample, feature engineer
│
│   └── models/
│           kmeans_geo.pkl
│           dbscan_geo.pkl
│           hierarchical_geo.pkl
│           kmeans_temporal.pkl
│           pca_model.pkl
│           scaler_geo.pkl
│           scaler_temporal.pkl
│           scaler_full.pkl
│           le_crime.pkl
│           le_location.pkl
│
└── mlruns/                          ← MLflow experiment data (local)
```

---

## Dataset

**Source:** [Chicago Data Portal — Crimes 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

| Property | Value |
|---|---|
| Full dataset size | 7.8 Million records (1.7 GB) |
| Sample used | 500,000 records (2015–2024) |
| Input features | 22 original + 4 engineered |
| Crime categories | 33 distinct types |
| Geographic coverage | City of Chicago, 22 districts |

### Engineered Features

| Feature | Description |
|---|---|
| `Hour` | Hour of day (0–23) extracted from datetime |
| `Day_of_Week` | Day name (Monday–Sunday) |
| `Month` | Month number (1–12) |
| `Season` | Winter / Spring / Summer / Fall |
| `Is_Weekend` | Boolean flag for Saturday/Sunday |
| `Crime_Severity_Score` | Custom 1–10 score by crime type |

---

## ML Models & Results

### Geographic Clustering

| Algorithm | Clusters | Silhouette Score | Best for |
|---|---|---|---|
| K-Means | 8 | ~0.45 | Patrol zone planning |
| DBSCAN | Auto-detected | ~0.42 | Natural hotspot detection |
| Hierarchical | 8 | ~0.43 | Zone hierarchy analysis |

### Temporal Clustering

| Algorithm | Clusters | Description |
|---|---|---|
| K-Means | 4 | Late-night · Morning · Afternoon · Weekend patterns |

### Dimensionality Reduction

| Technique | Input | Output | Variance Explained |
|---|---|---|---|
| PCA | 18 features | 3 components | ~65–75% |
| t-SNE | 18 features | 2D projection | 15,000-point sample |

---

## MLflow Experiment Tracking

6 runs logged across 2 experiments:

| Experiment | Run | Key Metrics |
|---|---|---|
| PatrolIQ_Clustering | KMeans_Geographic | silhouette, davies_bouldin, inertia |
| PatrolIQ_Clustering | DBSCAN_Geographic | n_clusters, noise_pct, silhouette |
| PatrolIQ_Clustering | Hierarchical_Geographic | silhouette, davies_bouldin |
| PatrolIQ_Clustering | KMeans_Temporal | silhouette, inertia |
| PatrolIQ_DimReduction | PCA_CrimeFeatures | pc1_variance, total_variance |
| PatrolIQ_DimReduction | tSNE_CrimeFeatures | kl_divergence |

To view locally:
```bash
mlflow ui
```
Then visit `http://localhost:5000`

---

## Dashboard Pages

### Home
Executive summary with 5 KPI metrics, crime type distribution, hourly pattern, and severity heatmap.

### Crime Map
Interactive folium map with 3 modes — heatmap, K-Means cluster zones with centroids, and marker clustering. Filterable by crime type and year.

### Temporal Patterns
4-tab analysis covering hourly frequency, day-of-week heatmap, monthly/seasonal breakdown, and temporal cluster profiles.

### Clustering Analysis
Side-by-side exploration of all 3 geographic clustering algorithms plus a comparison view with silhouette scores and algorithm characteristics.

### Dimensionality Reduction
PCA 2D and 3D scatter plots with color options, plus t-SNE visualizations colored by crime type, cluster, hour, and season.

### Model Performance
Silhouette score comparison chart, PCA scree plot, evaluation metrics explanation, and MLflow experiment summary.

---

## Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PatrolIQ.git
cd PatrolIQ

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Visit: https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD
# Save as: data/crimes_raw.csv

# Run sampling pipeline
python src/sample_data.py

# Launch app
streamlit run app/Home.py
```

---

## Deployment

The app is deployed on **Streamlit Cloud** connected to this GitHub repository.

Any push to the `main` branch automatically triggers a redeployment.

```
GitHub push → Streamlit Cloud pulls → Redeploys automatically
```

---

## Business Impact

| Stakeholder | Value Delivered |
|---|---|
| Police Departments | Identify 8 distinct patrol zones with crime concentration data |
| City Administration | Data-driven evidence for resource allocation decisions |
| Urban Planners | Seasonal and geographic crime trends for infrastructure planning |
| Emergency Services | High-risk time slots and zones for proactive deployment |

---

## Author

**Your Name**
Domain: Public Safety · Urban Analytics · Unsupervised Machine Learning

---

## License

This project is for educational purposes. Crime data sourced from the Chicago Data Portal under the City of Chicago Open Data License.
