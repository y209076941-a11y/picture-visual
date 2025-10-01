# pages/5_🤖_Machine_Learning.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Path Configuration and Module Import
# ============================================================================

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    # Machine learning imports
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        classification_report,
        confusion_matrix,
        mean_squared_error,
        r2_score
    )
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # Custom modules
    from utils.data_manager import DataManager
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box

except ImportError as e:
    logger.error(f"Module import failed: {e}")
    st.error(f"⚠️ Critical module import error: {e}")
    st.info("Please install required packages: scikit-learn, plotly")
    st.stop()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Machine Learning - SYPHU iGEM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Constants and Configuration
# ============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Clustering algorithms
CLUSTERING_ALGORITHMS = {
    "K-Means": {
        "description": "Partition-based clustering algorithm",
        "pros": ["Fast", "Scalable", "Well-understood"],
        "cons": ["Requires number of clusters", "Sensitive to initialization"],
        "best_for": "Spherical, evenly-sized clusters"
    },
    "DBSCAN": {
        "description": "Density-based spatial clustering",
        "pros": ["Finds arbitrary shapes", "Robust to outliers"],
        "cons": ["Sensitive to parameters", "Varying densities"],
        "best_for": "Non-spherical clusters, noise detection"
    },
    "Hierarchical": {
        "description": "Bottom-up hierarchical clustering",
        "pros": ["No need to specify k", "Dendrogram visualization"],
        "cons": ["Computationally expensive", "Memory intensive"],
        "best_for": "Small to medium datasets, hierarchical structure"
    }
}

# Dimensionality reduction methods
DIM_REDUCTION_METHODS = {
    "PCA": {
        "description": "Principal Component Analysis (linear)",
        "interpretable": True,
        "best_for": "Linear relationships, variance preservation"
    },
    "t-SNE": {
        "description": "t-Distributed Stochastic Neighbor Embedding (non-linear)",
        "interpretable": False,
        "best_for": "Visualization, non-linear relationships"
    }
}

# Color scheme (Nature-friendly)
COLORS = ['#2E86AB', '#A9D6E5', '#4CAF50', '#FFC107', '#FF6B6B', '#9B59B6']


# ============================================================================
# Utility Functions
# ============================================================================

def validate_numeric_data(df: pd.DataFrame, min_cols: int = 2) -> Tuple[bool, Optional[str]]:
    """
    Validate if dataframe has sufficient numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    min_cols : int, optional
        Minimum required numeric columns (default: 2).

    Returns
    -------
    Tuple[bool, str or None]
        (is_valid, error_message).
    """
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) < min_cols:
        return False, f"Need at least {min_cols} numeric columns (found {len(numeric_cols)})"

    return True, None


def prepare_ml_data(
        df: pd.DataFrame,
        feature_cols: List[str],
        scale: bool = True,
        handle_missing: str = 'mean'
) -> Tuple[np.ndarray, Optional[Any]]:
    """
    Prepare data for machine learning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : List[str]
        Columns to use as features.
    scale : bool, optional
        Whether to scale data (default: True).
    handle_missing : str, optional
        How to handle missing values: 'mean', 'median', 'drop' (default: 'mean').

    Returns
    -------
    Tuple[np.ndarray, scaler or None]
        (processed_data, scaler_object).
    """
    # Extract features
    X = df[feature_cols].copy()

    # Handle missing values
    if handle_missing == 'mean':
        X = X.fillna(X.mean())
    elif handle_missing == 'median':
        X = X.fillna(X.median())
    elif handle_missing == 'drop':
        X = X.dropna()

    # Convert to numpy
    X_array = X.values

    # Scale if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)

    return X_array, scaler


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate clustering evaluation metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Cluster labels.

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.

    Notes
    -----
    Metrics include:
    - Silhouette Score: [-1, 1], higher is better
    - Calinski-Harabasz Index: higher is better
    - Davies-Bouldin Index: lower is better
    """
    metrics = {}

    # Filter out noise points (-1 labels) for DBSCAN
    valid_mask = labels >= 0
    if valid_mask.sum() < 2:
        return {"error": "Insufficient valid clusters"}

    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    n_clusters = len(set(labels_valid))

    if n_clusters > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X_valid, labels_valid)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_valid, labels_valid)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_valid, labels_valid)
            metrics['n_clusters'] = n_clusters
            metrics['n_samples'] = len(labels_valid)
            metrics['noise_points'] = (labels == -1).sum()
        except Exception as e:
            logger.error(f"Metric calculation error: {e}")
            metrics['error'] = str(e)

    return metrics


# ============================================================================
# Main Page Rendering
# ============================================================================

def main():
    """Main function to render Machine Learning page."""

    render_sidebar()

    render_page_header(
        title="Machine Learning Laboratory",
        icon="🤖",
        subtitle="Advanced ML algorithms for biological data analysis"
    )

    # Data validation
    if not DataManager.validate_dataset():
        render_no_data_warning()
        return

    df = st.session_state.current_dataset
    dataset_info = DataManager.get_dataset_info()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Display dataset summary
    render_dataset_summary(dataset_info, numeric_cols)

    # Render ML tabs
    render_ml_tabs(df, numeric_cols)


def render_no_data_warning():
    """Display warning when no dataset is loaded."""
    render_info_box(
        content="""
        **No active dataset detected.**

        Please load a dataset in the Data Management Hub before proceeding with machine learning analysis.
        """,
        box_type="warning",
        title="Data Required"
    )

    if st.button("📂 Go to Data Hub", type="primary", use_container_width=True):
        st.switch_page("pages/2_📂_Data_Hub.py")


def render_dataset_summary(info: Dict[str, Any], numeric_cols: List[str]):
    """Display dataset summary."""
    st.success(f"**📊 Active Dataset:** `{info['name']}`")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Observations", f"{info['shape'][0]:,}")
    with col2:
        st.metric("Variables", info['shape'][1])
    with col3:
        st.metric("Numeric Columns", len(numeric_cols))
    with col4:
        st.metric("Memory", info.get('memory_usage', 'Unknown'))


def render_ml_tabs(df: pd.DataFrame, numeric_cols: List[str]):
    """Render machine learning analysis tabs."""

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Clustering",
        "📉 Dimensionality Reduction",
        "🔮 Classification & Regression",
        "📊 Model Evaluation"
    ])

    with tab1:
        render_clustering_tab(df, numeric_cols)

    with tab2:
        render_dimensionality_reduction_tab(df, numeric_cols)

    with tab3:
        render_supervised_learning_tab(df, numeric_cols)

    with tab4:
        render_evaluation_tab()


# ============================================================================
# Tab 1: Clustering Analysis
# ============================================================================

def render_clustering_tab(df: pd.DataFrame, numeric_cols: List[str]):
    """Render clustering analysis interface."""

    render_section_header("Unsupervised Clustering Analysis", "🔍")

    is_valid, error_msg = validate_numeric_data(df, min_cols=2)
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        render_clustering_controls(df, numeric_cols)

    with col2:
        render_clustering_info()


def render_clustering_controls(df: pd.DataFrame, numeric_cols: List[str]):
    """Render clustering control panel."""

    st.markdown("#### Configuration")

    # Feature selection
    selected_features = st.multiselect(
        "Select feature columns",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        help="Choose variables for clustering analysis"
    )

    # Algorithm selection
    clustering_method = st.selectbox(
        "Clustering algorithm",
        list(CLUSTERING_ALGORITHMS.keys()),
        help="Select clustering method based on your data characteristics"
    )

    # Algorithm-specific parameters
    if clustering_method == "K-Means":
        params = render_kmeans_params()
    elif clustering_method == "DBSCAN":
        params = render_dbscan_params()
    else:  # Hierarchical
        params = render_hierarchical_params()

    # Preprocessing options
    st.markdown("---")
    st.markdown("#### Preprocessing")

    scale_data = st.checkbox("Standardize features", value=True, help="Z-score normalization")
    handle_missing = st.selectbox("Handle missing values", ["mean", "median", "drop"])

    # Run button
    st.markdown("---")
    if st.button("🚀 Run Clustering Analysis", type="primary", use_container_width=True):
        if not selected_features:
            st.error("⚠️ Please select at least one feature")
        else:
            run_clustering_analysis(
                df, selected_features, clustering_method,
                params, scale_data, handle_missing
            )


def render_kmeans_params() -> Dict[str, Any]:
    """Render K-Means parameters."""
    with st.expander("K-Means Parameters", expanded=True):
        n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
        max_iter = st.slider("Maximum iterations", 100, 1000, 300, 100)
        n_init = st.slider("Number of initializations", 5, 20, 10)

        return {
            'n_clusters': n_clusters,
            'max_iter': max_iter,
            'n_init': n_init,
            'random_state': RANDOM_STATE
        }


def render_dbscan_params() -> Dict[str, Any]:
    """Render DBSCAN parameters."""
    with st.expander("DBSCAN Parameters", expanded=True):
        eps = st.slider("Neighborhood radius (ε)", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("Minimum samples", 2, 20, 5)

        st.info("💡 **Tip:** Start with ε=0.5 and min_samples=5, then adjust based on results")

        return {
            'eps': eps,
            'min_samples': min_samples
        }


def render_hierarchical_params() -> Dict[str, Any]:
    """Render Hierarchical clustering parameters."""
    with st.expander("Hierarchical Parameters", expanded=True):
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])

        return {
            'n_clusters': n_clusters,
            'linkage': linkage
        }


def render_clustering_info():
    """Display clustering algorithm information."""
    st.markdown("#### Algorithm Information")

    for method, info in CLUSTERING_ALGORITHMS.items():
        with st.expander(f"{method}", expanded=False):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Best for:** {info['best_for']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pros:**")
                for pro in info['pros']:
                    st.write(f"✓ {pro}")
            with col2:
                st.markdown("**Cons:**")
                for con in info['cons']:
                    st.write(f"× {con}")


def run_clustering_analysis(
        df: pd.DataFrame,
        features: List[str],
        method: str,
        params: Dict[str, Any],
        scale: bool,
        handle_missing: str
):
    """Execute clustering analysis."""

    with st.spinner(f"Running {method} clustering..."):
        try:
            # Prepare data
            X, scaler = prepare_ml_data(df, features, scale, handle_missing)

            # Perform clustering
            if method == "K-Means":
                model = KMeans(**params)
            elif method == "DBSCAN":
                model = DBSCAN(**params)
            else:  # Hierarchical
                model = AgglomerativeClustering(**params)

            labels = model.fit_predict(X)

            # Evaluate clustering
            metrics = evaluate_clustering(X, labels)

            if 'error' in metrics:
                st.error(f"⚠️ Clustering evaluation error: {metrics['error']}")
                return

            # Visualize results
            visualize_clustering_results(X, labels, features, method, metrics)

            # Store results
            store_clustering_results(method, features, params, metrics, labels)

            st.success("✅ Clustering analysis completed successfully!")

        except Exception as e:
            logger.error(f"Clustering analysis error: {e}")
            st.error(f"⚠️ Analysis failed: {str(e)}")


def visualize_clustering_results(
        X: np.ndarray,
        labels: np.ndarray,
        features: List[str],
        method: str,
        metrics: Dict[str, float]
):
    """Visualize clustering results."""

    st.markdown("---")
    st.markdown("### Results")

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Clusters Found", metrics.get('n_clusters', 0))
    with col2:
        st.metric("Silhouette Score", f"{metrics.get('silhouette_score', 0):.3f}")
    with col3:
        st.metric("Calinski-Harabasz", f"{metrics.get('calinski_harabasz_score', 0):.1f}")
    with col4:
        if 'noise_points' in metrics:
            st.metric("Noise Points", metrics['noise_points'])
        else:
            st.metric("Davies-Bouldin", f"{metrics.get('davies_bouldin_score', 0):.3f}")

    # Interpretation
    silhouette = metrics.get('silhouette_score', 0)
    if silhouette > 0.7:
        st.success("✅ Excellent clustering structure (Silhouette > 0.7)")
    elif silhouette > 0.5:
        st.info("Good clustering structure (Silhouette > 0.5)")
    elif silhouette > 0.25:
        st.warning("⚠️ Weak clustering structure (Silhouette > 0.25)")
    else:
        st.error("Poor clustering structure (Silhouette < 0.25) - consider different parameters")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Cluster distribution
        cluster_counts = pd.Series(labels[labels >= 0]).value_counts().sort_index()
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="Cluster Size Distribution",
            color_discrete_sequence=COLORS
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Cluster sizes bar chart
        fig_bar = px.bar(
            x=[f"C{i}" for i in cluster_counts.index],
            y=cluster_counts.values,
            title="Samples per Cluster",
            labels={'x': 'Cluster', 'y': 'Count'},
            color_discrete_sequence=[COLORS[0]]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # PCA visualization
    st.markdown("---")
    st.markdown("#### Cluster Visualization (PCA)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    result_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f"Cluster {l}" if l >= 0 else "Noise" for l in labels]
    })

    fig_scatter = px.scatter(
        result_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=f"{method} Clustering Results (PCA Projection)",
        color_discrete_sequence=COLORS,
        hover_data={'PC1': ':.3f', 'PC2': ':.3f'}
    )

    fig_scatter.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


def store_clustering_results(
        method: str,
        features: List[str],
        params: Dict[str, Any],
        metrics: Dict[str, float],
        labels: np.ndarray
):
    """Store clustering results in session state."""

    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {}

    st.session_state.ml_results['clustering'] = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'features': features,
        'parameters': params,
        'metrics': metrics,
        'labels': labels.tolist()
    }


# ============================================================================
# Tab 2: Dimensionality Reduction
# ============================================================================

def render_dimensionality_reduction_tab(df: pd.DataFrame, numeric_cols: List[str]):
    """Render dimensionality reduction interface."""

    render_section_header("Dimensionality Reduction", "📉")

    is_valid, error_msg = validate_numeric_data(df, min_cols=3)
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        render_dimred_controls(df, numeric_cols)

    with col2:
        render_dimred_info()


def render_dimred_controls(df: pd.DataFrame, numeric_cols: List[str]):
    """Render dimensionality reduction controls."""

    st.markdown("#### Configuration")

    dim_method = st.selectbox(
        "Reduction method",
        list(DIM_REDUCTION_METHODS.keys()),
        help="Choose dimensionality reduction algorithm"
    )

    # Method-specific parameters
    if dim_method == "PCA":
        n_components = st.slider(
            "Number of components",
            2, min(10, len(numeric_cols)), 2,
            help="How many dimensions to reduce to"
        )
        params = {'n_components': n_components}
    else:  # t-SNE
        n_components = 2  # t-SNE typically uses 2D
        perplexity = st.slider("Perplexity", 5, 50, 30)
        learning_rate = st.slider("Learning rate", 10, 1000, 200, 10)
        params = {
            'n_components': n_components,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'random_state': RANDOM_STATE
        }

    # Visualization options
    st.markdown("---")
    st.markdown("#### Visualization")

    color_by = st.selectbox("Color by", [None] + numeric_cols)
    scale_data = st.checkbox("Standardize features", value=True)

    # Run button
    st.markdown("---")
    if st.button("🚀 Run Dimensionality Reduction", type="primary", use_container_width=True):
        run_dimred_analysis(df, numeric_cols, dim_method, params, color_by, scale_data)


def render_dimred_info():
    """Display dimensionality reduction information."""
    st.markdown("#### Method Information")

    for method, info in DIM_REDUCTION_METHODS.items():
        with st.expander(f"{method}", expanded=False):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Interpretable:** {'Yes' if info['interpretable'] else 'No'}")
            st.markdown(f"**Best for:** {info['best_for']}")


def run_dimred_analysis(
        df: pd.DataFrame,
        numeric_cols: List[str],
        method: str,
        params: Dict[str, Any],
        color_by: Optional[str],
        scale: bool
):
    """Execute dimensionality reduction."""

    with st.spinner(f"Running {method} analysis..."):
        try:
            # Prepare data
            X, _ = prepare_ml_data(df, numeric_cols, scale, 'mean')

            # Perform reduction
            if method == "PCA":
                reducer = PCA(**params)
            else:  # t-SNE
                reducer = TSNE(**params)

            X_reduced = reducer.fit_transform(X)

            # Visualize results
            visualize_dimred_results(X_reduced, df, method, reducer, color_by, params)

            # Store results
            store_dimred_results(method, numeric_cols, params, X_reduced, reducer)

            st.success(f"✅ {method} analysis completed successfully!")

        except Exception as e:
            logger.error(f"Dimensionality reduction error: {e}")
            st.error(f"⚠️ Analysis failed: {str(e)}")


def visualize_dimred_results(
        X_reduced: np.ndarray,
        df: pd.DataFrame,
        method: str,
        reducer: Any,
        color_by: Optional[str],
        params: Dict[str, Any]
):
    """Visualize dimensionality reduction results."""

    st.markdown("---")
    st.markdown("### Results")

    # PCA-specific: variance explained
    if method == "PCA":
        variance_ratio = reducer.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)

        col1, col2 = st.columns(2)

        with col1:
            # Variance bar chart
            fig_var = px.bar(
                x=[f"PC{i + 1}" for i in range(len(variance_ratio))],
                y=variance_ratio,
                title="Variance Explained by Component",
                labels={'x': 'Component', 'y': 'Variance Ratio'},
                color_discrete_sequence=[COLORS[0]]
            )
            st.plotly_chart(fig_var, use_container_width=True)

        with col2:
            # Cumulative variance
            fig_cum = px.line(
                x=[f"PC{i + 1}" for i in range(len(cumulative_variance))],
                y=cumulative_variance,
                title="Cumulative Variance Explained",
                labels={'x': 'Component', 'y': 'Cumulative Variance'},
                markers=True
            )
            fig_cum.update_traces(line_color=COLORS[1])
            st.plotly_chart(fig_cum, use_container_width=True)

        # Display metrics
        st.markdown("#### Variance Statistics")
        for i, (var, cum_var) in enumerate(zip(variance_ratio, cumulative_variance)):
            st.write(f"PC{i + 1}: {var:.3f} (Cumulative: {cum_var:.3f})")

    # 2D Visualization
    st.markdown("---")
    st.markdown("#### Reduced Dimensions Visualization")

    if X_reduced.shape[1] >= 2:
        result_df = pd.DataFrame({
            'Dim1': X_reduced[:, 0],
            'Dim2': X_reduced[:, 1]
        })

        if color_by and color_by in df.columns:
            result_df['Color'] = df[color_by].values

            fig = px.scatter(
                result_df,
                x='Dim1',
                y='Dim2',
                color='Color',
                title=f"{method} Results (colored by {color_by})",
                color_continuous_scale='Viridis',
                hover_data={'Dim1': ':.3f', 'Dim2': ':.3f', 'Color': ':.3f'}
            )
        else:
            fig = px.scatter(
                result_df,
                x='Dim1',
                y='Dim2',
                title=f"{method} Results",
                color_discrete_sequence=[COLORS[0]],
                hover_data={'Dim1': ':.3f', 'Dim2': ':.3f'}
            )

        fig.update_layout(
            xaxis_title=f"{'PC1' if method == 'PCA' else 'Dimension 1'}",
            yaxis_title=f"{'PC2' if method == 'PCA' else 'Dimension 2'}"
        )

        st.plotly_chart(fig, use_container_width=True)


def store_dimred_results(
        method: str,
        features: List[str],
        params: Dict[str, Any],
        X_reduced: np.ndarray,
        reducer: Any
):
    """Store dimensionality reduction results."""

    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {}

    results = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'features': features,
        'parameters': params,
        'reduced_data': X_reduced.tolist()
    }

    if method == "PCA":
        results['variance_explained'] = reducer.explained_variance_ratio_.tolist()

    st.session_state.ml_results['dimensionality_reduction'] = results


# ============================================================================
# Tab 3: Supervised Learning
# ============================================================================

def render_supervised_learning_tab(df: pd.DataFrame, numeric_cols: List[str]):
    """Render supervised learning interface."""

    render_section_header("Supervised Learning", "🔮")

    render_info_box(
        content="""
        **Classification & Regression Analysis**

        This module supports supervised learning tasks including classification 
        and regression. Ensure your dataset contains a target variable (label) 
        for prediction.

        Current implementation uses Random Forest algorithms with proper 
        train-test splitting and cross-validation.
        """,
        box_type="info",
        title="About Supervised Learning"
    )

    is_valid, error_msg = validate_numeric_data(df, min_cols=2)
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        render_supervised_controls(df, numeric_cols)

    with col2:
        render_supervised_info()


def render_supervised_controls(df: pd.DataFrame, numeric_cols: List[str]):
    """Render supervised learning controls."""

    st.markdown("#### Configuration")

    # Target variable selection
    target_col = st.selectbox(
        "Target variable",
        numeric_cols,
        help="Select the variable to predict"
    )

    # Problem type detection
    problem_type = st.selectbox(
        "Problem type",
        ["Auto-detect", "Classification", "Regression"]
    )

    # Auto-detect if requested
    detected_type = None
    if problem_type == "Auto-detect" and target_col:
        unique_values = df[target_col].nunique()
        if unique_values < 10:
            detected_type = "Classification"
        else:
            detected_type = "Regression"
        st.info(f"🔍 Detected as: **{detected_type}** ({unique_values} unique values)")
        final_type = detected_type
    else:
        final_type = problem_type

    # Feature selection
    feature_cols = st.multiselect(
        "Feature variables",
        [col for col in numeric_cols if col != target_col],
        default=[col for col in numeric_cols[:min(5, len(numeric_cols))] if col != target_col],
        help="Select features for prediction"
    )

    st.markdown("---")
    st.markdown("#### Model Settings")

    # Train-test split
    test_size = st.slider("Test set proportion", 0.1, 0.5, 0.2, 0.05)

    # Model parameters
    with st.expander("Advanced Parameters", expanded=False):
        n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
        max_depth = st.slider("Maximum depth", 2, 20, 10)
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': RANDOM_STATE
    }

    # Run button
    st.markdown("---")
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        if not feature_cols:
            st.error("⚠️ Please select at least one feature")
        elif not target_col:
            st.error("⚠️ Please select a target variable")
        else:
            run_supervised_learning(
                df, feature_cols, target_col,
                final_type, test_size, params, cv_folds
            )


def render_supervised_info():
    """Display supervised learning information."""

    st.markdown("#### Random Forest Algorithm")

    with st.expander("About Random Forests", expanded=False):
        st.markdown("""
        **Random Forest** is an ensemble learning method that:

        **Advantages:**
        - Handles both classification and regression
        - Robust to overfitting
        - Provides feature importance
        - Works well with high-dimensional data
        - Minimal hyperparameter tuning needed

        **Best Practices:**
        - Use cross-validation for reliable estimates
        - Check for overfitting (train vs test performance)
        - Examine feature importances
        - Consider class imbalance in classification
        """)

    with st.expander("Evaluation Metrics", expanded=False):
        st.markdown("""
        **Classification Metrics:**
        - Accuracy: Overall correctness
        - Precision: True positives / Predicted positives
        - Recall: True positives / Actual positives
        - F1-Score: Harmonic mean of precision and recall

        **Regression Metrics:**
        - R² Score: Proportion of variance explained
        - RMSE: Root mean squared error
        - MAE: Mean absolute error
        """)


def run_supervised_learning(
        df: pd.DataFrame,
        features: List[str],
        target: str,
        problem_type: str,
        test_size: float,
        params: Dict[str, Any],
        cv_folds: int
):
    """Execute supervised learning."""

    with st.spinner(f"Training {problem_type} model..."):
        try:
            # Prepare data
            X = df[features].fillna(df[features].mean())
            y = df[target].fillna(df[target].mean())

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE
            )

            # Choose model
            if problem_type == "Classification":
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds)

            # Visualize results
            visualize_supervised_results(
                model, X_train, X_test, y_train, y_test,
                y_pred_train, y_pred_test, features,
                problem_type, cv_scores
            )

            # Store results
            store_supervised_results(
                problem_type, features, target, params,
                model, y_test, y_pred_test, cv_scores
            )

            st.success("✅ Model training completed successfully!")

        except Exception as e:
            logger.error(f"Supervised learning error: {e}")
            st.error(f"⚠️ Training failed: {str(e)}")


def visualize_supervised_results(
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
        features: List[str],
        problem_type: str,
        cv_scores: np.ndarray
):
    """Visualize supervised learning results."""

    st.markdown("---")
    st.markdown("### Model Performance")

    if problem_type == "Classification":
        visualize_classification_results(
            y_train, y_test, y_pred_train, y_pred_test, cv_scores
        )
    else:
        visualize_regression_results(
            y_train, y_test, y_pred_train, y_pred_test, cv_scores
        )

    # Feature importance
    st.markdown("---")
    st.markdown("### Feature Importance")

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance Ranking",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_importance, use_container_width=True)


def visualize_classification_results(
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
        cv_scores: np.ndarray
):
    """Visualize classification results."""

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Train Accuracy", f"{train_acc:.3f}")
    with col2:
        st.metric("Test Accuracy", f"{test_acc:.3f}")
    with col3:
        st.metric("CV Mean", f"{cv_mean:.3f}")
    with col4:
        st.metric("CV Std", f"±{cv_std:.3f}")

    # Check for overfitting
    if train_acc - test_acc > 0.1:
        st.warning("⚠️ Possible overfitting detected (train accuracy much higher than test)")
    elif abs(train_acc - test_acc) < 0.05:
        st.success("✅ Good generalization (train and test performance similar)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix",
        color_continuous_scale='Blues',
        text_auto=True
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    with st.expander("Detailed Classification Report", expanded=False):
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)


def visualize_regression_results(
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
        cv_scores: np.ndarray
):
    """Visualize regression results."""

    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Display metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Train R²", f"{train_r2:.3f}")
        st.metric("Test R²", f"{test_r2:.3f}")
    with col2:
        st.metric("Train RMSE", f"{train_rmse:.3f}")
        st.metric("Test RMSE", f"{test_rmse:.3f}")
    with col3:
        st.metric("CV R² Mean", f"{cv_mean:.3f}")
        st.metric("CV R² Std", f"±{cv_std:.3f}")

    # Check performance
    if test_r2 > 0.7:
        st.success("✅ Strong predictive performance (R² > 0.7)")
    elif test_r2 > 0.5:
        st.info("Good predictive performance (R² > 0.5)")
    else:
        st.warning("⚠️ Moderate predictive performance (R² < 0.5)")

    # Prediction vs Actual plots
    col1, col2 = st.columns(2)

    with col1:
        # Test set
        fig_test = px.scatter(
            x=y_test,
            y=y_pred_test,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Test Set: Predicted vs Actual",
            trendline="ols"
        )
        fig_test.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        st.plotly_chart(fig_test, use_container_width=True)

    with col2:
        # Residuals
        residuals = y_test - y_pred_test
        fig_residuals = px.scatter(
            x=y_pred_test,
            y=residuals,
            labels={'x': 'Predicted', 'y': 'Residuals'},
            title="Residual Plot"
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)


def store_supervised_results(
        problem_type: str,
        features: List[str],
        target: str,
        params: Dict[str, Any],
        model: Any,
        y_test: pd.Series,
        y_pred_test: np.ndarray,
        cv_scores: np.ndarray
):
    """Store supervised learning results."""

    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {}

    results = {
        'timestamp': datetime.now().isoformat(),
        'problem_type': problem_type,
        'features': features,
        'target': target,
        'parameters': params,
        'cv_scores': cv_scores.tolist()
    }

    if problem_type == "Classification":
        from sklearn.metrics import accuracy_score
        results['test_accuracy'] = accuracy_score(y_test, y_pred_test)
    else:
        results['test_r2'] = r2_score(y_test, y_pred_test)
        results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))

    st.session_state.ml_results['supervised_learning'] = results


# ============================================================================
# Tab 4: Model Evaluation
# ============================================================================

def render_evaluation_tab():
    """Render model evaluation and history tab."""

    render_section_header("Model Evaluation & History", "📊")

    if 'ml_results' not in st.session_state or not st.session_state.ml_results:
        render_info_box(
            content="""
            **No analysis results available.**

            Please run clustering, dimensionality reduction, or supervised learning 
            analyses first. Results will be displayed here for review and comparison.
            """,
            box_type="info",
            title="No Results"
        )
        return

    st.markdown("### Analysis History")

    # Display each analysis result
    for analysis_type, results in st.session_state.ml_results.items():
        render_analysis_summary(analysis_type, results)


def render_analysis_summary(analysis_type: str, results: Dict[str, Any]):
    """Render summary of a specific analysis."""

    with st.expander(
            f"{analysis_type.replace('_', ' ').title()} - {results.get('timestamp', 'Unknown time')}",
            expanded=True
    ):
        if analysis_type == 'clustering':
            render_clustering_summary(results)
        elif analysis_type == 'dimensionality_reduction':
            render_dimred_summary(results)
        elif analysis_type == 'supervised_learning':
            render_supervised_summary(results)


def render_clustering_summary(results: Dict[str, Any]):
    """Display clustering results summary."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration:**")
        st.write(f"- Method: {results['method']}")
        st.write(f"- Features: {', '.join(results['features'])}")

    with col2:
        st.markdown("**Performance:**")
        metrics = results.get('metrics', {})
        st.write(f"- Clusters: {metrics.get('n_clusters', 'N/A')}")
        st.write(f"- Silhouette Score: {metrics.get('silhouette_score', 0):.3f}")


def render_dimred_summary(results: Dict[str, Any]):
    """Display dimensionality reduction summary."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration:**")
        st.write(f"- Method: {results['method']}")
        st.write(f"- Components: {results['parameters']['n_components']}")

    with col2:
        if 'variance_explained' in results:
            st.markdown("**Variance Explained:**")
            total_var = sum(results['variance_explained'])
            st.write(f"- Total: {total_var:.3f}")


def render_supervised_summary(results: Dict[str, Any]):
    """Display supervised learning summary."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration:**")
        st.write(f"- Type: {results['problem_type']}")
        st.write(f"- Target: {results['target']}")
        st.write(f"- Features: {len(results['features'])}")

    with col2:
        st.markdown("**Performance:**")
        if results['problem_type'] == "Classification":
            st.write(f"- Test Accuracy: {results.get('test_accuracy', 0):.3f}")
        else:
            st.write(f"- Test R²: {results.get('test_r2', 0):.3f}")
            st.write(f"- Test RMSE: {results.get('test_rmse', 0):.3f}")

        cv_scores = results.get('cv_scores', [])
        if cv_scores:
            st.write(f"- CV Mean: {np.mean(cv_scores):.3f}")


# ============================================================================
# Custom Styling
# ============================================================================

def apply_custom_styles():
    """Apply custom CSS for enhanced UI."""

    st.markdown("""
    <style>
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2E86AB;
        }

        /* Button hover effect */
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
            transition: all 0.3s ease;
        }

        /* Expander header */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1.1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    apply_custom_styles()
    main()
