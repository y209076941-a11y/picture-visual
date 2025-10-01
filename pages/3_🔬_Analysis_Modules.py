# pages/3_🔬_Analysis_Modules.py

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
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Path Configuration and Module Import
# ============================================================================

# Add project root directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import custom modules with comprehensive error handling
try:
    from utils.data_manager import DataManager
    from utils.data_loader import DataLoader
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box
except ImportError as e:
    logger.error(f"Module import failed: {e}")
    st.error(f"⚠️ Critical module import error: {e}")
    st.info("Please ensure all required modules are installed and accessible.")
    st.stop()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Analysis Modules - SYPHU iGEM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Constants and Configuration
# ============================================================================

# Statistical significance threshold
ALPHA = 0.05

# Visualization color schemes (colorblind-friendly)
COLOR_SCHEMES = {
    'sequential': 'Viridis',
    'diverging': 'RdBu_r',
    'categorical': 'Set2',
    'nature': ['#2E86AB', '#A9D6E5', '#4CAF50', '#FFC107', '#FF6B6B']
}

# Maximum data points for interactive plots
MAX_INTERACTIVE_POINTS = 10000

# Statistical test mappings
STATISTICAL_TESTS = {
    'normality': ['Shapiro-Wilk', 'Kolmogorov-Smirnov'],
    'correlation': ['Pearson', 'Spearman', 'Kendall'],
    'comparison': ['t-test', 'Mann-Whitney U', 'ANOVA', 'Kruskal-Wallis']
}


# ============================================================================
# Utility Functions
# ============================================================================

def get_enhanced_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    Dict[str, Any]
        Enhanced metadata including column types, missing values, and statistics.

    Notes
    -----
    This function extends the basic DataManager info with detailed
    statistical characteristics for scientific analysis.
    """

    try:
        basic_info = DataManager.get_dataset_info()

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        null_count = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        null_percentage = (null_count / total_cells * 100) if total_cells > 0 else 0

        return {
            **basic_info,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'null_count': null_count,
            'null_percentage': f"{null_percentage:.2f}%",
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_bytes': df.memory_usage(deep=True).sum()
        }
    except Exception as e:
        logger.error(f"Error extracting dataset info: {e}")
        return {}


def test_normality(data: pd.Series, method: str = 'shapiro') -> Tuple[float, float, str]:
    """
    Test data for normality.

    Parameters
    ----------
    data : pd.Series
        Data to test.
    method : str, optional
        Test method: 'shapiro' or 'kstest' (default: 'shapiro').

    Returns
    -------
    Tuple[float, float, str]
        (statistic, p-value, interpretation).

    Notes
    -----
    Shapiro-Wilk test is recommended for n < 5000.
    Kolmogorov-Smirnov test for larger samples.
    """

    try:
        # Remove NaN values
        clean_data = data.dropna()

        if len(clean_data) < 3:
            return np.nan, np.nan, "Insufficient data"

        if method == 'shapiro' and len(clean_data) <= 5000:
            statistic, p_value = stats.shapiro(clean_data)
            test_name = "Shapiro-Wilk"
        else:
            statistic, p_value = stats.kstest(clean_data, 'norm')
            test_name = "Kolmogorov-Smirnov"

        interpretation = f"Data is {'normally' if p_value > ALPHA else 'not normally'} distributed (p={p_value:.4f})"

        return statistic, p_value, interpretation

    except Exception as e:
        logger.error(f"Normality test error: {e}")
        return np.nan, np.nan, f"Error: {str(e)}"


def calculate_effect_size(group1: pd.Series, group2: pd.Series) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size.

    Parameters
    ----------
    group1, group2 : pd.Series
        Two groups to compare.

    Returns
    -------
    Tuple[float, str]
        (effect_size, interpretation).

    Notes
    -----
    Cohen's d interpretation:
    - Small: 0.2
    - Medium: 0.5
    - Large: 0.8
    """

    try:
        mean_diff = group1.mean() - group2.mean()
        pooled_std = np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2)

        if pooled_std == 0:
            return np.nan, "Cannot calculate (zero variance)"

        d = mean_diff / pooled_std

        if abs(d) < 0.2:
            interpretation = "Negligible"
        elif abs(d) < 0.5:
            interpretation = "Small"
        elif abs(d) < 0.8:
            interpretation = "Medium"
        else:
            interpretation = "Large"

        return d, interpretation

    except Exception as e:
        logger.error(f"Effect size calculation error: {e}")
        return np.nan, f"Error: {str(e)}"


def create_publication_ready_plot(fig: go.Figure, title: str = "") -> go.Figure:
    """
    Apply Nature journal styling to plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure object.
    title : str, optional
        Plot title (default: "").

    Returns
    -------
    go.Figure
        Styled figure ready for publication.

    Notes
    -----
    Follows Nature journal figure guidelines:
    - Arial/Helvetica font
    - Minimum 6pt font size
    - Clear axis labels
    - High contrast colors
    """

    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        title=dict(text=title, font=dict(size=14, color="#000000")),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#E0E0E0',
            linecolor='#333333',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E0E0E0',
            linecolor='#333333',
            linewidth=1
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#333333',
            borderwidth=1
        ),
        hovermode='closest'
    )

    return fig


# ============================================================================
# Main Page Rendering
# ============================================================================

def main():
    """Main function to render the Analysis Modules page."""

    # Render sidebar
    render_sidebar()

    # Render page header
    render_page_header(
        title="Exploratory Data Analysis",
        icon="🔬",
        subtitle="Comprehensive statistical analysis and visualization"
    )

    # Data validation
    if not DataManager.validate_dataset():
        render_no_data_warning()
        return

    # Get dataset
    df = st.session_state.current_dataset
    dataset_info = get_enhanced_dataset_info(df)

    # Display dataset summary
    render_dataset_summary(dataset_info)

    # Create analysis tabs
    render_analysis_tabs(df, dataset_info)

    # Record analysis session
    record_analysis_session(dataset_info)


def render_no_data_warning() -> None:
    """Display warning when no dataset is loaded."""

    render_info_box(
        content="""
        **No active dataset detected.**

        Please load a dataset in the Data Management Hub before proceeding with analysis.
        """,
        box_type="warning",
        title="Data Required"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📂 Go to Data Hub", use_container_width=True, type="primary"):
            st.switch_page("pages/2_📂_Data_Hub.py")

    with col2:
        if st.button("🔄 Refresh Data Status", use_container_width=True):
            st.rerun()


def render_dataset_summary(info: Dict[str, Any]) -> None:
    """
    Display comprehensive dataset summary.

    Parameters
    ----------
    info : Dict[str, Any]
        Enhanced dataset information dictionary.
    """

    st.success(f"**📊 Active Dataset:** `{info['name']}`")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Observations", f"{info['shape'][0]:,}")
    with col2:
        st.metric("Variables", info['shape'][1])
    with col3:
        st.metric("Numeric", len(info['numeric_columns']))
    with col4:
        st.metric("Categorical", len(info['categorical_columns']))
    with col5:
        st.metric("Missing", info['null_percentage'])

    # Additional details in expander
    with st.expander("📋 Detailed Dataset Information", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Memory Usage:**")
            st.write(info.get('memory_usage', 'Unknown'))

            st.markdown("**Duplicate Rows:**")
            st.write(f"{info.get('duplicate_rows', 0):,}")

        with col2:
            st.markdown("**Data Types:**")
            for dtype, count in info.get('data_types', {}).items():
                st.write(f"- {dtype}: {count} column(s)")


def render_analysis_tabs(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """
    Render main analysis tabs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset for analysis.
    info : Dict[str, Any]
        Dataset metadata.
    """

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Descriptive Statistics",
        "📈 Correlation Analysis",
        "🎨 Visualization",
        "🛠️ Data Preprocessing"
    ])

    with tab1:
        render_descriptive_statistics_tab(df, info)

    with tab2:
        render_correlation_analysis_tab(df, info)

    with tab3:
        render_visualization_tab(df, info)

    with tab4:
        render_preprocessing_tab(df, info)


# ============================================================================
# Tab 1: Descriptive Statistics
# ============================================================================

def render_descriptive_statistics_tab(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Render descriptive statistics analysis tab."""

    render_section_header("Descriptive Statistics", "📊")

    numeric_cols = info['numeric_columns']
    categorical_cols = info['categorical_columns']

    col1, col2 = st.columns([1, 2])

    with col1:
        render_data_quality_metrics(df, info)

    with col2:
        render_statistical_summaries(df, numeric_cols, categorical_cols)


def render_data_quality_metrics(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Display data quality metrics."""

    st.markdown("#### Data Quality Metrics")

    # Completeness
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    st.metric("Completeness", f"{completeness:.1f}%")

    # Duplicate rate
    dup_rate = (info.get('duplicate_rows', 0) / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    st.metric("Duplicate Rate", f"{dup_rate:.2f}%")

    # Unique rate for categorical columns
    if info['categorical_cols']:
        unique_rates = [df[col].nunique() / len(df) * 100 for col in info['categorical_columns']]
        avg_unique = np.mean(unique_rates)
        st.metric("Avg Categorical Uniqueness", f"{avg_unique:.1f}%")

    st.markdown("---")

    # Data type distribution
    st.markdown("#### Data Type Distribution")
    for dtype, count in info.get('data_types', {}).items():
        st.write(f"**{str(dtype)}:** {count} column(s)")


def render_statistical_summaries(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
    """Display statistical summaries for numeric and categorical variables."""

    st.markdown("#### Statistical Summaries")

    # Numeric summary
    if numeric_cols:
        st.markdown("##### Numeric Variables")

        summary_stats = df[numeric_cols].describe().T
        summary_stats['missing'] = df[numeric_cols].isnull().sum()
        summary_stats['skewness'] = df[numeric_cols].skew()
        summary_stats['kurtosis'] = df[numeric_cols].kurtosis()

        st.dataframe(
            summary_stats.style.format("{:.3f}"),
            use_container_width=True,
            height=min(400, len(numeric_cols) * 40 + 50)
        )

        # Normality tests
        if st.checkbox("Run normality tests", value=False):
            render_normality_tests(df, numeric_cols)
    else:
        st.info("No numeric variables available for statistical summary")

    # Categorical summary
    if categorical_cols:
        st.markdown("---")
        st.markdown("##### Categorical Variables")

        cat_summary = []
        for col in categorical_cols[:10]:  # Limit to first 10
            cat_summary.append({
                'Variable': col,
                'Unique Values': df[col].nunique(),
                'Most Frequent': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                'Frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
                'Missing': df[col].isnull().sum()
            })

        st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

        # Detailed frequency tables
        with st.expander("📊 Frequency Tables", expanded=False):
            selected_cat = st.selectbox("Select categorical variable", categorical_cols)
            if selected_cat:
                freq_table = df[selected_cat].value_counts().head(20)
                st.bar_chart(freq_table)
                st.dataframe(
                    pd.DataFrame({
                        'Value': freq_table.index,
                        'Count': freq_table.values,
                        'Percentage': (freq_table.values / len(df) * 100).round(2)
                    }),
                    use_container_width=True
                )


def render_normality_tests(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Perform and display normality tests."""

    st.markdown("##### Normality Tests")

    results = []
    for col in numeric_cols:
        stat, p_val, interp = test_normality(df[col])
        results.append({
            'Variable': col,
            'Statistic': f"{stat:.4f}" if not np.isnan(stat) else 'N/A',
            'P-value': f"{p_val:.4f}" if not np.isnan(p_val) else 'N/A',
            'Normal?': '✅' if p_val > ALPHA else '❌',
            'Interpretation': interp
        })

    st.dataframe(pd.DataFrame(results), use_container_width=True)

    render_info_box(
        content=f"Using α = {ALPHA}. Variables with p > {ALPHA} are considered normally distributed.",
        box_type="info"
    )


# ============================================================================
# Tab 2: Correlation Analysis
# ============================================================================

def render_correlation_analysis_tab(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Render correlation analysis tab."""

    render_section_header("Correlation Analysis", "📈")

    numeric_cols = info['numeric_columns']

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric variables for correlation analysis")
        return

    # Correlation method selection
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        corr_method = st.selectbox(
            "Correlation method",
            ["pearson", "spearman", "kendall"],
            help="Pearson: linear relationships | Spearman: monotonic relationships | Kendall: ordinal data"
        )

    with col2:
        show_values = st.checkbox("Show values", value=True)

    with col3:
        threshold = st.slider("Highlight |r| >", 0.0, 1.0, 0.7, 0.05)

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr(method=corr_method)

    # Visualization
    fig = create_correlation_heatmap(corr_matrix, show_values, threshold, corr_method)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation pairs analysis
    render_correlation_pairs(corr_matrix, threshold)


def create_correlation_heatmap(
        corr_matrix: pd.DataFrame,
        show_values: bool,
        threshold: float,
        method: str
) -> go.Figure:
    """
    Create publication-ready correlation heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    show_values : bool
        Whether to display correlation values.
    threshold : float
        Threshold for highlighting strong correlations.
    method : str
        Correlation method name.

    Returns
    -------
    go.Figure
        Styled heatmap figure.
    """

    fig = px.imshow(
        corr_matrix,
        color_continuous_scale=COLOR_SCHEMES['diverging'],
        aspect="auto",
        zmin=-1,
        zmax=1
    )

    if show_values:
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        text=f"{value:.2f}",
                        x=j,
                        y=i,
                        showarrow=False,
                        font=dict(
                            color='white' if abs(value) > threshold else 'black',
                            size=10
                        )
                    )
                )
        fig.update_layout(annotations=annotations)

    fig = create_publication_ready_plot(
        fig,
        f"Correlation Matrix ({method.capitalize()} Method)"
    )

    return fig


def render_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float) -> None:
    """Display detailed correlation pairs analysis."""

    st.markdown("---")
    render_section_header("Correlation Pairs", "🔗")

    # Extract correlation pairs
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j],
                'Absolute': abs(corr_matrix.iloc[i, j])
            })

    pairs_df = pd.DataFrame(pairs).sort_values('Absolute', ascending=False)

    # Filter by threshold
    strong_pairs = pairs_df[pairs_df['Absolute'] >= threshold]

    if len(strong_pairs) > 0:
        st.success(f"Found {len(strong_pairs)} strong correlations (|r| ≥ {threshold})")

        st.dataframe(
            strong_pairs[['Variable 1', 'Variable 2', 'Correlation']].style.format({'Correlation': '{:.3f}'}),
            use_container_width=True
        )
    else:
        st.info(f"No correlations found with |r| ≥ {threshold}")

    # Show all pairs in expander
    with st.expander("📋 All Correlation Pairs", expanded=False):
        st.dataframe(
            pairs_df[['Variable 1', 'Variable 2', 'Correlation']].style.format({'Correlation': '{:.3f}'}),
            use_container_width=True,
            height=400
        )


# ============================================================================
# Tab 3: Visualization
# ============================================================================

def render_visualization_tab(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Render interactive visualization tab."""

    render_section_header("Data Visualization", "🎨")

    numeric_cols = info['numeric_columns']
    categorical_cols = info['categorical_columns']

    if not numeric_cols:
        st.warning("No numeric variables available for visualization")
        return

    # Visualization type selection
    viz_type = st.selectbox(
        "Select visualization type",
        [
            "Distribution Plot",
            "Scatter Plot",
            "Box Plot",
            "Violin Plot",
            "Pair Plot",
            "Time Series"
        ]
    )

    if viz_type == "Distribution Plot":
        render_distribution_plot(df, numeric_cols)
    elif viz_type == "Scatter Plot":
        render_scatter_plot(df, numeric_cols, categorical_cols)
    elif viz_type == "Box Plot":
        render_box_plot(df, numeric_cols, categorical_cols)
    elif viz_type == "Violin Plot":
        render_violin_plot(df, numeric_cols, categorical_cols)
    elif viz_type == "Pair Plot":
        render_pair_plot(df, numeric_cols)
    elif viz_type == "Time Series":
        render_time_series_plot(df, numeric_cols, info.get('datetime_columns', []))


def render_distribution_plot(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Render distribution analysis plot."""

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_col = st.selectbox("Select variable", numeric_cols)

    with col2:
        bins = st.slider("Number of bins", 10, 100, 30)

    # Create histogram with marginal distribution
    fig = px.histogram(
        df,
        x=selected_col,
        nbins=bins,
        marginal="box",
        color_discrete_sequence=COLOR_SCHEMES['nature']
    )

    fig = create_publication_ready_plot(fig, f"Distribution of {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

    # Statistical summary
    col1, col2, col3, col4 = st.columns(4)

    data = df[selected_col].dropna()

    with col1:
        st.metric("Mean", f"{data.mean():.3f}")
    with col2:
        st.metric("Std Dev", f"{data.std():.3f}")
    with col3:
        st.metric("Skewness", f"{data.skew():.3f}")
    with col4:
        st.metric("Kurtosis", f"{data.kurtosis():.3f}")

    # Normality test
    stat, p_val, interp = test_normality(data)
    if not np.isnan(p_val):
        if p_val > ALPHA:
            st.success(f"✅ {interp}")
        else:
            st.warning(f"⚠️ {interp}")


def render_scatter_plot(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
    """Render scatter plot with regression analysis."""

    col1, col2, col3 = st.columns(3)

    with col1:
        x_col = st.selectbox("X axis", numeric_cols, key='scatter_x')
    with col2:
        y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key='scatter_y')
    with col3:
        color_col = st.selectbox("Color grouping", [None] + categorical_cols, key='scatter_color')

    # Add trendline option
    trendline = st.checkbox("Add trendline", value=len(df) < MAX_INTERACTIVE_POINTS)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline="ols" if trendline and len(df) < MAX_INTERACTIVE_POINTS else None,
        color_discrete_sequence=COLOR_SCHEMES['nature']
    )

    fig = create_publication_ready_plot(fig, f"{y_col} vs {x_col}")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    correlation = df[x_col].corr(df[y_col])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pearson Correlation", f"{correlation:.4f}")
    with col2:
        if abs(correlation) > 0.7:
            st.success("Strong correlation")
        elif abs(correlation) > 0.4:
            st.info("Moderate correlation")
        else:
            st.warning("Weak correlation")


def render_box_plot(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
    """Render box plot for group comparisons."""

    col1, col2 = st.columns(2)

    with col1:
        value_col = st.selectbox("Value variable", numeric_cols, key='box_value')
    with col2:
        group_col = st.selectbox("Group variable", [None] + categorical_cols, key='box_group')

    if group_col:
        fig = px.box(
            df,
            x=group_col,
            y=value_col,
            color=group_col,
            color_discrete_sequence=COLOR_SCHEMES['nature']
        )
    else:
        fig = px.box(df, y=value_col, color_discrete_sequence=COLOR_SCHEMES['nature'])

    fig = create_publication_ready_plot(
        fig,
        f"Distribution of {value_col}" + (f" by {group_col}" if group_col else "")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Statistical comparison if grouped
    if group_col and df[group_col].nunique() == 2:
        groups = df[group_col].unique()
        group1_data = df[df[group_col] == groups[0]][value_col].dropna()
        group2_data = df[df[group_col] == groups[1]][value_col].dropna()

        # Perform t-test
        t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
        effect_size, effect_interp = calculate_effect_size(group1_data, group2_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("T-statistic", f"{t_stat:.4f}")
        with col2:
            st.metric("P-value", f"{p_val:.4f}")
        with col3:
            st.metric("Effect Size (Cohen's d)", f"{effect_size:.3f} ({effect_interp})")

        if p_val < ALPHA:
            st.success(f"✅ Significant difference detected (p < {ALPHA})")
        else:
            st.info(f"No significant difference (p = {p_val:.4f})")


def render_violin_plot(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
    """Render violin plot for detailed distribution comparison."""

    col1, col2 = st.columns(2)

    with col1:
        value_col = st.selectbox("Value variable", numeric_cols, key='violin_value')
    with col2:
        group_col = st.selectbox("Group variable", categorical_cols if categorical_cols else [None], key='violin_group')

    if group_col:
        fig = px.violin(
            df,
            x=group_col,
            y=value_col,
            color=group_col,
            box=True,
            points="all",
            color_discrete_sequence=COLOR_SCHEMES['nature']
        )
    else:
        fig = px.violin(
            df,
            y=value_col,
            box=True,
            points="all",
            color_discrete_sequence=COLOR_SCHEMES['nature']
        )

    fig = create_publication_ready_plot(
        fig,
        f"Violin Plot: {value_col}" + (f" by {group_col}" if group_col else "")
    )
    st.plotly_chart(fig, use_container_width=True)

    render_info_box(
        content="Violin plots combine box plot and kernel density estimation, showing the full distribution shape.",
        box_type="info"
    )


def render_pair_plot(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Render pair plot for multivariate exploration."""

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric variables for pair plot")
        return

    # Variable selection
    max_vars = min(6, len(numeric_cols))
    selected_vars = st.multiselect(
        "Select variables (max 6 for performance)",
        numeric_cols,
        default=numeric_cols[:max_vars]
    )

    if len(selected_vars) < 2:
        st.warning("Please select at least 2 variables")
        return

    if len(selected_vars) > 6:
        st.warning("Too many variables selected. Using first 6.")
        selected_vars = selected_vars[:6]

    # Create scatter matrix
    fig = px.scatter_matrix(
        df[selected_vars],
        dimensions=selected_vars,
        color_discrete_sequence=COLOR_SCHEMES['nature']
    )

    fig.update_traces(diagonal_visible=False)
    fig = create_publication_ready_plot(fig, "Pair Plot")

    st.plotly_chart(fig, use_container_width=True)


def render_time_series_plot(df: pd.DataFrame, numeric_cols: List[str], datetime_cols: List[str]) -> None:
    """Render time series visualization."""

    if not datetime_cols:
        st.warning("No datetime columns detected for time series analysis")
        st.info("You can convert a column to datetime in the Preprocessing tab")
        return

    col1, col2 = st.columns(2)

    with col1:
        time_col = st.selectbox("Time variable", datetime_cols)
    with col2:
        value_col = st.selectbox("Value variable", numeric_cols, key='ts_value')

    # Sort by time
    df_sorted = df.sort_values(time_col)

    fig = px.line(
        df_sorted,
        x=time_col,
        y=value_col,
        color_discrete_sequence=COLOR_SCHEMES['nature']
    )

    fig = create_publication_ready_plot(fig, f"Time Series: {value_col}")
    st.plotly_chart(fig, use_container_width=True)

    # Add rolling statistics option
    if st.checkbox("Show rolling statistics", value=False):
        window = st.slider("Rolling window size", 2, 50, 7)

        fig_rolling = go.Figure()

        fig_rolling.add_trace(go.Scatter(
            x=df_sorted[time_col],
            y=df_sorted[value_col],
            name='Original',
            line=dict(color='lightgray')
        ))

        fig_rolling.add_trace(go.Scatter(
            x=df_sorted[time_col],
            y=df_sorted[value_col].rolling(window=window).mean(),
            name=f'Rolling Mean ({window})',
            line=dict(color=COLOR_SCHEMES['nature'][0])
        ))

        fig_rolling = create_publication_ready_plot(fig_rolling, f"Rolling Statistics: {value_col}")
        st.plotly_chart(fig_rolling, use_container_width=True)


# ============================================================================
# Tab 4: Data Preprocessing
# ============================================================================

def render_preprocessing_tab(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Render data preprocessing interface."""

    render_section_header("Data Preprocessing", "🛠️")

    st.markdown("""
    Apply transformations to prepare data for analysis. Changes create a new dataset version.
    """)

    # Preprocessing options
    preprocessing_option = st.selectbox(
        "Select preprocessing operation",
        [
            "Handle Missing Values",
            "Remove Duplicates",
            "Type Conversion",
            "Outlier Detection",
            "Feature Scaling",
            "Feature Engineering"
        ]
    )

    if preprocessing_option == "Handle Missing Values":
        render_missing_values_handler(df, info)
    elif preprocessing_option == "Remove Duplicates":
        render_duplicate_remover(df, info)
    elif preprocessing_option == "Type Conversion":
        render_type_converter(df)
    elif preprocessing_option == "Outlier Detection":
        render_outlier_detector(df, info)
    elif preprocessing_option == "Feature Scaling":
        render_feature_scaler(df, info)
    elif preprocessing_option == "Feature Engineering":
        render_feature_engineer(df, info)


def render_missing_values_handler(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Handle missing values in dataset."""

    st.markdown("### Missing Values Treatment")

    # Display missing values summary
    null_summary = df.isnull().sum()
    null_cols = null_summary[null_summary > 0]

    col1, col2 = st.columns(2)

    with col1:
        if len(null_cols) > 0:
            st.warning(f"Found {len(null_cols)} columns with missing values")

            missing_df = pd.DataFrame({
                'Column': null_cols.index,
                'Missing Count': null_cols.values,
                'Missing %': (null_cols.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values detected")

    with col2:
        if len(null_cols) > 0:
            treatment_method = st.selectbox(
                "Treatment method",
                [
                    "Drop rows with any missing",
                    "Drop rows with all missing",
                    "Fill numeric with mean",
                    "Fill numeric with median",
                    "Forward fill",
                    "Backward fill",
                    "Fill with constant"
                ]
            )

            if treatment_method == "Fill with constant":
                fill_value = st.text_input("Fill value", "0")

            if st.button("Apply Treatment", type="primary", use_container_width=True):
                df_treated = apply_missing_value_treatment(df, treatment_method,
                                                           fill_value if treatment_method == "Fill with constant" else None)

                if df_treated is not None:
                    new_name = f"{info['name']}_missing_handled"
                    DataManager.set_active_dataset(df_treated, new_name)
                    st.success(f"✅ Treatment applied! New dataset: {new_name}")
                    st.rerun()


def apply_missing_value_treatment(df: pd.DataFrame, method: str, fill_value: Optional[str] = None) -> Optional[
    pd.DataFrame]:
    """Apply missing value treatment method."""

    try:
        df_clean = df.copy()

        if method == "Drop rows with any missing":
            df_clean = df_clean.dropna()
        elif method == "Drop rows with all missing":
            df_clean = df_clean.dropna(how='all')
        elif method == "Fill numeric with mean":
            numeric_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif method == "Fill numeric with median":
            numeric_cols = df_clean.select_dtypes(include=np.number).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif method == "Forward fill":
            df_clean = df_clean.fillna(method='ffill')
        elif method == "Backward fill":
            df_clean = df_clean.fillna(method='bfill')
        elif method == "Fill with constant":
            df_clean = df_clean.fillna(fill_value)

        return df_clean

    except Exception as e:
        logger.error(f"Missing value treatment error: {e}")
        st.error(f"Treatment failed: {str(e)}")
        return None


def render_duplicate_remover(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Remove duplicate rows from dataset."""

    st.markdown("### Duplicate Rows Removal")

    duplicate_count = df.duplicated().sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Duplicate Rows", duplicate_count)
        st.metric("Percentage", f"{(duplicate_count / len(df) * 100):.2f}%")

    with col2:
        if duplicate_count > 0:
            keep_option = st.radio(
                "Keep which duplicate?",
                ["first", "last", "none"]
            )

            if st.button("Remove Duplicates", type="primary", use_container_width=True):
                df_dedup = df.drop_duplicates(keep=keep_option if keep_option != "none" else False)

                new_name = f"{info['name']}_dedup"
                DataManager.set_active_dataset(df_dedup, new_name)
                st.success(f"✅ Removed {duplicate_count} duplicates!")
                st.rerun()
        else:
            st.success("✅ No duplicates found")


def render_type_converter(df: pd.DataFrame) -> None:
    """Convert column data types."""

    st.markdown("### Data Type Conversion")

    col1, col2 = st.columns(2)

    with col1:
        column_to_convert = st.selectbox("Select column", df.columns)
        current_dtype = str(df[column_to_convert].dtype)
        st.info(f"Current type: `{current_dtype}`")

    with col2:
        target_dtype = st.selectbox(
            "Target type",
            ["numeric", "string", "category", "datetime", "boolean"]
        )

        if target_dtype == "datetime":
            date_format = st.text_input("Date format (optional)", placeholder="%Y-%m-%d")

        if st.button("Convert Type", use_container_width=True):
            df_converted = convert_column_type(
                df,
                column_to_convert,
                target_dtype,
                date_format if target_dtype == "datetime" else None
            )

            if df_converted is not None:
                DataManager.set_active_dataset(df_converted, f"{DataManager.get_dataset_info()['name']}_converted")
                st.success("✅ Type conversion successful!")
                st.rerun()


def convert_column_type(df: pd.DataFrame, column: str, target_type: str, date_format: Optional[str] = None) -> Optional[
    pd.DataFrame]:
    """Convert column to target data type."""

    try:
        df_converted = df.copy()

        if target_type == "numeric":
            df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
        elif target_type == "string":
            df_converted[column] = df_converted[column].astype(str)
        elif target_type == "category":
            df_converted[column] = df_converted[column].astype('category')
        elif target_type == "datetime":
            if date_format:
                df_converted[column] = pd.to_datetime(df_converted[column], format=date_format, errors='coerce')
            else:
                df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
        elif target_type == "boolean":
            df_converted[column] = df_converted[column].astype(bool)

        return df_converted

    except Exception as e:
        logger.error(f"Type conversion error: {e}")
        st.error(f"Conversion failed: {str(e)}")
        return None


def render_outlier_detector(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Detect and handle outliers."""

    st.markdown("### Outlier Detection")

    numeric_cols = info['numeric_columns']

    if not numeric_cols:
        st.warning("No numeric columns for outlier detection")
        return

    selected_col = st.selectbox("Select variable", numeric_cols)
    method = st.radio("Detection method", ["IQR", "Z-score"])

    if method == "IQR":
        threshold = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
        outliers = detect_outliers_iqr(df[selected_col], threshold)
    else:
        threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1)
        outliers = detect_outliers_zscore(df[selected_col], threshold)

    outlier_count = outliers.sum()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Outliers Detected", outlier_count)
        st.metric("Percentage", f"{(outlier_count / len(df) * 100):.2f}%")

    with col2:
        # Visualize outliers
        fig = px.box(df, y=selected_col, points="all")
        fig = create_publication_ready_plot(fig, f"Outliers in {selected_col}")
        st.plotly_chart(fig, use_container_width=True, key='outlier_box')

    if outlier_count > 0:
        action = st.radio("Action", ["Remove outliers", "Cap outliers", "View only"])

        if action != "View only" and st.button("Apply Action", type="primary"):
            if action == "Remove outliers":
                df_processed = df[~outliers]
            else:  # Cap outliers
                df_processed = df.copy()
                if method == "IQR":
                    Q1 = df_processed[selected_col].quantile(0.25)
                    Q3 = df_processed[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                    df_processed[selected_col] = df_processed[selected_col].clip(lower, upper)

            DataManager.set_active_dataset(df_processed, f"{info['name']}_outliers_handled")
            st.success("✅ Outliers handled!")
            st.rerun()


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (data < lower) | (data > upper)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(data.dropna()))
    outliers = pd.Series(False, index=data.index)
    outliers[data.notna()] = z_scores > threshold
    return outliers


def render_feature_scaler(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Scale numeric features."""

    st.markdown("### Feature Scaling")

    numeric_cols = info['numeric_columns']

    if not numeric_cols:
        st.warning("No numeric columns to scale")
        return

    selected_cols = st.multiselect("Select columns to scale", numeric_cols,
                                   default=numeric_cols[:min(5, len(numeric_cols))])

    if not selected_cols:
        return

    scaling_method = st.selectbox(
        "Scaling method",
        ["Standardization (Z-score)", "Min-Max Normalization", "Robust Scaling"]
    )

    if st.button("Apply Scaling", type="primary"):
        df_scaled = scale_features(df, selected_cols, scaling_method)

        if df_scaled is not None:
            DataManager.set_active_dataset(df_scaled, f"{info['name']}_scaled")
            st.success("✅ Features scaled successfully!")
            st.rerun()


def scale_features(df: pd.DataFrame, columns: List[str], method: str) -> Optional[pd.DataFrame]:
    """Scale selected features."""

    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        df_scaled = df.copy()

        if method == "Standardization (Z-score)":
            scaler = StandardScaler()
        elif method == "Min-Max Normalization":
            scaler = MinMaxScaler()
        else:  # Robust Scaling
            scaler = RobustScaler()

        df_scaled[columns] = scaler.fit_transform(df[columns])

        return df_scaled

    except Exception as e:
        logger.error(f"Feature scaling error: {e}")
        st.error(f"Scaling failed: {str(e)}")
        return None


def render_feature_engineer(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    """Create new features from existing ones."""

    st.markdown("### Feature Engineering")

    st.info("Create new features by combining or transforming existing variables")

    # Feature creation options
    operation = st.selectbox(
        "Select operation",
        ["Arithmetic", "Logarithm", "Square Root", "Polynomial", "Binning"]
    )

    # Implementation would continue with specific feature engineering operations
    st.info("Feature engineering operations coming soon...")


# ============================================================================
# Session Recording
# ============================================================================

def record_analysis_session(info: Dict[str, Any]) -> None:
    """Record analysis session to session state."""

    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    session_record = {
        'timestamp': datetime.now().isoformat(),
        'dataset': info['name'],
        'analysis_type': 'Exploratory Data Analysis',
        'summary': {
            'observations': info['shape'][0],
            'variables': info['shape'][1],
            'numeric_vars': len(info['numeric_columns']),
            'categorical_vars': len(info['categorical_columns']),
            'missing_percentage': info['null_percentage']
        }
    }

    st.session_state.analysis_history.append(session_record)


# ============================================================================
# Custom Styling
# ============================================================================

def apply_custom_styles() -> None:
    """Apply custom CSS for enhanced UI."""

    st.markdown("""
    <style>
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2E86AB;
        }

        /* Buttons */
        .stButton > button {
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
        }

        /* Selectbox */
        .stSelectbox > div > div {
            background-color: #f8f9fa;
        }

        /* Dataframes */
        .dataframe {
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    apply_custom_styles()
    main()
