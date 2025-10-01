# pages/8_📈_Results.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import base64

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
    from utils.data_loader import DataLoader
    from utils.data_manager import DataManager
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box
    from config.data_config import DataConfig
except ImportError as e:
    logger.error(f"Module import failed: {e}")
    st.error(f"⚠️ Critical module import error: {e}")
    st.stop()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Results Dashboard - SYPHU iGEM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Constants
# ============================================================================

RESULT_TYPES = {
    'data_analysis': {
        'icon': '📊',
        'name': 'Statistical Analysis',
        'session_key': 'current_data_analysis'
    },
    'ml_results': {
        'icon': '🤖',
        'name': 'Machine Learning',
        'session_key': 'ml_results'
    },
    'image_analysis': {
        'icon': '🖼️',
        'name': 'Image Analysis',
        'session_key': 'analyzed_images'
    },
    'bioinformatics': {
        'icon': '🧬',
        'name': 'Bioinformatics',
        'session_key': 'bioinformatics_results'
    },
    'experiments': {
        'icon': '🧪',
        'name': 'Experiments',
        'session_key': 'experiment_records'
    }
}


# ============================================================================
# Utility Functions
# ============================================================================

def check_available_results() -> Dict[str, int]:
    """
    Check which analysis results are available.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping result type to count.
    """
    available = {}

    for result_type, config in RESULT_TYPES.items():
        session_key = config['session_key']
        data = st.session_state.get(session_key)

        if data:
            if isinstance(data, dict):
                count = len(data)
            elif isinstance(data, list):
                count = len(data)
            else:
                count = 1
            available[result_type] = count
        else:
            available[result_type] = 0

    return available


def export_results_to_json() -> str:
    """
    Export all results to JSON format.

    Returns
    -------
    str
        JSON string of all results.
    """
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'platform_version': '2.1.0',
        'results': {}
    }

    for result_type, config in RESULT_TYPES.items():
        session_key = config['session_key']
        data = st.session_state.get(session_key)

        if data:
            # Convert to JSON-serializable format
            try:
                export_data['results'][result_type] = json.loads(
                    json.dumps(data, default=str)
                )
            except:
                export_data['results'][result_type] = str(data)

    return json.dumps(export_data, indent=2)


def create_summary_report() -> pd.DataFrame:
    """
    Create summary report of all analyses.

    Returns
    -------
    pd.DataFrame
        Summary statistics DataFrame.
    """
    summary_data = []

    for result_type, config in RESULT_TYPES.items():
        session_key = config['session_key']
        data = st.session_state.get(session_key)

        if data:
            count = len(data) if isinstance(data, (dict, list)) else 1

            summary_data.append({
                'Analysis Type': config['name'],
                'Results Count': count,
                'Status': '✅ Available',
                'Last Modified': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
        else:
            summary_data.append({
                'Analysis Type': config['name'],
                'Results Count': 0,
                'Status': '⚪ No Results',
                'Last Modified': 'N/A'
            })

    return pd.DataFrame(summary_data)


# ============================================================================
# Main Page Rendering
# ============================================================================

def main():
    """Main function to render Results Dashboard."""

    render_sidebar()

    render_page_header(
        title="Results Dashboard",
        icon="📈",
        subtitle="Comprehensive analysis results and visualization"
    )

    # Check available results
    available_results = check_available_results()
    total_results = sum(available_results.values())

    if total_results == 0:
        render_no_results_page()
    else:
        render_results_dashboard(available_results)


def render_no_results_page():
    """Display page when no results are available."""

    render_info_box(
        content="""
        **No analysis results available yet.**

        Get started by:
        1. Loading a dataset in the Data Management Hub
        2. Running statistical analyses in the Analysis Modules
        3. Training machine learning models
        4. Performing image or bioinformatics analysis

        All results will automatically appear here for review and export.
        """,
        box_type="info",
        title="Get Started"
    )

    st.markdown("---")
    st.markdown("### Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📂 Data Management", use_container_width=True, type="primary"):
            st.switch_page("pages/2_📂_Data_Hub.py")

    with col2:
        if st.button("🔬 Statistical Analysis", use_container_width=True):
            st.switch_page("pages/3_🔬_Analysis_Modules.py")

    with col3:
        if st.button("🤖 Machine Learning", use_container_width=True):
            st.switch_page("pages/5_🤖_Machine_Learning.py")


def render_results_dashboard(available_results: Dict[str, int]):
    """
    Render main results dashboard.

    Parameters
    ----------
    available_results : Dict[str, int]
        Dictionary of available result counts.
    """

    # Overview metrics
    render_overview_metrics(available_results)

    # Active dataset info
    render_active_dataset_info()

    # Results tabs
    render_results_tabs(available_results)


def render_overview_metrics(available_results: Dict[str, int]):
    """Display overview metrics for all result types."""

    st.markdown("### Analysis Overview")

    cols = st.columns(len(RESULT_TYPES))

    for idx, (result_type, count) in enumerate(available_results.items()):
        config = RESULT_TYPES[result_type]

        with cols[idx]:
            delta = "Available" if count > 0 else "None"
            delta_color = "normal" if count > 0 else "off"

            st.metric(
                label=f"{config['icon']} {config['name']}",
                value=count,
                delta=delta,
                delta_color=delta_color
            )

    st.markdown("---")


def render_active_dataset_info():
    """Display information about active dataset."""

    if DataManager.validate_dataset():
        dataset_info = DataManager.get_dataset_info()

        st.success(f"**📊 Active Dataset:** `{dataset_info['name']}`")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Observations", f"{dataset_info['shape'][0]:,}")
        with col2:
            st.metric("Variables", dataset_info['shape'][1])
        with col3:
            st.metric("Memory Usage", dataset_info['memory_usage'])
        with col4:
            st.metric("Data Type", "Active")

        st.markdown("---")


def render_results_tabs(available_results: Dict[str, int]):
    """
    Render tabs for different result types.

    Parameters
    ----------
    available_results : Dict[str, int]
        Dictionary of available result counts.
    """

    # Create tabs
    tab_names = [
        "📊 Summary",
        "🔬 Statistical",
        "🤖 ML",
        "🖼️ Image",
        "🧬 Bioinformatics",
        "🧪 Experiments"
    ]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_summary_tab(available_results)

    with tabs[1]:
        render_statistical_results_tab()

    with tabs[2]:
        render_ml_results_tab()

    with tabs[3]:
        render_image_results_tab()

    with tabs[4]:
        render_bioinformatics_results_tab()

    with tabs[5]:
        render_experiments_tab()


# ============================================================================
# Tab 1: Summary
# ============================================================================

def render_summary_tab(available_results: Dict[str, int]):
    """Render summary tab with overall statistics."""

    render_section_header("Analysis Summary", "📊")

    # Summary table
    summary_df = create_summary_report()
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Visualization
    st.markdown("---")
    st.markdown("### Results Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig_bar = px.bar(
            x=[RESULT_TYPES[k]['name'] for k in available_results.keys()],
            y=list(available_results.values()),
            title="Results by Analysis Type",
            labels={'x': 'Analysis Type', 'y': 'Count'},
            color=list(available_results.values()),
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Pie chart
        non_zero = {k: v for k, v in available_results.items() if v > 0}
        if non_zero:
            fig_pie = px.pie(
                values=list(non_zero.values()),
                names=[RESULT_TYPES[k]['name'] for k in non_zero.keys()],
                title="Results Composition"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Export options
    st.markdown("---")
    st.markdown("### Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export to JSON
        json_data = export_results_to_json()
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"results_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        # Export summary to CSV
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary CSV",
            data=csv,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col3:
        if st.button("🔄 Refresh Results", use_container_width=True):
            st.rerun()


# ============================================================================
# Tab 2: Statistical Results
# ============================================================================

def render_statistical_results_tab():
    """Render statistical analysis results."""

    render_section_header("Statistical Analysis Results", "🔬")

    analysis_data = st.session_state.get('current_data_analysis')

    if not analysis_data:
        st.info("No statistical analysis results available. Run analyses in the Analysis Modules.")
        return

    st.success(f"Found {len(analysis_data)} analysis session(s)")

    # Display each analysis
    for idx, (key, analysis) in enumerate(analysis_data.items(), 1):
        with st.expander(f"Analysis {idx}: {analysis.get('dataset', 'Unknown')} - {analysis.get('timestamp', 'N/A')}",
                         expanded=False):

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Analysis Information:**")
                st.write(f"- Type: {analysis.get('analysis_type', 'Unknown')}")
                st.write(f"- Dataset: {analysis.get('dataset', 'Unknown')}")
                st.write(f"- Timestamp: {analysis.get('timestamp', 'N/A')}")

            with col2:
                st.markdown("**Summary Statistics:**")
                summary = analysis.get('summary', {})
                for key, value in summary.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")


# ============================================================================
# Tab 3: ML Results
# ============================================================================

def render_ml_results_tab():
    """Render machine learning results."""

    render_section_header("Machine Learning Results", "🤖")

    ml_results = st.session_state.get('ml_results')

    if not ml_results:
        st.info("No machine learning results available. Train models in the ML module.")
        return

    # Display results by type
    for analysis_type, results in ml_results.items():
        st.markdown(f"### {analysis_type.replace('_', ' ').title()}")

        with st.expander("View Details", expanded=True):
            if analysis_type == 'clustering':
                render_clustering_results(results)
            elif analysis_type == 'dimensionality_reduction':
                render_dimred_results(results)
            elif analysis_type == 'supervised_learning':
                render_supervised_results(results)


def render_clustering_results(results: Dict[str, Any]):
    """Display clustering results."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration:**")
        st.write(f"- Method: {results.get('method', 'Unknown')}")
        st.write(f"- Features: {', '.join(results.get('features', []))}")

    with col2:
        st.markdown("**Performance:**")
        metrics = results.get('metrics', {})
        st.write(f"- Clusters: {metrics.get('n_clusters', 'N/A')}")
        st.write(f"- Silhouette Score: {metrics.get('silhouette_score', 0):.3f}")


def render_dimred_results(results: Dict[str, Any]):
    """Display dimensionality reduction results."""
    st.write(f"**Method:** {results.get('method', 'Unknown')}")
    st.write(f"**Components:** {results.get('parameters', {}).get('n_components', 'N/A')}")


def render_supervised_results(results: Dict[str, Any]):
    """Display supervised learning results."""
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Type:** {results.get('problem_type', 'Unknown')}")
        st.write(f"**Target:** {results.get('target', 'Unknown')}")

    with col2:
        if results.get('problem_type') == 'Classification':
            st.write(f"**Test Accuracy:** {results.get('test_accuracy', 0):.3f}")
        else:
            st.write(f"**Test R²:** {results.get('test_r2', 0):.3f}")


# ============================================================================
# Tab 4: Image Analysis Results
# ============================================================================

def render_image_results_tab():
    """Render image analysis results."""

    render_section_header("Image Analysis Results", "🖼️")

    image_results = st.session_state.get('analyzed_images')

    if not image_results:
        st.info("No image analysis results available.")
        return

    st.success(f"Found {len(image_results)} analyzed image(s)")

    for image_path, analysis in image_results.items():
        with st.expander(f"📷 {analysis.get('file_name', 'Unknown')}", expanded=False):

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Analysis Info:**")
                st.write(f"- Model: {analysis.get('model', 'Unknown')}")
                st.write(f"- Timestamp: {analysis.get('timestamp', 'N/A')}")

            with col2:
                if 'image_statistics' in analysis:
                    stats = analysis['image_statistics']
                    st.markdown("**Image Statistics:**")
                    st.write(f"- Dimensions: {stats.get('width', 0)} × {stats.get('height', 0)}")
                    st.write(f"- Channels: {stats.get('channels', 0)}")


# ============================================================================
# Tab 5: Bioinformatics Results
# ============================================================================

def render_bioinformatics_results_tab():
    """Render bioinformatics analysis results."""

    render_section_header("Bioinformatics Results", "🧬")

    bio_results = st.session_state.get('bioinformatics_results')

    if not bio_results:
        st.info("No bioinformatics results available.")
        return

    for result in bio_results:
        analysis_type = result.get('analysis_type', 'Unknown')

        with st.expander(f"{analysis_type} - {result.get('timestamp', 'N/A')}", expanded=False):
            if analysis_type == 'Gene Enrichment':
                st.write(f"**Gene Count:** {result.get('gene_count', 0)}")
                st.write(f"**Database:** {result.get('database', 'Unknown')}")
                st.write(f"**Significant Pathways:** {result.get('significant_pathways', 0)}")


# ============================================================================
# Tab 6: Experiments
# ============================================================================

def render_experiments_tab():
    """Render experiment records."""

    render_section_header("Experiment Records", "🧪")

    experiments = st.session_state.get('experiment_records')

    if not experiments:
        st.info("No experiment records available.")
        return

    # Create summary table
    exp_data = []
    for exp_id, exp in experiments.items():
        exp_data.append({
            'ID': exp_id,
            'Name': exp.get('name', 'Unknown'),
            'Type': exp.get('type', 'Unknown'),
            'Status': exp.get('status', 'Unknown'),
            'Researcher': exp.get('researcher', 'Unknown')
        })

    df_exp = pd.DataFrame(exp_data)
    st.dataframe(df_exp, use_container_width=True, hide_index=True)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
