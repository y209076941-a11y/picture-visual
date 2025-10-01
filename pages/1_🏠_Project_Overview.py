# pages/1_🏠_Project_Overview.py

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

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

# Import custom components with error handling
try:
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box
    from utils.data_loader import DataLoader
    from config.data_config import DataConfig
except ImportError as e:
    logger.error(f"Module import failed: {e}")
    st.error(f"⚠️ Critical module import error: {e}")
    st.info("Please ensure all required modules are installed and accessible.")
    st.stop()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Project Overview - SYPHU iGEM",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/syphu-china/igem-platform',
        'Report a bug': 'https://github.com/syphu-china/igem-platform/issues',
        'About': 'SYPHU-CHINA iGEM 2024 Integrated Research Platform'
    }
)

# ============================================================================
# Platform Configuration Constants
# ============================================================================

PLATFORM_INFO = {
    "version": "2.1.0",
    "release_date": "2025-10",
    "team": "SYPHU-CHINA iGEM 2024",
    "license": "MIT"
}

PLATFORM_FEATURES = {
    "Data Management": {
        "icon": "📂",
        "description": "Local data auto-detection and Google Drive integration",
        "capabilities": ["CSV/Excel import", "Image datasets", "Sequence files"]
    },
    "Statistical Analysis": {
        "icon": "📊",
        "description": "Comprehensive statistical testing and visualization",
        "capabilities": ["Descriptive statistics", "Hypothesis testing", "Correlation analysis"]
    },
    "Machine Learning": {
        "icon": "🤖",
        "description": "Automated ML workflows for biological data",
        "capabilities": ["Clustering", "Dimensionality reduction", "Classification"]
    },
    "Image Analysis": {
        "icon": "🖼️",
        "description": "AI-powered microscopy image processing",
        "capabilities": ["Cell detection", "Morphology analysis", "Tracking"]
    },
    "Bioinformatics": {
        "icon": "🧬",
        "description": "Genomic and proteomic analysis tools",
        "capabilities": ["Enrichment analysis", "Sequence alignment", "Pathway mapping"]
    }
}

TECHNOLOGY_STACK = {
    "Frontend": ["Streamlit", "Plotly", "Altair"],
    "Data Processing": ["Pandas", "NumPy", "Polars"],
    "Machine Learning": ["Scikit-learn", "TensorFlow", "PyTorch"],
    "Bioinformatics": ["Biopython", "Scanpy", "GSEApy"],
    "Visualization": ["Matplotlib", "Seaborn", "Plotly"]
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_system_status() -> Dict[str, any]:
    """
    Retrieve current system and data status.

    Returns
    -------
    Dict[str, any]
        Dictionary containing data availability, file count, and status flags.

    Examples
    --------
    >>> status = get_system_status()
    >>> print(status['has_data'])
    True
    """

    try:
        data_root = DataConfig.get_active_local_root()
        has_data, file_count = DataLoader.check_data_availability(str(data_root))

        return {
            "has_data": has_data,
            "file_count": file_count,
            "data_root": data_root,
            "status_ok": True
        }
    except Exception as e:
        logger.error(f"Error retrieving system status: {e}")
        return {
            "has_data": False,
            "file_count": 0,
            "data_root": None,
            "status_ok": False
        }


def render_platform_metrics(status: Dict[str, any]) -> None:
    """
    Display platform capability metrics in a four-column layout.

    Parameters
    ----------
    status : Dict[str, any]
        System status dictionary from get_system_status().

    Notes
    -----
    Metrics are displayed using Streamlit's metric component for
    clear visual representation of platform capabilities.
    """

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Core Modules",
            value="9",
            delta="Functional",
            help="Number of integrated analysis modules"
        )

    with col2:
        supported_formats = len(DataConfig.get_supported_extensions_list())
        st.metric(
            label="File Formats",
            value=f"{supported_formats}+",
            delta="Supported",
            help="Total supported data file formats"
        )

    with col3:
        st.metric(
            label="Analysis Methods",
            value="25+",
            delta="Available",
            help="Statistical and ML analysis methods"
        )

    with col4:
        data_status = "Ready" if status["has_data"] else "Setup Required"
        delta_color = "normal" if status["has_data"] else "off"
        st.metric(
            label="Data Status",
            value=data_status,
            delta=f"{status['file_count']} files",
            delta_color=delta_color,
            help="Current data availability status"
        )


def render_feature_cards() -> None:
    """
    Display platform features as interactive information cards.

    Notes
    -----
    Uses Streamlit expanders for organized presentation of features
    and their capabilities following Nature journal web design patterns.
    """

    render_section_header("Platform Capabilities", "🎯")

    for feature_name, feature_info in PLATFORM_FEATURES.items():
        with st.expander(f"{feature_info['icon']} {feature_name}", expanded=False):
            st.markdown(f"**{feature_info['description']}**")
            st.markdown("##### Key Capabilities:")
            for capability in feature_info['capabilities']:
                st.markdown(f"- {capability}")


def render_quick_start_guide() -> None:
    """
    Display step-by-step quick start guide for new users.

    Notes
    -----
    Provides clear onboarding instructions following best practices
    for scientific software documentation.
    """

    render_section_header("Quick Start Guide", "🚀")

    steps = [
        {
            "number": "1",
            "title": "Data Preparation",
            "content": "Navigate to **Data Management** to load local datasets or download shared data from Google Drive.",
            "icon": "📂"
        },
        {
            "number": "2",
            "title": "Select Analysis Module",
            "content": "Choose the appropriate analysis module based on your research objectives and data type.",
            "icon": "🔬"
        },
        {
            "number": "3",
            "title": "Configure Parameters",
            "content": "Set analysis parameters and configure visualization options according to your needs.",
            "icon": "⚙️"
        },
        {
            "number": "4",
            "title": "View Results",
            "content": "Access comprehensive analysis reports, visualizations, and exportable results in the Results page.",
            "icon": "📈"
        }
    ]

    for step in steps:
        st.markdown(
            f"""
            <div style='
                background: #f8f9fa;
                border-left: 4px solid #2E86AB;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 4px;
            '>
                <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;'>
                    <span style='
                        background: #2E86AB;
                        color: white;
                        width: 2rem;
                        height: 2rem;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                    '>{step['number']}</span>
                    <span style='font-size: 1.5rem;'>{step['icon']}</span>
                    <h4 style='margin: 0; color: #2E86AB;'>{step['title']}</h4>
                </div>
                <p style='margin: 0 0 0 3.5rem; color: #666;'>{step['content']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_data_source_info(status: Dict[str, any]) -> None:
    """
    Display information about data source configuration.

    Parameters
    ----------
    status : Dict[str, any]
        System status dictionary containing data root information.

    Notes
    -----
    Informs users about both local and cloud data access options,
    facilitating collaborative research workflows.
    """

    render_section_header("Data Source Configuration", "🌐")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### 📍 Local Data Access

            The platform automatically scans designated project directories
            for available datasets.

            **Features:**
            - Automatic directory scanning
            - Multiple file format support
            - Real-time status monitoring

            **Current Root:**
            """
        )
        if status["data_root"]:
            st.code(str(status["data_root"]), language="bash")
        else:
            st.warning("No active data root configured")

    with col2:
        st.markdown(
            """
            #### ☁️ Cloud Data Integration

            Access shared datasets through Google Drive for collaborative
            research and reproducible workflows.

            **Features:**
            - Public dataset repository
            - Version-controlled data
            - Cross-platform accessibility

            **Available Datasets:**
            """
        )
        dataset_count = len(DataConfig.GOOGLE_DRIVE_DATASETS)
        st.info(f"{dataset_count} shared datasets available in Data Management")


def render_technology_stack() -> None:
    """
    Display the technology stack powering the platform.

    Notes
    -----
    Provides transparency about dependencies and technologies,
    important for reproducibility and citation purposes.
    """

    render_section_header("Technology Stack", "🛠️")

    tabs = st.tabs(list(TECHNOLOGY_STACK.keys()))

    for idx, (category, technologies) in enumerate(TECHNOLOGY_STACK.items()):
        with tabs[idx]:
            st.markdown(f"##### {category} Technologies")
            cols = st.columns(3)
            for i, tech in enumerate(technologies):
                with cols[i % 3]:
                    st.markdown(f"- {tech}")


# ============================================================================
# Main Page Rendering
# ============================================================================

def main():
    """
    Main function to render the Project Overview page.

    Orchestrates all page components in a logical flow, from header
    to footer, providing comprehensive platform introduction.
    """

    # Render sidebar navigation
    render_sidebar()

    # Render page header
    render_page_header(
        title="Project Overview",
        icon="🏠",
        subtitle="SYPHU-CHINA iGEM 2024 Integrated Research Platform"
    )

    # Get current system status
    status = get_system_status()

    # Main content layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Platform introduction
        st.markdown(
            """
            ## Welcome to the SYPHU iGEM Research Platform

            This integrated analysis platform is designed specifically for biological
            research workflows, providing end-to-end data management, analysis, and
            visualization capabilities.

            ### Platform Philosophy

            Our platform follows FAIR data principles (Findable, Accessible, Interoperable,
            Reusable) and emphasizes:

            - **Reproducibility**: All analyses are fully documented and reproducible
            - **Accessibility**: Intuitive interface for researchers of all skill levels
            - **Integration**: Seamless workflow from data to publication-ready figures
            - **Flexibility**: Modular design supports diverse research needs
            """
        )

    with col2:
        # System status card
        if status["status_ok"]:
            if status["has_data"]:
                render_info_box(
                    content=f"System operational with {status['file_count']} data files available.",
                    box_type="success",
                    title="System Status"
                )
            else:
                render_info_box(
                    content="System operational. Upload data in Data Management to begin analysis.",
                    box_type="warning",
                    title="System Status"
                )
        else:
            render_info_box(
                content="System check failed. Please review configuration.",
                box_type="error",
                title="System Status"
            )

        # Version information
        st.markdown("---")
        render_info_box(
            content=f"""
            **Version**: {PLATFORM_INFO['version']}  
            **Release**: {PLATFORM_INFO['release_date']}  
            **Team**: {PLATFORM_INFO['team']}  
            **License**: {PLATFORM_INFO['license']}
            """,
            box_type="info",
            title="Platform Information"
        )

    # Platform metrics
    st.markdown("---")
    render_platform_metrics(status)

    # Feature cards
    st.markdown("---")
    render_feature_cards()

    # Quick start guide
    st.markdown("---")
    render_quick_start_guide()

    # Data source information
    st.markdown("---")
    render_data_source_info(status)

    # Technology stack
    st.markdown("---")
    render_technology_stack()

    # Footer with additional resources
    st.markdown("---")
    render_section_header("Additional Resources", "📚")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            #### Documentation
            - User Guide
            - API Reference
            - Tutorial Videos
            """
        )

    with col2:
        st.markdown(
            """
            #### Support
            - GitHub Issues
            - Community Forum
            - Email Support
            """
        )

    with col3:
        st.markdown(
            """
            #### Citation
            - Publication DOI
            - Software Citation
            - License Terms
            """
        )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
