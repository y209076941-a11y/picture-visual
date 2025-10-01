# app.py
"""
SYPHU-CHINA iGEM 2025 Research Platform
========================================

An integrated computational biology platform for data analysis, machine learning,
and bioinformatics workflows in synthetic biology research.

This application follows Nature journal standards for scientific software:
- Modular architecture for reproducibility
- Comprehensive documentation
- Clear data provenance tracking
- Standardized visualization formats

Authors: SYPHU-CHINA iGEM 2025 Team
License: MIT
Version: 2.1.0
Last Updated: October 2025

References
----------
.. [1] Nature Methods (2014). Software with impact. doi:10.1038/nmeth.2880
.. [2] Nature Computational Science (2025). Software in science is ubiquitous
       yet overlooked. doi:10.1038/s43588-024-00651-2
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION: Scientific Computing Standards
# ============================================================================

# Configure scientific logging with ISO 8601 timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to Python path for reproducible imports
PLATFORM_ROOT = Path(__file__).parent.resolve()
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

# ============================================================================
# MODULE IMPORTS: Dependency Management
# ============================================================================

# Import core modules
try:
    from components.sidebar import render_sidebar
    from utils.data_manager import DataManager
    from utils.data_loader import DataLoader
    from config.data_config import DataConfig

    logger.info("✓ All platform modules successfully imported")
except ImportError as e:
    logger.error(f"✗ Critical module import failed: {e}")
    st.error(f"""
    ⚠️ **Module Import Error**

    Missing dependency: `{e.name if hasattr(e, 'name') else 'unknown'}`

    **Troubleshooting:**
    1. Ensure all required modules exist in the correct directories
    2. Verify Python path includes project root: `{PLATFORM_ROOT}`
    3. Check that __init__.py files exist in all module directories

    **For support:** https://github.com/syphu-china/igem-platform/issues
    """)
    st.stop()

# ============================================================================
# PAGE CONFIGURATION: Nature-Style Interface
# ============================================================================

st.set_page_config(
    page_title="SYPHU iGEM Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/syphu-china/igem-platform',
        'Report a bug': 'https://github.com/syphu-china/igem-platform/issues',
        'About': '''
        ## SYPHU-CHINA iGEM 2025 Research Platform

        **Version:** 2.1.0  
        **License:** MIT  
        **Team:** SYPHU-CHINA iGEM 2025

        ### Citation
        If you use this platform in your research, please cite:
        ```
        SYPHU-CHINA iGEM Team (2025). 
        Integrated Research Platform for Synthetic Biology.
        https://github.com/syphu-china/igem-platform
        ```

        ### Key Features
        - Multi-omics data integration
        - Machine learning pipelines
        - Bioinformatics analysis
        - Reproducible workflows

        ### Standards Compliance
        - FAIR data principles
        - Nature journal formatting
        - ISO 8601 timestamps
        - Semantic versioning
        '''
    }
)

# ============================================================================
# CONSTANTS: Platform Configuration
# ============================================================================

PLATFORM_METADATA = {
    'name': 'SYPHU-CHINA iGEM Research Platform',
    'version': '2.1.0',
    'version_date': '2025-10-01',
    'team': 'SYPHU-CHINA iGEM 2025',
    'license': 'MIT',
    'doi': None,
    'repository': 'https://github.com/syphu-china/igem-platform',
    'documentation': 'https://syphu-china.github.io/igem-platform-docs',
    'citation': 'SYPHU-CHINA iGEM Team (2025). Integrated Research Platform for Synthetic Biology.'
}

RESEARCH_MODULES = {
    'data_hub': {
        'icon': '📂',
        'title': 'Data Management Hub',
        'subtitle': 'Centralized repository for multi-omics data',
        'description': '''
        Unified interface for data acquisition, organization, and quality control.
        Supports multiple data formats (CSV, Excel, HDF5, FASTA) with automated
        validation and metadata extraction.
        ''',
        'page': 'pages/2_📂_Data_Hub.py',
        'methods': [
            'Cloud-based data retrieval',
            'Local file upload and validation',
            'Interactive data preview',
            'Dataset activation and versioning'
        ],
        'supported_formats': ['CSV', 'Excel', 'TSV', 'HDF5', 'JSON', 'H5AD'],
        'typical_use': 'Initial data import and preprocessing'
    },

    'statistical_analysis': {
        'icon': '🔬',
        'title': 'Statistical Analysis',
        'subtitle': 'Exploratory data analysis and hypothesis testing',
        'description': '''
        Comprehensive statistical toolkit for biological data analysis.
        Implements standard methods from descriptive statistics to multivariate
        analysis, with automated assumption checking and effect size reporting.
        ''',
        'page': 'pages/3_🔬_Analysis_Modules.py',
        'methods': [
            'Descriptive statistics (mean, median, SD, IQR)',
            'Correlation analysis (Pearson, Spearman)',
            'Hypothesis testing (t-test, ANOVA, Mann-Whitney)',
            'Distribution analysis and normality tests'
        ],
        'outputs': ['Summary tables', 'Statistical plots', 'P-value matrices'],
        'typical_use': 'Initial data exploration and quality assessment'
    },

    'machine_learning': {
        'icon': '🤖',
        'title': 'Machine Learning',
        'subtitle': 'Supervised and unsupervised learning pipelines',
        'description': '''
        State-of-the-art machine learning algorithms optimized for biological data.
        Includes dimensionality reduction, clustering, classification, and regression
        with cross-validation and performance metrics.
        ''',
        'page': 'pages/5_🤖_Machine_Learning.py',
        'methods': [
            'Unsupervised: PCA, t-SNE, UMAP, hierarchical clustering',
            'Supervised: Random Forest, SVM, Gradient Boosting',
            'Feature selection and importance ranking',
            'Model validation and hyperparameter tuning'
        ],
        'algorithms': ['scikit-learn', 'XGBoost', 'TensorFlow'],
        'typical_use': 'Pattern discovery and predictive modeling'
    },

    'image_analysis': {
        'icon': '🖼️',
        'title': 'AI-Powered Image Analysis',
        'subtitle': 'Deep learning for microscopy and experimental imaging',
        'description': '''
        Automated image analysis using convolutional neural networks.
        Supports object detection, segmentation, and feature extraction
        for high-throughput microscopy workflows.
        ''',
        'page': 'pages/4_🖼️_AI_Image_Analysis.py',
        'methods': [
            'Image quality assessment',
            'Feature extraction (texture, morphology)',
            'Object detection and counting',
            'Batch processing pipelines'
        ],
        'supported_formats': ['PNG', 'JPEG', 'TIFF', 'CZI', 'ND2'],
        'typical_use': 'Microscopy data quantification'
    },

    'bioinformatics': {
        'icon': '🧬',
        'title': 'Bioinformatics Toolkit',
        'subtitle': 'Sequence analysis and pathway enrichment',
        'description': '''
        Specialized tools for genomic and proteomic analysis.
        Integrates sequence manipulation, alignment, gene enrichment,
        and metabolic pathway analysis.
        ''',
        'page': 'pages/7_🧬_Bioinformatics.py',
        'methods': [
            'Sequence manipulation (translation, complement)',
            'ORF detection and annotation',
            'Gene ontology enrichment',
            'Pathway analysis (KEGG, Reactome)'
        ],
        'databases': ['NCBI', 'UniProt', 'KEGG', 'GO'],
        'typical_use': 'Genomic sequence characterization'
    },

    'experiment_management': {
        'icon': '🧪',
        'title': 'Experiment Management',
        'subtitle': 'Electronic lab notebook and protocol tracking',
        'description': '''
        Digital laboratory notebook for experiment documentation.
        Tracks protocols, results, and metadata with version control
        and reproducibility features.
        ''',
        'page': 'pages/6_🧪_Experiment_Hub.py',
        'methods': [
            'Protocol versioning and templates',
            'Progress tracking and milestones',
            'Structured note-taking',
            'Result archiving with metadata'
        ],
        'standards': ['ELN best practices', 'ISO 17025'],
        'typical_use': 'Laboratory workflow documentation'
    }
}


# ============================================================================
# STYLING: Nature Journal Visual Standards
# ============================================================================

def apply_nature_styling():
    """Apply Nature journal-inspired styling to the interface."""

    st.markdown("""
    <style>
        :root {
            --nature-blue: #0066CC;
            --nature-gray: #555555;
            --alert-red: #E63946;
            --success-green: #06A77D;
            --background: #FFFFFF;
            --surface: #F8F9FA;
            --border: #E0E0E0;
            --text-primary: #2C3E50;
            --text-secondary: #7F8C8D;
        }

        * { box-sizing: border-box; }

        body {
            font-family: 'Arial', sans-serif;
            color: var(--text-primary);
            line-height: 1.6;
        }

        .main-header {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-size: 2.25rem;
            font-weight: 400;
            color: var(--text-primary);
            text-align: center;
            padding: 2.5rem 0 1rem 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--nature-blue);
            letter-spacing: -0.02em;
        }

        .main-header::after {
            content: '';
            display: block;
            width: 60px;
            height: 3px;
            background: var(--nature-blue);
            margin: 1rem auto 0;
        }

        .section-header {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-size: 1.5rem;
            font-weight: 500;
            color: var(--text-primary);
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .research-module-card {
            background: var(--background);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            height: 100%;
        }

        .research-module-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
            border-color: var(--nature-blue);
        }

        .module-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 0.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .module-subtitle {
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-style: italic;
            margin: 0 0 1rem 0;
        }

        .module-description {
            font-size: 0.95rem;
            color: var(--text-primary);
            line-height: 1.6;
            margin: 1rem 0;
            text-align: justify;
        }

        .status-panel {
            background: linear-gradient(135deg, var(--nature-blue), #0052A3);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 12px rgba(0, 102, 204, 0.2);
        }

        .status-panel h3 {
            margin: 0 0 1rem 0;
            font-size: 1.25rem;
            font-weight: 500;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .status-label {
            font-weight: 500;
        }

        .status-value {
            font-family: 'Courier New', monospace;
            background: rgba(255, 255, 255, 0.2);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
            color: var(--nature-blue);
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stButton > button {
            background: var(--nature-blue);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0, 102, 204, 0.2);
        }

        .stButton > button:hover {
            background: #0052A3;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3);
        }

        .info-box {
            padding: 1rem 1.5rem;
            border-radius: 6px;
            margin: 1rem 0;
            border-left: 4px solid;
        }

        .info-box-warning {
            background: #FFF3E0;
            border-color: #F57C00;
            color: #E65100;
        }

        .footer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
            padding: 2rem 0 1rem 0;
            margin-top: 3rem;
            border-top: 1px solid var(--border);
        }

        .footer a {
            color: var(--nature-blue);
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        @media (max-width: 768px) {
            .main-header {
                font-size: 1.75rem;
            }

            .research-module-card {
                padding: 1rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SYSTEM DIAGNOSTICS: Platform Health Monitoring
# ============================================================================

def perform_system_diagnostics() -> Dict[str, Any]:
    """
    Comprehensive system status check following FAIR principles.

    Returns
    -------
    Dict[str, Any]
        Diagnostic report with system health status
    """

    logger.info("Initiating system diagnostics...")

    try:
        # Retrieve configured data directory
        data_root = DataConfig.get_active_local_root()
        logger.info(f"Data root: {data_root}")

        # Check directory existence
        root_exists = data_root.exists()

        # Scan for research data
        if root_exists:
            has_data, file_count = DataLoader.check_data_availability(str(data_root))
            file_inventory = DataLoader.scan_research_data(str(data_root)) if has_data else {}
        else:
            has_data, file_count = False, 0
            file_inventory = {}
            logger.warning(f"Data directory not found: {data_root}")

        # Assess system health
        if has_data and file_count > 0:
            system_health = 'healthy'
            recommendations = []
        elif root_exists and file_count == 0:  # Fixed: 'but' -> 'and'
            system_health = 'degraded'
            recommendations = [
                'No research data detected in configured directory',
                'Navigate to Data Hub to import datasets',
                'Verify data directory configuration'
            ]
        else:
            system_health = 'critical'
            recommendations = [
                'Data directory does not exist',
                'Check configuration in config/data_config.py',
                'Create directory or update path'
            ]

        # Compile diagnostic report
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'data_root': data_root,
            'root_exists': root_exists,
            'data_available': has_data,
            'file_count': file_count,
            'file_inventory': file_inventory,
            'system_health': system_health,
            'recommendations': recommendations,
            'metadata': {
                'platform_version': PLATFORM_METADATA['version'],
                'python_version': sys.version.split()[0],
                'streamlit_version': st.__version__
            }
        }

        logger.info(f"Diagnostics complete - Status: {system_health}")
        return diagnostics

    except Exception as e:
        logger.error(f"Diagnostic failure: {str(e)}", exc_info=True)
        return {
            'timestamp': datetime.now().isoformat(),
            'data_root': None,
            'root_exists': False,
            'data_available': False,
            'file_count': 0,
            'file_inventory': {},
            'system_health': 'critical',
            'recommendations': ['System error occurred', 'Check logs for details'],
            'error': str(e)
        }


# ============================================================================
# INTERFACE COMPONENTS
# ============================================================================

def render_platform_header():
    """Render main platform header in Nature journal style."""

    st.markdown(f'''
    <div class="main-header">
        {PLATFORM_METADATA['name']}
    </div>
    ''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'''
        <p style="text-align: center; color: #7F8C8D; font-size: 0.95rem; margin-top: -1rem;">
            <strong>Version {PLATFORM_METADATA['version']}</strong> | 
            {PLATFORM_METADATA['team']} | 
            <a href="{PLATFORM_METADATA['repository']}" target="_blank" style="color: #0066CC;">GitHub</a>
        </p>
        ''', unsafe_allow_html=True)


def render_system_status_panel(diagnostics: Dict[str, Any]):
    """Display system status in an information-dense panel."""

    health_icons = {
        'healthy': '✅',
        'degraded': '⚠️',
        'critical': '❌'
    }

    health_colors = {
        'healthy': 'linear-gradient(135deg, #06A77D, #048860)',
        'degraded': 'linear-gradient(135deg, #F59E0B, #D97706)',
        'critical': 'linear-gradient(135deg, #E63946, #D32F3D)'
    }

    health = diagnostics['system_health']
    icon = health_icons.get(health, '❓')
    color = health_colors.get(health, 'linear-gradient(135deg, #6B7280, #4B5563)')

    st.markdown(f'''
    <div class="status-panel" style="background: {color};">
        <h3>{icon} System Status: {health.capitalize()}</h3>
        <div class="status-item">
            <span class="status-label">Data Directory</span>
            <span class="status-value">{diagnostics['data_root']}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Directory Status</span>
            <span class="status-value">{'Accessible' if diagnostics['root_exists'] else 'Not Found'}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Research Files</span>
            <span class="status-value">{diagnostics['file_count']} detected</span>
        </div>
        <div class="status-item">
            <span class="status-label">Last Check</span>
            <span class="status-value">{datetime.fromisoformat(diagnostics['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if diagnostics['recommendations']:
        with st.expander("📋 System Recommendations", expanded=(health != 'healthy')):
            for rec in diagnostics['recommendations']:
                st.info(rec)


def render_data_repository_metrics(file_inventory: Dict[str, List[str]]):
    """Display file statistics using Nature-style metrics."""

    st.markdown('<h2 class="section-header">📊 Data Repository Overview</h2>',
                unsafe_allow_html=True)

    st.markdown('''
    <p style="color: #7F8C8D; margin-bottom: 1.5rem;">
    Current inventory of research data files organized by type. 
    File counts include all recognized formats in the active data directory.
    </p>
    ''', unsafe_allow_html=True)

    metric_definitions = {
        'data': {
            'label': 'Tabular Data',
            'icon': '📊',
            'formats': 'CSV, Excel, TSV, HDF5',
            'description': 'Structured datasets for statistical analysis'
        },
        'images': {
            'label': 'Image Files',
            'icon': '🖼️',
            'formats': 'PNG, JPG, TIFF, microscopy',
            'description': 'Experimental images and microscopy data'
        },
        'sequences': {
            'label': 'Sequence Data',
            'icon': '🧬',
            'formats': 'FASTA, FASTQ, GenBank',
            'description': 'Genomic and proteomic sequences'
        },
        'results': {
            'label': 'Analysis Results',
            'icon': '📈',
            'formats': 'JSON, HTML, PDF',
            'description': 'Generated analysis outputs'
        },
        'notebooks': {
            'label': 'Documentation',
            'icon': '📓',
            'formats': 'Jupyter, Markdown, text',
            'description': 'Research notes and protocols'
        }
    }

    cols = st.columns(5)

    for idx, (category, definition) in enumerate(metric_definitions.items()):
        file_count = len(file_inventory.get(category, []))

        with cols[idx]:
            st.metric(
                label=f"{definition['icon']} {definition['label']}",
                value=file_count,
                help=f"{definition['description']}\n\n**Formats:** {definition['formats']}"
            )

    if any(len(files) > 0 for files in file_inventory.values()):
        with st.expander("📁 View Detailed File Inventory"):
            for category, definition in metric_definitions.items():
                files = file_inventory.get(category, [])
                if files:
                    st.markdown(f"**{definition['icon']} {definition['label']}** ({len(files)} files)")

                    display_files = files[:5]
                    for file in display_files:
                        file_path = Path(file)
                        try:
                            file_size = file_path.stat().st_size if file_path.exists() else 0
                            size_mb = file_size / (1024 * 1024)
                            st.markdown(f"- `{file_path.name}` ({size_mb:.2f} MB)")
                        except:
                            st.markdown(f"- `{file_path.name}`")

                    if len(files) > 5:
                        st.markdown(f"*... and {len(files) - 5} more files*")

                    st.markdown("---")


def render_research_module_cards():
    """Display research module cards in Nature journal figure style."""

    st.markdown('<h2 class="section-header">🚀 Research Modules</h2>',
                unsafe_allow_html=True)

    st.markdown('''
    <p style="color: #7F8C8D; margin-bottom: 2rem;">
    Integrated computational tools for multi-omics data analysis. 
    Each module provides specialized functionality with standardized interfaces 
    and reproducible workflows.
    </p>
    ''', unsafe_allow_html=True)

    module_items = list(RESEARCH_MODULES.items())

    for row_start in range(0, len(module_items), 3):
        cols = st.columns(3)
        row_modules = module_items[row_start:row_start + 3]

        for col_idx, (module_id, module_config) in enumerate(row_modules):
            with cols[col_idx]:
                render_single_module_card(module_id, module_config)


def render_single_module_card(module_id: str, config: Dict[str, Any]):
    """Render individual research module card."""

    st.markdown(f'''
    <div class="research-module-card">
        <div class="module-title">
            <span>{config['icon']}</span>
            <span>{config['title']}</span>
        </div>
        <div class="module-subtitle">{config['subtitle']}</div>
        <div class="module-description">{config['description']}</div>
    </div>
    ''', unsafe_allow_html=True)

    with st.expander("Methods & Capabilities"):
        st.markdown("**Key Features:**")
        for method in config['methods']:
            st.markdown(f"- {method}")

        if 'typical_use' in config:
            st.markdown(f"\n**Typical Use Case:** {config['typical_use']}")

        if 'supported_formats' in config:
            st.markdown(f"\n**Supported Formats:** {', '.join(config['supported_formats'])}")

    if st.button(
            f"Launch {config['title']}",
            key=f"launch_{module_id}",
            use_container_width=True,
            type="primary"
    ):
        try:
            st.switch_page(config['page'])
        except Exception as e:
            st.warning(f"Page not yet available: {config['page']}")


def render_quick_navigation():
    """Provide quick access buttons to frequently used modules."""

    st.markdown('<h2 class="section-header">⚡ Quick Navigation</h2>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    quick_links = [
        (col1, "📂 Data Hub", "pages/2_📂_Data_Hub.py", "Import & manage data"),
        (col2, "🔬 Analysis", "pages/3_🔬_Analysis_Modules.py", "Statistical tests"),
        (col3, "🤖 ML Tools", "pages/5_🤖_Machine_Learning.py", "Machine learning"),
        (col4, "📊 Results", "pages/8_📈_Results.py", "View outputs")
    ]

    for col, label, page, description in quick_links:
        with col:
            if st.button(label, key=f"quick_{page}", use_container_width=True, help=description):
                try:
                    st.switch_page(page)
                except:
                    st.info(f"Page not yet available: {page}")


def render_no_data_guidance(diagnostics: Dict[str, Any]):
    """Provide guidance when no research data is detected."""

    st.markdown('<h2 class="section-header">⚠️ Getting Started</h2>',
                unsafe_allow_html=True)

    st.markdown(f'''
    <div class="info-box info-box-warning">
        <h3>No Research Data Detected</h3>
        <p><strong>Data Directory:</strong> <code>{diagnostics['data_root']}</code></p>
        <p><strong>Status:</strong> {'Directory exists but is empty' if diagnostics['root_exists'] else 'Directory not found'}</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("### 📋 Setup Instructions")

    st.markdown("""
    **To begin using the platform:**

    1. **Navigate to Data Hub**  
       Click the button below to access data management tools

    2. **Import Research Data**  
       - Download from cloud storage (if configured)
       - Upload local files (CSV, Excel, images, sequences)
       - Connect to external databases

    3. **Verify Data Loading**  
       Return to this page to confirm files are detected

    4. **Start Analysis**  
       Access analysis modules once data is available

    **Supported File Formats:**
    - **Data**: CSV, Excel, TSV, HDF5, JSON, H5AD
    - **Images**: PNG, JPG, TIFF, CZI, ND2
    - **Sequences**: FASTA, FASTQ, GenBank
    - **Results**: JSON, HTML, PDF
    """)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("📂 Open Data Hub", type="primary", use_container_width=True):
            try:
                st.switch_page("pages/2_📂_Data_Hub.py")
            except:
                st.info("Data Hub page will be available soon")

    with col2:
        if st.button("📚 View Documentation", use_container_width=True):
            try:
                st.switch_page("pages/9_📚_Documentation.py")
            except:
                st.info("Documentation page will be available soon")

    with col3:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()


def render_platform_footer():
    """Render footer with citation information and links."""

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📖 Citation")
        st.markdown(f"""
        {PLATFORM_METADATA['citation']}

        **Version:** {PLATFORM_METADATA['version']}  
        **Released:** {PLATFORM_METADATA['version_date']}
        """)

        if st.button("📋 Copy Citation", key="copy_citation"):
            st.code(PLATFORM_METADATA['citation'], language=None)
            st.success("Citation text displayed above")

    with col2:
        st.markdown("### 📚 Resources")
        st.markdown(f"""
        - [Documentation]({PLATFORM_METADATA['documentation']})
        - [Source Code]({PLATFORM_METADATA['repository']})
        - [Issue Tracker]({PLATFORM_METADATA['repository']}/issues)
        - [Team Website](https://syphu-china.igem.org)
        """)

    with col3:
        st.markdown("### ℹ️ Information")
        st.markdown(f"""
        **License:** {PLATFORM_METADATA['license']}  
        **Team:** {PLATFORM_METADATA['team']}  
        **Platform:** Streamlit {st.__version__}  
        **Python:** {sys.version.split()[0]}
        """)

    st.markdown(f'''
    <div class="footer">
        <p>© 2025 {PLATFORM_METADATA['team']}. All rights reserved.</p>
        <p>This platform adheres to <a href="https://www.nature.com/documents/nr-software-policy.pdf" target="_blank">Nature journal standards</a> for scientific software.</p>
    </div>
    ''', unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION CONTROLLER
# ============================================================================

def main():
    """
    Main application controller following Nature journal standards.

    Architecture
    ------------
    1. Initialize session state and configuration
    2. Perform system diagnostics
    3. Render interface components conditionally
    4. Provide clear navigation paths

    Notes
    -----
    Implements error handling and logging throughout for
    reproducible execution and debugging.
    """

    logger.info("=" * 70)
    logger.info(f"Platform startup: {PLATFORM_METADATA['name']}")
    logger.info(f"Version: {PLATFORM_METADATA['version']}")
    logger.info("=" * 70)

    try:
        # Initialize application state
        DataManager.initialize_session_state()
        logger.info("Session state initialized")

        # Apply styling
        apply_nature_styling()

        # Render sidebar navigation
        render_sidebar()

        # Main content area
        render_platform_header()

        # System diagnostics
        with st.spinner("Performing system diagnostics..."):
            diagnostics = perform_system_diagnostics()

        # Status panel
        render_system_status_panel(diagnostics)

        # Conditional content based on data availability
        if diagnostics['data_available']:
            # Data repository metrics
            render_data_repository_metrics(diagnostics['file_inventory'])

            st.markdown("---")

            # Research modules
            render_research_module_cards()

            st.markdown("---")

            # Quick navigation
            render_quick_navigation()

        else:
            # No data guidance
            render_no_data_guidance(diagnostics)

            st.markdown("---")

            # Still show modules for reference
            render_research_module_cards()

        # Footer
        render_platform_footer()

        logger.info("Main page rendering complete")

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)

        st.error(f"""
        **⚠️ Application Error**

        An unexpected error occurred during execution.

        **Error Details:**
        ```
        {str(e)}
        ```

        **Troubleshooting Steps:**
        1. Refresh the page (F5)
        2. Clear browser cache
        3. Check system logs
        4. Report issue on GitHub

        **Support:** {PLATFORM_METADATA['repository']}/issues
        """)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.critical(f"Critical application failure: {str(e)}", exc_info=True)
        st.error("Critical system error. Please contact support.")
