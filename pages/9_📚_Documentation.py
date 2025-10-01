# pages/9_📚_Documentation.py

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, List
import logging

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
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box
except ImportError as e:
    logger.error(f"Module import failed: {e}")
    st.error(f"⚠️ Critical module import error: {e}")
    st.stop()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Documentation - SYPHU iGEM",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Documentation Content
# ============================================================================

QUICK_START_GUIDE = {
    "Data Preparation": {
        "icon": "📂",
        "description": "Load and manage your research data",
        "steps": [
            "Navigate to the **Data Management Hub**",
            "Choose upload method: Cloud download or local upload",
            "For cloud: Select dataset and click download",
            "For local: Browse files and upload",
            "Verify data appears in file browser"
        ],
        "tips": [
            "Supported formats: CSV, Excel, TSV, FASTA, FASTQ, images",
            "Maximum file size: 200 MB per file",
            "Use descriptive file names for easy identification"
        ]
    },
    "Dataset Activation": {
        "icon": "⚡",
        "description": "Set a dataset as active for analysis",
        "steps": [
            "Open **Data Management Hub**",
            "Browse available data files",
            "Click on desired file to preview",
            "Review data structure and content",
            "Click **Set as Active Dataset**"
        ],
        "tips": [
            "Only one dataset can be active at a time",
            "Active dataset is used across all analysis modules",
            "Check data quality before activating"
        ]
    },
    "Statistical Analysis": {
        "icon": "📊",
        "description": "Perform exploratory data analysis",
        "steps": [
            "Ensure dataset is activated",
            "Navigate to **Analysis Modules**",
            "View descriptive statistics automatically",
            "Explore correlation analysis",
            "Create visualizations",
            "Export results"
        ],
        "tips": [
            "Check for missing values first",
            "Understand data distribution",
            "Use appropriate statistical tests",
            "Document your analysis workflow"
        ]
    },
    "Machine Learning": {
        "icon": "🤖",
        "description": "Train and evaluate ML models",
        "steps": [
            "Go to **Machine Learning** module",
            "Choose analysis type (clustering/classification/regression)",
            "Select relevant features",
            "Configure model parameters",
            "Run analysis and review metrics",
            "Save results for reporting"
        ],
        "tips": [
            "Start with simple models",
            "Use train-test splitting",
            "Validate with cross-validation",
            "Check for overfitting"
        ]
    }
}

SYSTEM_REQUIREMENTS = {
    "Software": {
        "Python": "3.8 or higher",
        "Streamlit": "1.28 or higher",
        "Pandas": "1.5.0+",
        "NumPy": "1.20.0+",
        "Scikit-learn": "1.0.0+",
        "Plotly": "5.0.0+",
        "Pillow": "9.0.0+"
    },
    "Hardware": {
        "RAM": "Minimum 4GB, recommended 8GB+",
        "Storage": "At least 1GB free space",
        "Processor": "Multi-core processor recommended",
        "Display": "1920x1080 or higher resolution"
    },
    "Browser": {
        "Supported": "Chrome, Firefox, Safari, Edge (latest versions)",
        "JavaScript": "Must be enabled",
        "Cookies": "Must be enabled for session management"
    }
}

FAQ_ITEMS = [
    {
        "question": "How do I load data into the platform?",
        "answer": """
        There are three ways to load data:

        1. **Cloud Download**: In Data Hub, select from pre-configured Google Drive datasets
        2. **File Upload**: Upload your own files (CSV, Excel, etc.)
        3. **Dataset Selection**: Use the built-in example datasets

        After loading, activate the dataset by clicking "Set as Active Dataset" in the file browser.
        """
    },
    {
        "question": "What file formats are supported?",
        "answer": """
        **Data Files**: CSV, TSV, Excel (.xlsx, .xls), JSON, HDF5 (.h5ad)

        **Image Files**: PNG, JPEG, TIFF, BMP, SVG, microscopy formats (.nd2, .czi)

        **Sequence Files**: FASTA, FASTQ, GenBank, GFF, VCF

        **Model Files**: Pickle (.pkl), Joblib, HDF5, PyTorch (.pt)

        See the Data Configuration documentation for complete list.
        """
    },
    {
        "question": "How do I export my analysis results?",
        "answer": """
        Results can be exported from multiple locations:

        1. **Results Dashboard**: Download comprehensive JSON export
        2. **Individual Modules**: Each analysis has download buttons for specific results
        3. **Visualizations**: Use Plotly's built-in export to save figures as PNG/SVG
        4. **Tables**: Export data tables as CSV files

        All exports include timestamps and analysis metadata.
        """
    },
    {
        "question": "Why is my analysis taking a long time?",
        "answer": """
        Analysis speed depends on:

        - **Dataset size**: Larger datasets require more processing time
        - **Analysis complexity**: ML models and complex statistics take longer
        - **Hardware**: Available RAM and CPU cores affect performance
        - **Browser**: Some browsers perform better with interactive visualizations

        For very large datasets (>100,000 rows), consider:
        - Filtering or sampling data first
        - Using more powerful hardware
        - Running analyses in batches
        """
    },
    {
        "question": "How do I cite this platform in my publication?",
        "answer": """
        Suggested citation format:

        SYPHU-CHINA iGEM Team (2024). SYPHU iGEM Research Platform (Version 2.1.0) 
        [Computer software]. https://github.com/syphu-china/igem-platform

        For specific analysis methods, also cite the underlying libraries:
        - Statistical analysis: SciPy, Pandas
        - Machine learning: Scikit-learn
        - Visualization: Plotly
        - Bioinformatics: BioPython (if applicable)
        """
    },
    {
        "question": "What should I do if I encounter an error?",
        "answer": """
        Error troubleshooting steps:

        1. **Check the error message**: Note exact error text
        2. **Verify data format**: Ensure data matches expected format
        3. **Clear cache**: Refresh browser and clear session state
        4. **Check logs**: Browser console may show additional details
        5. **Report issue**: Use GitHub issues with error details and steps to reproduce

        Common issues:
        - Missing values: Clean data before analysis
        - Type errors: Ensure numeric columns are properly formatted
        - Memory errors: Reduce dataset size or increase RAM
        """
    },
    {
        "question": "How is my data stored and protected?",
        "answer": """
        **Data Storage**:
        - Local mode: All data stays on your machine
        - Session state: Data cleared when browser closes
        - No automatic cloud upload (unless you explicitly use cloud features)

        **Security**:
        - No data collection or telemetry
        - Open source code for transparency
        - Follow your institution's data policies
        - Use encrypted storage for sensitive data

        For collaborative work, use your institution's secure file sharing.
        """
    }
]

TUTORIALS = {
    "Basic Statistical Analysis": {
        "difficulty": "Beginner",
        "time": "10 minutes",
        "objective": "Learn how to load data and perform basic statistical analysis",
        "steps": [
            {
                "title": "Prepare Sample Data",
                "content": "Create or obtain a CSV file with numerical and categorical columns. Example: cell measurements with groups.",
                "code": """
# Example CSV structure:
# group,measurement_A,measurement_B,measurement_C
# Control,23.5,45.2,12.3
# Treatment,28.9,52.1,15.7
# ...
                """
            },
            {
                "title": "Upload and Activate Dataset",
                "content": "1. Go to Data Management Hub\n2. Click 'Upload Data' tab\n3. Select your CSV file\n4. Click upload\n5. Go to 'Browse Data' tab\n6. Select your file and click 'Set as Active Dataset'"
            },
            {
                "title": "View Descriptive Statistics",
                "content": "1. Navigate to Analysis Modules\n2. View automatically generated statistics\n3. Note mean, median, standard deviation for each variable"
            },
            {
                "title": "Create Visualizations",
                "content": "1. Go to 'Visualization' tab\n2. Select 'Distribution Plot'\n3. Choose a numerical variable\n4. Adjust bin size as needed\n5. Download figure using Plotly controls"
            },
            {
                "title": "Perform Statistical Tests",
                "content": "1. Go to 'Correlation Analysis' tab\n2. Select correlation method (Pearson/Spearman)\n3. Review correlation heatmap\n4. Identify significant correlations (|r| > 0.7)"
            }
        ]
    },
    "Machine Learning Clustering": {
        "difficulty": "Intermediate",
        "time": "15 minutes",
        "objective": "Use K-Means clustering to identify patterns in data",
        "steps": [
            {
                "title": "Prepare Numerical Data",
                "content": "Ensure dataset has at least 2 numerical columns. Handle missing values if present."
            },
            {
                "title": "Navigate to ML Module",
                "content": "1. Confirm active dataset is loaded\n2. Go to Machine Learning page\n3. Select 'Clustering' tab"
            },
            {
                "title": "Configure Clustering",
                "content": "1. Select K-Means algorithm\n2. Choose feature columns (2-5 recommended)\n3. Set number of clusters (start with 3)\n4. Enable data standardization\n5. Set random seed for reproducibility"
            },
            {
                "title": "Run Analysis",
                "content": "1. Click 'Run Clustering Analysis'\n2. Wait for processing\n3. Review silhouette score (>0.5 is good)\n4. Examine cluster sizes and distribution"
            },
            {
                "title": "Interpret Results",
                "content": "1. View PCA projection of clusters\n2. Check cluster characteristics\n3. Consider biological/experimental meaning\n4. Export results for further analysis"
            }
        ]
    },
    "Gene Enrichment Analysis": {
        "difficulty": "Intermediate",
        "time": "20 minutes",
        "objective": "Identify enriched pathways in a gene list",
        "steps": [
            {
                "title": "Prepare Gene List",
                "content": "Create a text file with gene symbols, one per line. Use official symbols (e.g., TP53, BRCA1, EGFR).",
                "code": """
# Example gene list:
TP53
BRCA1
EGFR
MYC
AKT1
VEGFA
PTEN
KRAS
PIK3CA
ERBB2
                """
            },
            {
                "title": "Upload Gene List",
                "content": "1. Go to Bioinformatics page\n2. Select 'Gene Enrichment' tab\n3. Choose input method: 'File Upload' or 'Manual Input'\n4. Upload/paste gene list\n5. Verify gene count"
            },
            {
                "title": "Configure Analysis",
                "content": "1. Select pathway database (KEGG, GO, etc.)\n2. Choose organism (Human, Mouse, etc.)\n3. Set p-value threshold (typically 0.05)\n4. Set minimum gene overlap (typically 3)\n5. Choose max pathways to display"
            },
            {
                "title": "Run Enrichment",
                "content": "1. Click 'Run Enrichment Analysis'\n2. Wait for results\n3. Review significant pathways\n4. Check p-values and gene counts"
            },
            {
                "title": "Interpret Findings",
                "content": "1. Examine top enriched pathways\n2. View genes in each pathway\n3. Consider biological context\n4. Download results table\n5. Use in manuscript preparation"
            }
        ]
    }
}


# ============================================================================
# Main Page Rendering
# ============================================================================

def main():
    """Main function to render Documentation page."""

    render_sidebar()

    render_page_header(
        title="Platform Documentation",
        icon="📚",
        subtitle="Comprehensive guides and references"
    )

    # Main navigation tabs
    tabs = st.tabs([
        "🚀 Quick Start",
        "📖 User Guide",
        "🎓 Tutorials",
        "❓ FAQ",
        "⚙️ System Requirements",
        "📝 Best Practices"
    ])

    with tabs[0]:
        render_quick_start_tab()

    with tabs[1]:
        render_user_guide_tab()

    with tabs[2]:
        render_tutorials_tab()

    with tabs[3]:
        render_faq_tab()

    with tabs[4]:
        render_system_requirements_tab()

    with tabs[5]:
        render_best_practices_tab()


# ============================================================================
# Tab 1: Quick Start
# ============================================================================

def render_quick_start_tab():
    """Render quick start guide."""

    render_section_header("Quick Start Guide", "🚀")

    render_info_box(
        content="""
        This guide will help you get started with the platform in under 5 minutes.
        Follow these steps to load data and perform your first analysis.
        """,
        box_type="info",
        title="Welcome!"
    )

    for idx, (title, content) in enumerate(QUICK_START_GUIDE.items(), 1):
        st.markdown(f"### {idx}. {content['icon']} {title}")
        st.markdown(f"_{content['description']}_")

        st.markdown("**Steps:**")
        for step in content['steps']:
            st.markdown(f"- {step}")

        with st.expander("💡 Tips & Best Practices"):
            for tip in content['tips']:
                st.markdown(f"- {tip}")

        if idx < len(QUICK_START_GUIDE):
            st.markdown("---")


# ============================================================================
# Tab 2: User Guide
# ============================================================================

def render_user_guide_tab():
    """Render comprehensive user guide."""

    render_section_header("User Guide", "📖")

    # Module-specific guides
    modules = {
        "Data Management": {
            "description": "Load, organize, and manage research data",
            "features": [
                "Cloud data download from Google Drive",
                "Local file upload (multiple formats)",
                "Data browsing and preview",
                "Dataset activation for analysis",
                "File validation and quality checks"
            ],
            "usage": """
            **Basic Workflow:**
            1. Upload or download data files
            2. Browse available datasets
            3. Preview data structure
            4. Activate dataset for analysis
            5. Manage data versions
            """
        },
        "Statistical Analysis": {
            "description": "Exploratory data analysis and hypothesis testing",
            "features": [
                "Descriptive statistics",
                "Correlation analysis (Pearson, Spearman, Kendall)",
                "Distribution analysis and normality tests",
                "Interactive visualizations",
                "Missing value handling"
            ],
            "usage": """
            **Basic Workflow:**
            1. Ensure dataset is activated
            2. Review descriptive statistics
            3. Explore correlations
            4. Create visualizations
            5. Export results
            """
        },
        "Machine Learning": {
            "description": "Unsupervised and supervised learning algorithms",
            "features": [
                "Clustering (K-Means, DBSCAN, Hierarchical)",
                "Dimensionality reduction (PCA, t-SNE)",
                "Classification and regression (Random Forest)",
                "Model evaluation metrics",
                "Cross-validation"
            ],
            "usage": """
            **Basic Workflow:**
            1. Select analysis type
            2. Choose features
            3. Configure parameters
            4. Train model
            5. Evaluate performance
            6. Save results
            """
        },
        "Image Analysis": {
            "description": "AI-powered microscopy image analysis",
            "features": [
                "Automated feature extraction",
                "Quality assessment",
                "Object detection (demonstration)",
                "Intensity analysis",
                "Batch processing support"
            ],
            "usage": """
            **Basic Workflow:**
            1. Upload images
            2. Select analysis type
            3. Configure AI model
            4. Run analysis
            5. Review metrics
            6. Export results
            """
        },
        "Bioinformatics": {
            "description": "Sequence and pathway analysis tools",
            "features": [
                "Gene enrichment analysis (KEGG, GO, Reactome)",
                "Sequence analysis (DNA/RNA/Protein)",
                "ORF detection",
                "Pattern searching",
                "Multiple sequence alignment support"
            ],
            "usage": """
            **Basic Workflow:**
            1. Prepare gene list or sequence
            2. Select analysis type
            3. Configure parameters
            4. Run analysis
            5. Interpret results
            6. Download reports
            """
        }
    }

    for module_name, module_info in modules.items():
        with st.expander(f"📘 {module_name}", expanded=False):
            st.markdown(f"**{module_info['description']}**")

            st.markdown("#### Key Features")
            for feature in module_info['features']:
                st.markdown(f"- {feature}")

            st.markdown("#### Usage")
            st.markdown(module_info['usage'])


# ============================================================================
# Tab 3: Tutorials
# ============================================================================

def render_tutorials_tab():
    """Render interactive tutorials."""

    render_section_header("Step-by-Step Tutorials", "🎓")

    for tutorial_name, tutorial_data in TUTORIALS.items():
        with st.expander(f"{tutorial_name} - {tutorial_data['difficulty']}", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Difficulty:** {tutorial_data['difficulty']}")
            with col2:
                st.markdown(f"**Time:** {tutorial_data['time']}")
            with col3:
                st.markdown(f"**Steps:** {len(tutorial_data['steps'])}")

            st.markdown(f"**Objective:** {tutorial_data['objective']}")
            st.markdown("---")

            for idx, step in enumerate(tutorial_data['steps'], 1):
                st.markdown(f"### Step {idx}: {step['title']}")
                st.markdown(step['content'])

                if 'code' in step:
                    st.code(step['code'], language='python')

                if idx < len(tutorial_data['steps']):
                    st.markdown("")


# ============================================================================
# Tab 4: FAQ
# ============================================================================

def render_faq_tab():
    """Render frequently asked questions."""

    render_section_header("Frequently Asked Questions", "❓")

    # Search functionality
    search_term = st.text_input("🔍 Search FAQ", placeholder="Enter keywords...")

    filtered_faq = FAQ_ITEMS
    if search_term:
        filtered_faq = [
            item for item in FAQ_ITEMS
            if search_term.lower() in item['question'].lower() or
               search_term.lower() in item['answer'].lower()
        ]

    if not filtered_faq:
        st.info("No matching questions found. Try different keywords.")
    else:
        for idx, item in enumerate(filtered_faq, 1):
            with st.expander(f"Q{idx}: {item['question']}", expanded=False):
                st.markdown(item['answer'])


# ============================================================================
# Tab 5: System Requirements
# ============================================================================

def render_system_requirements_tab():
    """Render system requirements."""

    render_section_header("System Requirements", "⚙️")

    for category, requirements in SYSTEM_REQUIREMENTS.items():
        st.markdown(f"### {category} Requirements")

        if isinstance(requirements, dict):
            for key, value in requirements.items():
                st.markdown(f"- **{key}:** {value}")

        st.markdown("")


# ============================================================================
# Tab 6: Best Practices
# ============================================================================

def render_best_practices_tab():
    """Render best practices guide."""

    render_section_header("Best Practices", "📝")

    practices = {
        "Data Quality": [
            "Check for missing values before analysis",
            "Validate data types (numeric vs categorical)",
            "Remove duplicates when appropriate",
            "Document data preprocessing steps",
            "Keep raw data separate from processed data"
        ],
        "Statistical Analysis": [
            "Check assumptions before applying tests",
            "Use appropriate significance levels (α = 0.05 typical)",
            "Apply multiple testing corrections when needed",
            "Report effect sizes alongside p-values",
            "Visualize data before and after analysis"
        ],
        "Machine Learning": [
            "Always split data into train/test sets",
            "Use cross-validation for robust estimates",
            "Check for overfitting (train vs test performance)",
            "Scale features when using distance-based methods",
            "Set random seeds for reproducibility"
        ],
        "Reproducibility": [
            "Document all analysis parameters",
            "Save random seeds used",
            "Export analysis logs",
            "Version control analysis scripts",
            "Include software versions in reports"
        ],
        "Data Security": [
            "Never upload sensitive data without permission",
            "Use encrypted storage for confidential data",
            "Follow institutional data policies",
            "Clear browser cache after sensitive work",
            "Be cautious with cloud features"
        ],
        "Publication Preparation": [
            "Export high-resolution figures (300 DPI minimum)",
            "Use colorblind-friendly palettes",
            "Include figure captions and legends",
            "Report statistical methods clearly",
            "Make data and code available when possible"
        ]
    }

    for category, items in practices.items():
        st.markdown(f"### {category}")
        for item in items:
            st.markdown(f"- {item}")
        st.markdown("")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
