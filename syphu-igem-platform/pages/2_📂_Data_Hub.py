# pages/2_📂_Data_Hub.py

import streamlit as st
import os
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

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
    from utils.data_loader import DataLoader
    from utils.data_manager import DataManager
    from components.sidebar import render_sidebar
    from components.headers import render_page_header, render_section_header, render_info_box
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
    page_title="Data Management - SYPHU iGEM",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Constants and Configuration
# ============================================================================

MAX_UPLOAD_SIZE_MB = 200
PREVIEW_ROW_LIMIT = 100
SUPPORTED_PREVIEW_FORMATS = ['.csv', '.xlsx', '.xls', '.tsv', '.txt']


# ============================================================================
# Utility Functions
# ============================================================================

def format_file_size(size_bytes: int) -> str:
    """
    Convert file size from bytes to human-readable format.

    Parameters
    ----------
    size_bytes : int
        File size in bytes.

    Returns
    -------
    str
        Formatted file size string (e.g., "1.5 MB", "340 KB").

    Examples
    --------
    >>> format_file_size(1536000)
    '1.5 MB'
    """

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_file_metadata(file_path: str) -> Dict[str, str]:
    """
    Extract comprehensive metadata from a file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    Dict[str, str]
        Dictionary containing file metadata (name, size, type, modified date).

    Notes
    -----
    Returns empty dict if file doesn't exist or metadata extraction fails.
    """

    try:
        path = Path(file_path)
        if not path.exists():
            return {}

        stat = path.stat()
        modified_time = datetime.fromtimestamp(stat.st_mtime)

        return {
            'name': path.name,
            'size': format_file_size(stat.st_size),
            'size_bytes': stat.st_size,
            'type': path.suffix,
            'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
            'category': DataConfig.get_file_category(file_path) or 'unknown'
        }
    except Exception as e:
        logger.error(f"Error getting file metadata: {e}")
        return {}


def calculate_directory_stats(directory: str) -> Dict[str, any]:
    """
    Calculate statistics for a directory.

    Parameters
    ----------
    directory : str
        Path to directory.

    Returns
    -------
    Dict[str, any]
        Statistics including total size, file count, and file type distribution.
    """

    try:
        total_size = 0
        file_count = 0
        type_distribution = {}

        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                try:
                    filepath = os.path.join(dirpath, filename)
                    size = os.path.getsize(filepath)
                    total_size += size
                    file_count += 1

                    # Track file type distribution
                    ext = Path(filename).suffix.lower()
                    type_distribution[ext] = type_distribution.get(ext, 0) + 1
                except OSError:
                    continue

        return {
            'total_size': total_size,
            'total_size_formatted': format_file_size(total_size),
            'file_count': file_count,
            'type_distribution': type_distribution
        }
    except Exception as e:
        logger.error(f"Error calculating directory stats: {e}")
        return {
            'total_size': 0,
            'total_size_formatted': '0 B',
            'file_count': 0,
            'type_distribution': {}
        }


# ============================================================================
# Tab 1: Download Data from Cloud
# ============================================================================

def render_download_tab() -> None:
    """
    Render the cloud data download interface.

    Displays available Google Drive datasets with download functionality
    and status tracking.

    Notes
    -----
    Requires valid Google Drive file IDs to be configured in DataConfig.
    """

    render_section_header("Cloud Data Repository", "🌐")

    render_info_box(
        content="""
        Access shared research datasets from the team's Google Drive repository.
        These datasets are version-controlled and curated for reproducibility.
        """,
        box_type="info",
        title="About Cloud Data"
    )

    google_datasets = DataConfig.GOOGLE_DRIVE_DATASETS

    if not google_datasets:
        st.warning("⚠️ No cloud datasets configured")
        render_info_box(
            content="Please configure Google Drive dataset IDs in `config/data_config.py`",
            box_type="warning"
        )
        return

    # Display datasets in organized cards
    for dataset_id, dataset_info in google_datasets.items():
        with st.expander(
                f"{dataset_info.get('name', 'Unnamed Dataset')}",
                expanded=False
        ):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**Description:** {dataset_info['description']}")
                st.markdown(f"**Format:** `{dataset_info['file_type']}`")
                st.markdown(f"**Size:** ~{dataset_info.get('size_mb', 'Unknown')}")
                if 'last_updated' in dataset_info:
                    st.markdown(f"**Last Updated:** {dataset_info['last_updated']}")

            with col2:
                # Check download status
                local_data_dir = DataConfig.get_active_local_root()
                local_path = local_data_dir / f"{dataset_id}.{dataset_info['file_type']}"

                if local_path.exists():
                    st.success("✅ Downloaded")
                    metadata = get_file_metadata(str(local_path))
                    if metadata:
                        st.caption(f"Local size: {metadata['size']}")
                else:
                    st.warning("⬇️ Not downloaded")

            with col3:
                # Download button
                drive_id = dataset_info.get('drive_id', '')

                if drive_id == "REPLACE_WITH_ACTUAL_DRIVE_ID":
                    st.error("⚠️ ID not configured")
                    if st.button("Setup Guide", key=f"guide_{dataset_id}"):
                        st.info("""
                        **Setup Instructions:**
                        1. Share your file on Google Drive
                        2. Get the file ID from the sharing link
                        3. Update `config/data_config.py`
                        4. Replace `REPLACE_WITH_ACTUAL_DRIVE_ID`
                        """)
                else:
                    button_label = "Re-download" if local_path.exists() else "Download"
                    if st.button(
                            button_label,
                            key=f"download_{dataset_id}",
                            use_container_width=True
                    ):
                        download_from_google_drive(
                            drive_id,
                            str(local_path),
                            dataset_info['name']
                        )


def download_from_google_drive(
        file_id: str,
        destination: str,
        dataset_name: str
) -> None:
    """
    Download file from Google Drive.

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    destination : str
        Local destination path.
    dataset_name : str
        Human-readable dataset name for display.

    Notes
    -----
    This function provides a placeholder implementation. For production use,
    integrate with gdown or google-auth libraries.
    """

    try:
        with st.spinner(f"Downloading {dataset_name}..."):
            # Placeholder for actual download implementation
            st.info(f"""
            **Download Implementation Required**

            To implement actual downloads, install and use:
            ```bash
            pip install gdown
            ```

            Then use:
            ```python
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, destination, quiet=False)
            ```

            **Current file ID:** `{file_id}`
            """)

            logger.info(f"Download initiated for {dataset_name}")
    except Exception as e:
        logger.error(f"Download error: {e}")
        st.error(f"⚠️ Download failed: {str(e)}")


# ============================================================================
# Tab 2: Upload Local Data
# ============================================================================

def render_upload_tab() -> None:
    """
    Render the local file upload interface.

    Provides drag-and-drop file upload with validation, size checking,
    and automatic organization.
    """

    render_section_header("Upload Local Data", "📤")

    local_data_dir = DataConfig.get_active_local_root()

    # Display upload target and current status
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Select files to upload",
            accept_multiple_files=True,
            type=None,  # Accept all types, validate manually
            help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB per file"
        )

    with col2:
        render_info_box(
            content=f"""
            **Upload Directory:**  
            `{local_data_dir}`

            **Current Files:**  
            {len(list(local_data_dir.glob('*')) if local_data_dir.exists() else [])}
            """,
            box_type="info",
            title="Storage Info"
        )

    if uploaded_files:
        st.markdown("---")
        st.markdown("#### 📋 Upload Queue")

        # Validate and display upload queue
        valid_files = []
        invalid_files = []

        for uploaded_file in uploaded_files:
            # Validate file
            is_valid, error_msg = validate_upload(uploaded_file)

            if is_valid:
                valid_files.append(uploaded_file)
            else:
                invalid_files.append((uploaded_file, error_msg))

        # Display invalid files
        if invalid_files:
            st.warning(f"⚠️ {len(invalid_files)} file(s) failed validation:")
            for file, error in invalid_files:
                st.error(f"- {file.name}: {error}")

        # Display valid files
        if valid_files:
            st.success(f"✅ {len(valid_files)} file(s) ready for upload")

            for uploaded_file in valid_files:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"**{uploaded_file.name}**")
                    category = DataConfig.get_file_category(uploaded_file.name)
                    st.caption(f"Type: {Path(uploaded_file.name).suffix} | Category: {category or 'Other'}")

                with col2:
                    st.write(format_file_size(uploaded_file.size))

                with col3:
                    if st.button("Upload", key=f"upload_{uploaded_file.name}"):
                        process_upload(uploaded_file, local_data_dir)

        # Batch upload button
        if valid_files:
            st.markdown("---")
            if st.button("📦 Upload All Valid Files", type="primary", use_container_width=True):
                success_count = 0
                for uploaded_file in valid_files:
                    if process_upload(uploaded_file, local_data_dir, show_success=False):
                        success_count += 1

                if success_count > 0:
                    st.success(f"✅ Successfully uploaded {success_count} file(s)")
                    st.rerun()


def validate_upload(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file before processing.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object.

    Returns
    -------
    Tuple[bool, str or None]
        (is_valid, error_message). error_message is None if valid.
    """

    # Check file size
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return False, f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit ({size_mb:.1f} MB)"

    # Check file extension
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in DataConfig.get_supported_extensions_list():
        return False, f"Unsupported file format: {file_ext}"

    return True, None


def process_upload(uploaded_file, target_dir: Path, show_success: bool = True) -> bool:
    """
    Process and save uploaded file.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object.
    target_dir : Path
        Target directory for saving the file.
    show_success : bool, optional
        Whether to display success message (default: True).

    Returns
    -------
    bool
        True if upload successful, False otherwise.
    """

    try:
        # Create target directory if doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate file path
        file_path = target_dir / uploaded_file.name

        # Handle file conflicts
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = file_path.stem
            suffix = file_path.suffix
            file_path = target_dir / f"{stem}_{timestamp}{suffix}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"File uploaded successfully: {file_path}")

        if show_success:
            st.success(f"✅ {uploaded_file.name} uploaded successfully")

        return True

    except Exception as e:
        logger.error(f"Upload error: {e}")
        st.error(f"⚠️ Upload failed: {str(e)}")
        return False


# ============================================================================
# Tab 3: Browse Data
# ============================================================================

def render_browse_tab() -> None:
    """
    Render the data browsing and preview interface.

    Displays all available data files with metadata, preview capabilities,
    and dataset activation functionality.
    """

    render_section_header("Browse Local Data", "📁")

    local_data_dir = DataConfig.get_active_local_root()

    # Scan for research data files
    research_files = DataLoader.scan_research_data(str(local_data_dir))
    data_files = research_files.get('data', [])

    # Display file statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Data Files", len(data_files), help="CSV, Excel, TSV files")
    with col2:
        st.metric("Images", len(research_files.get('images', [])), help="Microscopy and analysis images")
    with col3:
        st.metric("Sequences", len(research_files.get('sequences', [])), help="FASTA, FASTQ, GenBank files")
    with col4:
        other_count = (len(research_files.get('results', [])) +
                       len(research_files.get('notebooks', [])) +
                       len(research_files.get('models', [])))
        st.metric("Other Files", other_count, help="Models, notebooks, reports")

    if not data_files:
        st.markdown("---")
        render_info_box(
            content="""
            **No data files found in the local directory.**

            You can:
            - Download shared datasets from the **Cloud Repository** tab
            - Upload your own data files in the **Upload Data** tab
            - Check that your data directory is correctly configured
            """,
            box_type="info",
            title="No Data Available"
        )
        return

    st.markdown("---")

    # File selection and preview
    render_section_header("File Explorer", "🗂️")

    # Create file options with metadata
    file_options = {}
    for file_path in data_files:
        metadata = get_file_metadata(file_path)
        if metadata:
            display_name = f"{metadata['name']} ({metadata['size']}) - {metadata['modified']}"
            file_options[file_path] = display_name

    selected_file = st.selectbox(
        "Select a file to preview and manage",
        options=list(file_options.keys()),
        format_func=lambda x: file_options.get(x, x),
        help="Choose a file to view its contents and set as active dataset"
    )

    if selected_file:
        render_file_details(selected_file)


def render_file_details(file_path: str) -> None:
    """
    Render detailed view of selected file.

    Parameters
    ----------
    file_path : str
        Path to the selected file.
    """

    col1, col2 = st.columns([1, 2])

    with col1:
        render_file_info_card(file_path)

    with col2:
        render_file_preview(file_path)


def render_file_info_card(file_path: str) -> None:
    """
    Render file information and action buttons.

    Parameters
    ----------
    file_path : str
        Path to the file.
    """

    render_section_header("File Operations", "🛠️")

    metadata = get_file_metadata(file_path)

    if not metadata:
        st.error("Unable to retrieve file metadata")
        return

    # Display metadata
    info_items = [
        ("Name", metadata['name']),
        ("Size", metadata['size']),
        ("Type", metadata['type']),
        ("Category", metadata['category'].title()),
        ("Modified", metadata['modified'])
    ]

    for label, value in info_items:
        st.markdown(f"**{label}:** {value}")

    st.markdown("---")

    # Action buttons
    if st.button(
            "✅ Set as Active Dataset",
            type="primary",
            use_container_width=True,
            help="Load this file as the active dataset for analysis"
    ):
        activate_dataset(file_path)

    if st.button(
            "📥 Download File",
            use_container_width=True,
            help="Download this file to your computer"
    ):
        with open(file_path, "rb") as f:
            st.download_button(
                label="Click to Download",
                data=f,
                file_name=Path(file_path).name,
                mime="application/octet-stream"
            )

    if st.button(
            "🗑️ Delete File",
            use_container_width=True,
            help="Permanently delete this file"
    ):
        if 'confirm_delete' not in st.session_state:
            st.session_state.confirm_delete = True
            st.warning("⚠️ Click again to confirm deletion")
        else:
            try:
                os.remove(file_path)
                del st.session_state.confirm_delete
                st.success("✅ File deleted successfully")
                st.rerun()
            except Exception as e:
                logger.error(f"Deletion error: {e}")
                st.error(f"⚠️ Deletion failed: {str(e)}")


def render_file_preview(file_path: str) -> None:
    """
    Render data preview for supported file formats.

    Parameters
    ----------
    file_path : str
        Path to the file.
    """

    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name

    render_section_header(f"Data Preview: {file_name}", "👁️")

    if file_ext not in SUPPORTED_PREVIEW_FORMATS:
        st.info(f"Preview not available for {file_ext} files")
        return

    try:
        # Load data
        df = DataLoader.load_file(file_path)

        if df is None:
            st.error("Unable to load file data")
            return

        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing:,}")
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("Duplicates", f"{duplicates:,}")

        # Data preview
        st.markdown("#### Data Sample")
        st.dataframe(
            df.head(PREVIEW_ROW_LIMIT),
            use_container_width=True,
            height=400
        )

        # Column information
        with st.expander("📊 Column Information", expanded=False):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                'Unique': df.nunique().values
            })
            st.dataframe(col_info, use_container_width=True)

        # Basic statistics
        with st.expander("📈 Descriptive Statistics", expanded=False):
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns for statistical summary")

    except Exception as e:
        logger.error(f"Preview error: {e}")
        st.error(f"⚠️ Preview failed: {str(e)}")


def activate_dataset(file_path: str) -> None:
    """
    Load file and set as active dataset.

    Parameters
    ----------
    file_path : str
        Path to the file to activate.
    """

    try:
        df = DataLoader.load_file(file_path)

        if df is None:
            st.error("⚠️ Unable to load file")
            return

        DataManager.set_active_dataset(
            df,
            Path(file_path).name,
            file_path
        )

        st.success("🎉 Dataset activated successfully! Navigate to Analysis Modules to begin analysis.")
        st.balloons()
        st.rerun()

    except Exception as e:
        logger.error(f"Dataset activation error: {e}")
        st.error(f"⚠️ Activation failed: {str(e)}")


# ============================================================================
# Tab 4: Data Management
# ============================================================================

def render_management_tab() -> None:
    """
    Render the data management interface.

    Displays active dataset information, cache management, and
    directory statistics.
    """

    render_section_header("Active Dataset Management", "⚙️")

    # Active dataset information
    if DataManager.validate_dataset():
        render_active_dataset_info()
    else:
        render_info_box(
            content="""
            **No active dataset currently loaded.**

            To activate a dataset:
            1. Go to the **Browse Data** tab
            2. Select a file from the list
            3. Click "Set as Active Dataset"
            """,
            box_type="info",
            title="No Active Dataset"
        )

    st.markdown("---")

    # Directory management
    render_directory_management()


def render_active_dataset_info() -> None:
    """Render active dataset information and management controls."""

    st.success("**✅ Active Dataset Loaded**")

    dataset_info = DataManager.get_dataset_info()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Name", dataset_info['name'])
    with col2:
        st.metric("Dimensions", f"{dataset_info['shape'][0]:,} × {dataset_info['shape'][1]}")
    with col3:
        st.metric("Memory Usage", dataset_info['memory_usage'])
    with col4:
        file_path = dataset_info.get('file_path', 'Unknown')
        display_path = ("..." + file_path[-30:]) if len(file_path) > 30 else file_path
        st.metric("Source", display_path)

    # Management buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Clear Active Dataset", use_container_width=True):
            DataManager.clear_active_dataset()
            st.success("Active dataset cleared")
            st.rerun()

    with col2:
        if st.button("🔄 Reload Dataset", use_container_width=True):
            file_path = dataset_info.get('file_path')
            if file_path and os.path.exists(file_path):
                activate_dataset(file_path)
            else:
                st.error("Source file not found")


def render_directory_management() -> None:
    """Render directory statistics and management tools."""

    render_section_header("Directory Management", "📂")

    local_data_dir = DataConfig.get_active_local_root()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Data Directory:** `{local_data_dir}`")

        if local_data_dir.exists():
            stats = calculate_directory_stats(str(local_data_dir))

            st.markdown(f"**Total Size:** {stats['total_size_formatted']}")
            st.markdown(f"**Total Files:** {stats['file_count']:,}")

            if stats['type_distribution']:
                with st.expander("📊 File Type Distribution"):
                    for ext, count in sorted(stats['type_distribution'].items()):
                        st.write(f"- {ext}: {count} file(s)")
        else:
            st.warning("⚠️ Directory does not exist")
            if st.button("Create Directory"):
                local_data_dir.mkdir(parents=True, exist_ok=True)
                st.success("Directory created")
                st.rerun()

    with col2:
        st.markdown("#### Quick Actions")

        if st.button("🔄 Refresh Directory Scan", use_container_width=True):
            st.rerun()

        if st.button("📁 Open Directory", use_container_width=True):
            try:
                if sys.platform == 'win32':
                    os.startfile(local_data_dir)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{local_data_dir}"')
                else:  # linux
                    os.system(f'xdg-open "{local_data_dir}"')
                st.success("Directory opened")
            except Exception as e:
                st.error(f"Unable to open directory: {e}")
                st.info(f"Manual path: {local_data_dir}")


# ============================================================================
# Custom Styling
# ============================================================================

# ============================================================================
# Custom Styling
# ============================================================================

def apply_custom_styles() -> None:
    """Apply custom CSS styles for enhanced UI."""

    st.markdown("""
    <style>
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2E86AB, #A9D6E5);
            color: white;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 600;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1.1rem;
        }

        /* Button styling */
        .stButton > button {
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* File uploader styling */
        [data-testid="stFileUploader"] {
            border: 2px dashed #2E86AB;
            border-radius: 8px;
            padding: 2rem;
            background: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Main Page Execution
# ============================================================================

def main():
    """
    Main function to render the Data Management Hub page.

    Orchestrates all tab components and applies custom styling for
    a cohesive user experience.
    """

    # Render sidebar
    render_sidebar()

    # Render page header
    render_page_header(
        title="Data Management Hub",
        icon="📂",
        subtitle="Comprehensive data access, upload, and organization"
    )

    # Apply custom styling
    apply_custom_styles()

    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Download Data",
        "📤 Upload Data",
        "📁 Browse Data",
        "⚙️ Manage Data"
    ])

    with tab1:
        render_download_tab()

    with tab2:
        render_upload_tab()

    with tab3:
        render_browse_tab()

    with tab4:
        render_management_tab()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
