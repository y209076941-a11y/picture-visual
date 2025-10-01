# config/data_config.py
"""
Data Configuration Management Module
=====================================

This module provides centralized configuration for data source management,
including local file system scanning and Google Drive integration for
collaborative scientific research projects.

Author: SYPHU-CHINA iGEM Team
Date: 2025-10-01
License: MIT

Dependencies:
    - pathlib (standard library)
    - typing (standard library)
    - os (standard library)

Notes
-----
This configuration module supports both local data access and cloud-based
data sharing, facilitating reproducible research workflows and multi-site
collaboration as recommended by Nature Data standards.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Source Configuration Class
# ============================================================================

class DataConfig:
    """
    Centralized configuration for data management and file handling.

    This class manages both local file system data sources and remote
    cloud storage integration (Google Drive), providing a unified interface
    for data access across different storage backends.

    Attributes
    ----------
    LOCAL_PROJECT_ROOTS : List[str]
        Ordered list of potential local project root directories.
        The first existing directory will be used as the active root.
    GOOGLE_DRIVE_DATASETS : Dict[str, Dict[str, str]]
        Configuration for publicly shared datasets on Google Drive.
    SUPPORTED_EXTENSIONS : Dict[str, List[str]]
        File extension categories for scientific data management.

    Methods
    -------
    get_active_local_root()
        Identify and return the active local data directory.
    get_supported_extensions_list()
        Return flat list of all supported file extensions.
    get_google_drive_download_url(file_id)
        Generate direct download URL for Google Drive files.
    validate_file_path(filepath)
        Check if file path exists and has supported extension.
    get_file_category(filepath)
        Determine file category based on extension.

    Examples
    --------
    >>> config = DataConfig()
    >>> root = config.get_active_local_root()
    >>> extensions = config.get_supported_extensions_list()
    >>> url = config.get_google_drive_download_url("1ABC123")

    Notes
    -----
    - Local paths are checked in order of priority
    - Google Drive IDs must be updated with actual shared file IDs
    - File type categories follow standard bioinformatics conventions
    - All paths use OS-independent Path objects for cross-platform compatibility

    See Also
    --------
    pathlib.Path : Object-oriented filesystem paths
    """

    # ========================================================================
    # Local File System Configuration
    # ========================================================================

    LOCAL_PROJECT_ROOTS: List[str] = [
        r"C:\Users\Administrator\PycharmProjects\picture\syphu-igem-platform\syphu-china-model",
        "./syphu-china-model",  # Relative path for portability
        str(Path.home() / "syphu-china-model"),  # User home directory fallback
    ]

    # Default directories for data organization
    DEFAULT_DATA_STRUCTURE: Dict[str, str] = {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "results": "results",
        "figures": "figures",
        "models": "models",
        "sequences": "sequences",
        "images": "images",
        "metadata": "metadata"
    }

    # ========================================================================
    # Cloud Storage Configuration (Google Drive)
    # ========================================================================

    GOOGLE_DRIVE_DATASETS: Dict[str, Dict[str, str]] = {
        "cell_track_data": {
            "name": "Cell Tracking Dataset",
            "drive_id": "REPLACE_WITH_ACTUAL_DRIVE_ID",  # Must be updated
            "description": "Cell movement trajectory and morphology time-series data",
            "file_type": "csv",
            "size_mb": "~50",
            "last_updated": "2025-01-15"
        },
        "plasmid_analysis": {
            "name": "Plasmid Construction Analysis",
            "drive_id": "REPLACE_WITH_ACTUAL_DRIVE_ID",
            "description": "Plasmid design, assembly, and validation results",
            "file_type": "xlsx",
            "size_mb": "~15",
            "last_updated": "2025-01-20"
        },
        "microscopy_images": {
            "name": "Microscopy Image Collection",
            "drive_id": "REPLACE_WITH_ACTUAL_DRIVE_ID",
            "description": "High-resolution cellular imaging dataset with annotations",
            "file_type": "zip",
            "size_mb": "~500",
            "last_updated": "2025-01-25"
        },
        "sequencing_data": {
            "name": "NGS Sequencing Results",
            "drive_id": "REPLACE_WITH_ACTUAL_DRIVE_ID",
            "description": "Next-generation sequencing data (FASTQ format)",
            "file_type": "fastq.gz",
            "size_mb": "~1000",
            "last_updated": "2025-02-01"
        }
    }

    # ========================================================================
    # File Type Classification
    # ========================================================================

    SUPPORTED_EXTENSIONS: Dict[str, List[str]] = {
        'tabular_data': [
            '.csv',  # Comma-separated values
            '.tsv',  # Tab-separated values
            '.txt',  # Plain text data
            '.xlsx',  # Excel workbook
            '.xls',  # Legacy Excel format
            '.json',  # JSON data format
            '.h5ad',  # AnnData HDF5 format (single-cell)
            '.parquet'  # Apache Parquet columnar format
        ],
        'image_data': [
            '.jpg', '.jpeg',  # JPEG images
            '.png',  # PNG images
            '.tif', '.tiff',  # TIFF images (microscopy standard)
            '.bmp',  # Bitmap images
            '.svg',  # Scalable vector graphics
            '.nd2',  # Nikon microscopy format
            '.czi',  # Zeiss microscopy format
            '.lif'  # Leica microscopy format
        ],
        'sequence_data': [
            '.fasta', '.fa',  # FASTA sequence format
            '.fastq', '.fq',  # FASTQ with quality scores
            '.gb', '.gbk',  # GenBank format
            '.gff', '.gff3',  # Gene feature format
            '.vcf',  # Variant call format
            '.sam', '.bam',  # Sequence alignment format
            '.ab1'  # ABI sequencing trace
        ],
        'model_files': [
            '.pkl',  # Python pickle
            '.joblib',  # Joblib serialization
            '.h5',  # HDF5 format
            '.hdf5',  # HDF5 format
            '.pt',  # PyTorch model
            '.pth',  # PyTorch checkpoint
            '.ckpt',  # TensorFlow checkpoint
            '.pb'  # Protocol buffer (TensorFlow)
        ],
        'code_notebooks': [
            '.py',  # Python scripts
            '.ipynb',  # Jupyter notebooks
            '.r',  # R scripts
            '.rmd',  # R Markdown
            '.md',  # Markdown documentation
            '.sh',  # Shell scripts
            '.yaml', '.yml'  # Configuration files
        ],
        'documentation': [
            '.pdf',  # PDF documents
            '.html',  # HTML reports
            '.doc',  # Word document
            '.docx',  # Word document (modern)
            '.tex',  # LaTeX source
            '.rtf'  # Rich text format
        ]
    }

    # Maximum file size limits (in MB) for different categories
    FILE_SIZE_LIMITS: Dict[str, int] = {
        'tabular_data': 500,
        'image_data': 1000,
        'sequence_data': 5000,
        'model_files': 1000,
        'code_notebooks': 50,
        'documentation': 100
    }

    # ========================================================================
    # Class Methods - Local File System Management
    # ========================================================================

    @classmethod
    def get_active_local_root(cls) -> Path:
        """
        Identify and return the first existing local project root directory.

        Searches through LOCAL_PROJECT_ROOTS in order and returns the first
        directory that exists. If none exist, creates and returns the relative
        path directory.

        Returns
        -------
        Path
            Path object pointing to the active project root directory.

        Raises
        ------
        OSError
            If directory creation fails when no existing roots are found.

        Examples
        --------
        >>> root = DataConfig.get_active_local_root()
        >>> print(root.exists())
        True

        Notes
        -----
        This method ensures cross-platform compatibility by using pathlib.Path
        and automatically creates necessary directories with appropriate permissions.
        """

        for root_str in cls.LOCAL_PROJECT_ROOTS:
            root_path = Path(root_str).resolve()
            if root_path.exists():
                logger.info(f"Using existing project root: {root_path}")
                return root_path

        # Create default directory if none exist
        default_path = Path("./syphu-china-model").resolve()
        try:
            default_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new project root: {default_path}")
            return default_path
        except OSError as e:
            logger.error(f"Failed to create project root: {e}")
            raise

    @classmethod
    def initialize_data_structure(cls, root_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        Create standardized directory structure for data organization.

        Parameters
        ----------
        root_path : Path, optional
            Root directory for data structure. If None, uses active local root.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping directory names to their Path objects.

        Examples
        --------
        >>> paths = DataConfig.initialize_data_structure()
        >>> print(paths['raw_data'])
        PosixPath('syphu-china-model/data/raw')

        Notes
        -----
        Creates directories following FAIR data principles for reproducibility.
        """

        if root_path is None:
            root_path = cls.get_active_local_root()

        created_paths = {}
        for key, rel_path in cls.DEFAULT_DATA_STRUCTURE.items():
            full_path = root_path / rel_path
            full_path.mkdir(parents=True, exist_ok=True)
            created_paths[key] = full_path
            logger.info(f"Initialized directory: {full_path}")

        return created_paths

    # ========================================================================
    # Class Methods - File Extension Management
    # ========================================================================

    @classmethod
    def get_supported_extensions_list(cls) -> List[str]:
        """
        Return flattened list of all supported file extensions.

        Returns
        -------
        List[str]
            Alphabetically sorted list of supported file extensions.

        Examples
        --------
        >>> extensions = DataConfig.get_supported_extensions_list()
        >>> '.csv' in extensions
        True
        """

        all_extensions: Set[str] = set()
        for category_extensions in cls.SUPPORTED_EXTENSIONS.values():
            all_extensions.update(category_extensions)
        return sorted(list(all_extensions))

    @classmethod
    def get_file_category(cls, filepath: str) -> Optional[str]:
        """
        Determine file category based on extension.

        Parameters
        ----------
        filepath : str
            Path to file or filename with extension.

        Returns
        -------
        str or None
            Category name if extension is supported, None otherwise.

        Examples
        --------
        >>> DataConfig.get_file_category("data.csv")
        'tabular_data'
        >>> DataConfig.get_file_category("image.tiff")
        'image_data'
        """

        file_ext = Path(filepath).suffix.lower()
        for category, extensions in cls.SUPPORTED_EXTENSIONS.items():
            if file_ext in extensions:
                return category
        return None

    @classmethod
    def validate_file_path(cls, filepath: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file path existence and extension support.

        Parameters
        ----------
        filepath : str
            Path to validate.

        Returns
        -------
        Tuple[bool, str or None]
            (is_valid, error_message). error_message is None if valid.

        Examples
        --------
        >>> valid, error = DataConfig.validate_file_path("data.csv")
        >>> print(valid)
        True
        """

        path = Path(filepath)

        # Check existence
        if not path.exists():
            return False, f"File does not exist: {filepath}"

        # Check extension
        if path.suffix.lower() not in cls.get_supported_extensions_list():
            return False, f"Unsupported file extension: {path.suffix}"

        # Check file size
        category = cls.get_file_category(filepath)
        if category:
            size_mb = path.stat().st_size / (1024 * 1024)
            limit = cls.FILE_SIZE_LIMITS.get(category, float('inf'))
            if size_mb > limit:
                return False, f"File exceeds size limit ({size_mb:.1f}MB > {limit}MB)"

        return True, None

    # ========================================================================
    # Class Methods - Google Drive Integration
    # ========================================================================

    @classmethod
    def get_google_drive_download_url(cls, file_id: str) -> str:
        """
        Generate direct download URL for Google Drive files.

        Parameters
        ----------
        file_id : str
            Google Drive file ID (found in sharing link).

        Returns
        -------
        str
            Direct download URL for the file.

        Examples
        --------
        >>> url = DataConfig.get_google_drive_download_url("1ABC123")
        >>> print(url)
        https://drive.google.com/uc?export=download&id=1ABC123

        Notes
        -----
        This URL format works for files < 100MB. Larger files may require
        additional confirmation steps handled by download utilities.
        """

        return f"https://drive.google.com/uc?export=download&id={file_id}"

    @classmethod
    def get_dataset_info(cls, dataset_key: str) -> Optional[Dict[str, str]]:
        """
        Retrieve metadata for a specific Google Drive dataset.

        Parameters
        ----------
        dataset_key : str
            Key identifier for the dataset in GOOGLE_DRIVE_DATASETS.

        Returns
        -------
        Dict[str, str] or None
            Dataset metadata dictionary, or None if key not found.

        Examples
        --------
        >>> info = DataConfig.get_dataset_info("cell_track_data")
        >>> print(info['name'])
        Cell Tracking Dataset
        """

        return cls.GOOGLE_DRIVE_DATASETS.get(dataset_key)

    @classmethod
    def list_available_datasets(cls) -> List[Dict[str, str]]:
        """
        List all available Google Drive datasets with metadata.

        Returns
        -------
        List[Dict[str, str]]
            List of dataset metadata dictionaries.

        Examples
        --------
        >>> datasets = DataConfig.list_available_datasets()
        >>> for ds in datasets:
        ...     print(ds['name'], ds['size_mb'])
        """

        return [
            {**info, 'key': key}
            for key, info in cls.GOOGLE_DRIVE_DATASETS.items()
        ]


# ============================================================================
# Module-Level Utility Functions
# ============================================================================

def get_data_root() -> Path:
    """
    Convenience function to get active data root directory.

    Returns
    -------
    Path
        Active project root directory path.
    """
    return DataConfig.get_active_local_root()


def is_supported_file(filepath: str) -> bool:
    """
    Quick check if file has supported extension.

    Parameters
    ----------
    filepath : str
        Path to check.

    Returns
    -------
    bool
        True if file extension is supported.
    """
    ext = Path(filepath).suffix.lower()
    return ext in DataConfig.get_supported_extensions_list()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'DataConfig',
    'get_data_root',
    'is_supported_file'
]
