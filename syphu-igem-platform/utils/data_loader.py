# utils/data_loader.py
"""
Data Loading Utilities - SYPHU iGEM Research Platform
======================================================

Comprehensive data loading and validation utilities supporting multiple
scientific data formats including tabular data, images, sequences, and
specialized formats like AnnData (h5ad) for single-cell analysis.

Author: SYPHU-CHINA iGEM Team
Date: 2025-10-01
License: MIT

Dependencies:
    - pandas >= 1.5.0
    - numpy >= 1.20.0
    - openpyxl >= 3.0.0 (for Excel files)
    - anndata >= 0.8.0 (optional, for .h5ad files)
    - pathlib (standard library)

Notes
-----
Supports various file formats common in biological research:
- Tabular: CSV, TSV, Excel, JSON
- Single-cell: AnnData (.h5ad)
- Sequences: FASTA, FASTQ, GenBank
- Images: PNG, JPG, TIFF, SVG
- Models: Pickle, Joblib, HDF5
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# Constants
# ============================================================================

# Supported file format categories
EXTENSION_MAP = {
    'data': {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.json', '.h5ad', '.parquet'},
    'images': {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.svg', '.nd2', '.czi', '.lif'},
    'sequences': {'.fasta', '.fa', '.fastq', '.fq', '.gb', '.gbk', '.gff', '.vcf'},
    'results': {'.pdf', '.html', '.doc', '.docx', '.txt', '.md'},
    'notebooks': {'.ipynb', '.py', '.r', '.rmd'},
    'models': {'.pkl', '.joblib', '.h5', '.hdf5', '.pt', '.pth'}
}

# Directories to skip during scanning
SKIP_DIRS = {
    '.ipynb_checkpoints', '__pycache__', '.git', '.vscode',
    '.DS_Store', 'node_modules', '.idea', '.pytest_cache'
}

# Maximum file size for loading (in MB)
MAX_FILE_SIZE_MB = 500


# ============================================================================
# Main DataLoader Class
# ============================================================================

class DataLoader:
    """
    Data loading utility class for various scientific data formats.

    This class provides static methods for:
    - Scanning research data directories
    - Loading files in multiple formats
    - Validating data availability
    - File format detection and conversion

    Methods
    -------
    scan_research_data(data_dir: str) -> Dict[str, List[str]]
        Scan and categorize research data files.
    check_data_availability(data_dir: str) -> Tuple[bool, int]
        Check if data files exist in directory.
    load_file(file_path: str) -> Optional[pd.DataFrame]
        Load file and return as DataFrame when possible.
    get_file_info(file_path: str) -> Dict[str, Any]
        Get metadata about a file.
    validate_file(file_path: str) -> Tuple[bool, Optional[str]]
        Validate file before loading.

    Examples
    --------
    >>> loader = DataLoader()
    >>> files = loader.scan_research_data("./data")
    >>> print(f"Found {len(files['data'])} data files")

    >>> df = loader.load_file("data/experiment.csv")
    >>> print(df.shape)
    (100, 5)
    """

    @staticmethod
    def scan_research_data(data_dir: str = "./syphu-china-model") -> Dict[str, List[str]]:
        """
        Scan research data directory and categorize files by type.

        Parameters
        ----------
        data_dir : str, optional
            Path to data directory (default: "./syphu-china-model").

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping categories to file path lists.
            Categories: 'data', 'images', 'sequences', 'results', 'notebooks', 'models'

        Notes
        -----
        - Recursively scans all subdirectories
        - Skips hidden and system directories
        - Only includes readable files
        - Categorizes by file extension

        Examples
        --------
        >>> files = DataLoader.scan_research_data("./data")
        >>> print(f"Data files: {len(files['data'])}")
        >>> print(f"Images: {len(files['images'])}")
        """
        abs_path = os.path.abspath(data_dir)

        # Initialize empty result structure
        research_files = {category: [] for category in EXTENSION_MAP.keys()}

        # Check if directory exists
        if not os.path.exists(abs_path):
            logger.warning(f"Directory does not exist: {abs_path}")
            return research_files

        try:
            # Walk through directory tree
            for root, dirs, filenames in os.walk(abs_path):
                # Filter out directories to skip
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

                # Process each file
                for filename in filenames:
                    try:
                        file_path = os.path.join(root, filename)

                        # Skip if not readable
                        if not os.access(file_path, os.R_OK):
                            continue

                        # Get file extension
                        file_ext = Path(filename).suffix.lower()

                        # Categorize file
                        for category, extensions in EXTENSION_MAP.items():
                            if file_ext in extensions:
                                research_files[category].append(file_path)
                                break

                    except (PermissionError, OSError) as e:
                        logger.debug(f"Skipping file {filename}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error scanning directory {abs_path}: {str(e)}")

        # Log summary
        total_files = sum(len(files) for files in research_files.values())
        logger.info(f"Scanned {abs_path}: found {total_files} files")

        return research_files

    @staticmethod
    def check_data_availability(data_dir: str = "./syphu-china-model") -> Tuple[bool, int]:
        """
        Check if data files exist in specified directory.

        Parameters
        ----------
        data_dir : str, optional
            Path to data directory (default: "./syphu-china-model").

        Returns
        -------
        Tuple[bool, int]
            (has_data, file_count) - whether data exists and total file count.

        Examples
        --------
        >>> has_data, count = DataLoader.check_data_availability("./data")
        >>> if has_data:
        ...     print(f"Found {count} files")
        Found 15 files
        """
        if not os.path.exists(data_dir):
            logger.info(f"Directory does not exist: {data_dir}")
            return False, 0

        try:
            research_files = DataLoader.scan_research_data(data_dir)
            total_files = sum(len(files) for files in research_files.values())
            has_data = total_files > 0

            logger.info(f"Data availability check: {'Available' if has_data else 'No data'} ({total_files} files)")
            return has_data, total_files

        except Exception as e:
            logger.error(f"Error checking data availability: {str(e)}")
            return False, 0

    @staticmethod
    def load_file(file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data file into pandas DataFrame.

        Supports multiple formats with automatic detection:
        - CSV, TSV: Comma/tab-separated values
        - Excel: .xlsx, .xls files
        - JSON: Structured data
        - H5AD: AnnData single-cell format
        - Parquet: Columnar data format

        Parameters
        ----------
        file_path : str
            Path to data file.

        Returns
        -------
        pd.DataFrame or None
            Loaded data as DataFrame, or None if loading fails.

        Examples
        --------
        >>> df = DataLoader.load_file("data/experiment.csv")
        >>> print(df.shape)
        (100, 5)

        >>> df = DataLoader.load_file("data/single_cell.h5ad")
        >>> print(df.columns)  # Cell metadata columns

        Notes
        -----
        - Automatically handles encoding issues
        - Validates file size before loading
        - Provides detailed error messages
        - For .h5ad files, returns cell metadata (not full expression matrix)
        """
        try:
            # Validate file
            is_valid, error_msg = DataLoader.validate_file(file_path)
            if not is_valid:
                logger.error(f"File validation failed: {error_msg}")
                return None

            # Get file extension
            file_ext = Path(file_path).suffix.lower()

            # Load based on format
            if file_ext == '.csv':
                return pd.read_csv(
                    file_path,
                    encoding='utf-8',
                    encoding_errors='ignore',
                    low_memory=False
                )

            elif file_ext in ['.tsv', '.txt']:
                # Try tab-separated first
                try:
                    return pd.read_csv(
                        file_path,
                        sep='\t',
                        encoding='utf-8',
                        encoding_errors='ignore',
                        low_memory=False
                    )
                except:
                    # Fallback to comma-separated
                    return pd.read_csv(
                        file_path,
                        encoding='utf-8',
                        encoding_errors='ignore',
                        low_memory=False
                    )

            elif file_ext == '.xlsx':
                return pd.read_excel(file_path, engine='openpyxl')

            elif file_ext == '.xls':
                return pd.read_excel(file_path, engine='xlrd')

            elif file_ext == '.json':
                return pd.read_json(file_path)

            elif file_ext == '.h5ad':
                return DataLoader._load_anndata(file_path)

            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)

            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return None

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    @staticmethod
    def _load_anndata(file_path: str) -> Optional[pd.DataFrame]:
        """
        Load AnnData (.h5ad) file and convert to DataFrame.

        Parameters
        ----------
        file_path : str
            Path to .h5ad file.

        Returns
        -------
        pd.DataFrame or None
            Cell metadata with expression statistics.

        Notes
        -----
        AnnData structure:
        - adata.X: Expression matrix (cells × genes)
        - adata.obs: Cell metadata (observations)
        - adata.var: Gene metadata (variables)

        This function returns adata.obs enriched with:
        - total_counts: Sum of expression per cell
        - n_genes_detected: Number of detected genes per cell
        - (Optionally) Top variable genes as columns

        For full single-cell analysis, use specialized tools like scanpy.

        Examples
        --------
        >>> df = DataLoader._load_anndata("pbmc3k.h5ad")
        >>> print(df.columns)
        Index(['n_genes', 'n_counts', 'total_counts', 'n_genes_detected'], dtype='object')
        """
        try:
            # Check if anndata is installed
            try:
                import anndata
            except ImportError:
                logger.error(
                    "anndata library not installed. "
                    "Install with: pip install anndata"
                )
                return None

            # Load AnnData object
            logger.info(f"Loading AnnData file: {file_path}")
            adata = anndata.read_h5ad(file_path)

            logger.info(
                f"Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes"
            )

            # Start with cell metadata
            df = adata.obs.copy()

            # Add expression statistics
            if hasattr(adata, 'X') and adata.X is not None:
                try:
                    # Convert to array if sparse
                    if hasattr(adata.X, 'toarray'):
                        X_array = adata.X.toarray()
                    else:
                        X_array = adata.X

                    # Calculate statistics
                    df['total_counts'] = np.array(X_array.sum(axis=1)).flatten()
                    df['n_genes_detected'] = np.array((X_array > 0).sum(axis=1)).flatten()

                    # For small datasets, add gene expression
                    if adata.n_vars <= 100:
                        gene_df = pd.DataFrame(
                            X_array,
                            columns=adata.var_names,
                            index=adata.obs_names
                        )
                        df = pd.concat([df, gene_df], axis=1)
                        logger.info("Added gene expression columns (n_vars ≤ 100)")

                except Exception as e:
                    logger.warning(f"Could not add expression statistics: {str(e)}")

            logger.info(f"Converted to DataFrame: {df.shape[0]} × {df.shape[1]}")
            return df

        except Exception as e:
            logger.error(f"Error loading AnnData file: {str(e)}")
            return None

    @staticmethod
    def validate_file(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file before loading.

        Checks:
        - File exists
        - File is readable
        - File size is within limits
        - File format is supported

        Parameters
        ----------
        file_path : str
            Path to file.

        Returns
        -------
        Tuple[bool, str or None]
            (is_valid, error_message). error_message is None if valid.

        Examples
        --------
        >>> is_valid, error = DataLoader.validate_file("data.csv")
        >>> if not is_valid:
        ...     print(f"Validation failed: {error}")
        """
        path = Path(file_path)

        # Check existence
        if not path.exists():
            return False, f"File does not exist: {file_path}"

        # Check if it's a file
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"

        # Check readability
        if not os.access(file_path, os.R_OK):
            return False, f"File is not readable: {file_path}"

        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return False, f"File too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)"

        # Check format
        ext = path.suffix.lower()
        all_extensions = set()
        for exts in EXTENSION_MAP.values():
            all_extensions.update(exts)

        if ext not in all_extensions:
            return False, f"Unsupported file format: {ext}"

        return True, None

    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata about a file.

        Parameters
        ----------
        file_path : str
            Path to file.

        Returns
        -------
        Dict[str, Any]
            Dictionary with file metadata:
            - name: filename
            - size: file size in MB
            - extension: file extension
            - category: data category
            - readable: whether file is readable
            - modified: last modification time

        Examples
        --------
        >>> info = DataLoader.get_file_info("data/experiment.csv")
        >>> print(f"Size: {info['size']:.2f} MB")
        >>> print(f"Category: {info['category']}")
        """
        try:
            path = Path(file_path)

            # Basic info
            info = {
                'name': path.name,
                'path': str(path.absolute()),
                'extension': path.suffix.lower(),
                'size_bytes': path.stat().st_size,
                'size_mb': path.stat().st_size / (1024 * 1024),
                'readable': os.access(file_path, os.R_OK),
                'exists': path.exists(),
                'modified': path.stat().st_mtime
            }

            # Determine category
            ext = info['extension']
            for category, extensions in EXTENSION_MAP.items():
                if ext in extensions:
                    info['category'] = category
                    break
            else:
                info['category'] = 'unknown'

            return info

        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {
                'name': Path(file_path).name,
                'error': str(e)
            }

    @staticmethod
    def get_preview(file_path: str, n_rows: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get preview of data file content.

        Parameters
        ----------
        file_path : str
            Path to file.
        n_rows : int, optional
            Number of rows to preview (default: 10).

        Returns
        -------
        Dict[str, Any] or None
            Dictionary with:
            - shape: (n_rows, n_cols)
            - columns: column names
            - dtypes: data types
            - preview: first n rows as DataFrame
            - missing: missing value counts
            - memory: memory usage

        Examples
        --------
        >>> preview = DataLoader.get_preview("data.csv", n_rows=5)
        >>> print(preview['shape'])
        (100, 5)
        >>> print(preview['preview'])  # First 5 rows
        """
        try:
            df = DataLoader.load_file(file_path)

            if df is None:
                return None

            return {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'preview': df.head(n_rows),
                'missing': df.isnull().sum().to_dict(),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Error getting file preview: {str(e)}")
            return None


# ============================================================================
# Utility Functions
# ============================================================================

def get_supported_formats() -> List[str]:
    """
    Get list of all supported file formats.

    Returns
    -------
    List[str]
        Sorted list of supported file extensions.
    """
    all_formats = set()
    for extensions in EXTENSION_MAP.values():
        all_formats.update(extensions)
    return sorted(list(all_formats))


def get_format_category(file_path: str) -> Optional[str]:
    """
    Determine category of file based on extension.

    Parameters
    ----------
    file_path : str
        Path to file.

    Returns
    -------
    str or None
        Category name ('data', 'images', etc.) or None if unknown.
    """
    ext = Path(file_path).suffix.lower()

    for category, extensions in EXTENSION_MAP.items():
        if ext in extensions:
            return category

    return None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'DataLoader',
    'get_supported_formats',
    'get_format_category',
    'EXTENSION_MAP',
    'MAX_FILE_SIZE_MB'
]
