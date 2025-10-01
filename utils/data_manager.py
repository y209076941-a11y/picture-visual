# utils/data_manager.py
"""
Data Management Utilities - SYPHU iGEM Research Platform
=========================================================

Session state management and data lifecycle control for the research platform.
Provides centralized management of active datasets, analysis results, and
experimental records throughout the user session.

Author: SYPHU-CHINA iGEM Team
Date: 2025-10-01
License: MIT

Dependencies:
    - streamlit >= 1.0.0
    - pandas >= 1.5.0
    - numpy >= 1.20.0

Notes
-----
This module manages the platform's session state, ensuring data persistence
across page navigation and providing thread-safe access to shared resources.
Follows best practices for Streamlit session state management.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Session state keys
SESSION_KEYS = {
    'dataset': 'current_dataset',
    'dataset_name': 'current_dataset_name',
    'dataset_path': 'current_dataset_path',
    'dataset_metadata': 'current_dataset_metadata',
    'data_analysis': 'current_data_analysis',
    'ml_results': 'ml_results',
    'image_analysis': 'analyzed_images',
    'bioinformatics': 'bioinformatics_results',
    'experiments': 'experiment_records',
    'analysis_history': 'analysis_history'
}

# Maximum dataset size (in MB)
MAX_DATASET_SIZE_MB = 500


# ============================================================================
# Main DataManager Class
# ============================================================================

class DataManager:
    """
    Centralized data management for the research platform.

    This class provides static methods for managing the platform's data
    lifecycle, including:
    - Session state initialization
    - Active dataset management
    - Dataset validation and metadata
    - Analysis results storage
    - Data persistence across pages

    All data is stored in Streamlit's session state for persistence
    across page navigation within a single user session.

    Methods
    -------
    initialize_session_state()
        Initialize all required session state variables.
    set_active_dataset(df, name, file_path=None)
        Set a dataset as the active dataset for analysis.
    get_active_dataset() -> pd.DataFrame
        Retrieve the current active dataset.
    validate_dataset() -> bool
        Check if a valid dataset is currently active.
    get_dataset_info() -> Dict[str, Any]
        Get comprehensive metadata about the active dataset.
    clear_active_dataset()
        Remove the active dataset from session state.
    save_analysis_result(analysis_type, result)
        Store analysis results in session state.
    get_analysis_history() -> List[Dict]
        Retrieve history of all analyses performed.
    export_session_data() -> Dict[str, Any]
        Export all session data for backup/download.

    Examples
    --------
    >>> DataManager.initialize_session_state()
    >>> DataManager.set_active_dataset(df, "experiment_data.csv")
    >>> if DataManager.validate_dataset():
    ...     info = DataManager.get_dataset_info()
    ...     print(f"Dataset: {info['name']}, Shape: {info['shape']}")
    """

    @staticmethod
    def initialize_session_state():
        """
        Initialize all required session state variables.

        Creates default empty values for all session state keys used by
        the platform. This method is idempotent - calling it multiple
        times is safe and will not overwrite existing data.

        Notes
        -----
        Should be called once at application startup (typically in app.py).
        Initializes storage for:
        - Active dataset and metadata
        - Analysis results (statistical, ML, image, bioinformatics)
        - Experiment records
        - Analysis history

        Examples
        --------
        >>> DataManager.initialize_session_state()
        >>> # Now safe to use other DataManager methods
        """
        try:
            # Active dataset
            if SESSION_KEYS['dataset'] not in st.session_state:
                st.session_state[SESSION_KEYS['dataset']] = None

            if SESSION_KEYS['dataset_name'] not in st.session_state:
                st.session_state[SESSION_KEYS['dataset_name']] = None

            if SESSION_KEYS['dataset_path'] not in st.session_state:
                st.session_state[SESSION_KEYS['dataset_path']] = None

            if SESSION_KEYS['dataset_metadata'] not in st.session_state:
                st.session_state[SESSION_KEYS['dataset_metadata']] = {}

            # Analysis results storage
            if SESSION_KEYS['data_analysis'] not in st.session_state:
                st.session_state[SESSION_KEYS['data_analysis']] = {}

            if SESSION_KEYS['ml_results'] not in st.session_state:
                st.session_state[SESSION_KEYS['ml_results']] = {}

            if SESSION_KEYS['image_analysis'] not in st.session_state:
                st.session_state[SESSION_KEYS['image_analysis']] = {}

            if SESSION_KEYS['bioinformatics'] not in st.session_state:
                st.session_state[SESSION_KEYS['bioinformatics']] = []

            if SESSION_KEYS['experiments'] not in st.session_state:
                st.session_state[SESSION_KEYS['experiments']] = {}

            if SESSION_KEYS['analysis_history'] not in st.session_state:
                st.session_state[SESSION_KEYS['analysis_history']] = []

            logger.info("Session state initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing session state: {str(e)}")
            raise

    @staticmethod
    def set_active_dataset(
            df: pd.DataFrame,
            name: str,
            file_path: Optional[str] = None
    ) -> bool:
        """
        Set a dataset as the active dataset for analysis.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to set as active.
        name : str
            Human-readable name for the dataset.
        file_path : str, optional
            Original file path if loaded from file.

        Returns
        -------
        bool
            True if successfully set, False otherwise.

        Notes
        -----
        - Validates dataset before setting
        - Calculates and stores comprehensive metadata
        - Logs dataset activation
        - Replaces any previously active dataset

        Examples
        --------
        >>> df = pd.read_csv("data.csv")
        >>> DataManager.set_active_dataset(df, "Experiment Data", "data.csv")
        True
        """
        try:
            # Validate input
            if df is None or df.empty:
                logger.error("Cannot set empty dataset as active")
                return False

            # Check size
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_mb > MAX_DATASET_SIZE_MB:
                logger.warning(
                    f"Dataset size ({memory_mb:.1f}MB) exceeds "
                    f"recommended limit ({MAX_DATASET_SIZE_MB}MB)"
                )

            # Set dataset
            st.session_state[SESSION_KEYS['dataset']] = df.copy()
            st.session_state[SESSION_KEYS['dataset_name']] = name
            st.session_state[SESSION_KEYS['dataset_path']] = file_path

            # Calculate and store metadata
            metadata = {
                'activation_time': datetime.now().isoformat(),
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'missing_values': df.isnull().sum().sum(),
                'memory_mb': memory_mb,
                'source_file': file_path or 'Unknown'
            }

            st.session_state[SESSION_KEYS['dataset_metadata']] = metadata

            logger.info(
                f"Dataset activated: '{name}' with shape {df.shape}, "
                f"memory: {memory_mb:.2f}MB"
            )

            return True

        except Exception as e:
            logger.error(f"Error setting active dataset: {str(e)}")
            return False

    @staticmethod
    def get_active_dataset() -> Optional[pd.DataFrame]:
        """
        Retrieve the current active dataset.

        Returns
        -------
        pd.DataFrame or None
            The active dataset, or None if no dataset is active.

        Notes
        -----
        Returns a copy to prevent unintended modifications to the
        session state dataset.

        Examples
        --------
        >>> df = DataManager.get_active_dataset()
        >>> if df is not None:
        ...     print(df.head())
        """
        try:
            dataset = st.session_state.get(SESSION_KEYS['dataset'])
            return dataset.copy() if dataset is not None else None
        except Exception as e:
            logger.error(f"Error getting active dataset: {str(e)}")
            return None

    @staticmethod
    def validate_dataset() -> bool:
        """
        Check if a valid dataset is currently active.

        Returns
        -------
        bool
            True if a valid dataset is active, False otherwise.

        Notes
        -----
        Validates that:
        - Dataset exists in session state
        - Dataset is not None
        - Dataset is not empty
        - Dataset is a valid DataFrame

        Examples
        --------
        >>> if DataManager.validate_dataset():
        ...     # Proceed with analysis
        ...     pass
        ... else:
        ...     print("Please load a dataset first")
        """
        try:
            dataset = st.session_state.get(SESSION_KEYS['dataset'])

            if dataset is None:
                return False

            if not isinstance(dataset, pd.DataFrame):
                return False

            if dataset.empty:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return False

    @staticmethod
    def get_dataset_info() -> Dict[str, Any]:
        """
        Get comprehensive metadata about the active dataset.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - name: dataset name
            - shape: (n_rows, n_cols)
            - memory_usage: formatted memory string
            - columns: list of column names
            - dtypes: data types
            - numeric_columns: list of numeric column names
            - categorical_columns: list of categorical column names
            - missing_values: total missing value count
            - activation_time: when dataset was activated
            - file_path: source file path

        Examples
        --------
        >>> info = DataManager.get_dataset_info()
        >>> print(f"Dataset: {info['name']}")
        >>> print(f"Shape: {info['shape']}")
        >>> print(f"Memory: {info['memory_usage']}")
        """
        if not DataManager.validate_dataset():
            return {}

        try:
            # Get basic info
            name = st.session_state.get(SESSION_KEYS['dataset_name'], 'Unknown')
            file_path = st.session_state.get(SESSION_KEYS['dataset_path'], '')
            metadata = st.session_state.get(SESSION_KEYS['dataset_metadata'], {})

            # Get dataset
            df = st.session_state[SESSION_KEYS['dataset']]

            # Calculate current memory usage
            memory_bytes = df.memory_usage(deep=True).sum()
            memory_str = (
                f"{memory_bytes / (1024 * 1024):.2f} MB"
                if memory_bytes > 1024 * 1024
                else f"{memory_bytes / 1024:.2f} KB"
            )

            # Compile info
            info = {
                'name': name,
                'file_path': file_path,
                'shape': df.shape,
                'memory_usage': memory_str,
                'memory_bytes': memory_bytes,
                'columns': df.columns.tolist(),
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'missing_values': int(df.isnull().sum().sum()),
                'activation_time': metadata.get('activation_time', 'Unknown'),
                'duplicate_rows': int(df.duplicated().sum())
            }

            return info

        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return {}

    @staticmethod
    def clear_active_dataset():
        """
        Remove the active dataset from session state.

        Notes
        -----
        - Clears dataset, name, path, and metadata
        - Does not clear analysis results
        - Logs the clearing action

        Examples
        --------
        >>> DataManager.clear_active_dataset()
        >>> assert not DataManager.validate_dataset()
        """
        try:
            st.session_state[SESSION_KEYS['dataset']] = None
            st.session_state[SESSION_KEYS['dataset_name']] = None
            st.session_state[SESSION_KEYS['dataset_path']] = None
            st.session_state[SESSION_KEYS['dataset_metadata']] = {}

            logger.info("Active dataset cleared")

        except Exception as e:
            logger.error(f"Error clearing dataset: {str(e)}")

    @staticmethod
    def save_analysis_result(
            analysis_type: str,
            result: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store analysis results in session state.

        Parameters
        ----------
        analysis_type : str
            Type of analysis: 'statistical', 'ml', 'image', 'bioinformatics'.
        result : Dict[str, Any]
            Analysis result dictionary.
        metadata : Dict[str, Any], optional
            Additional metadata about the analysis.

        Notes
        -----
        Results are stored with timestamps and can be retrieved later
        for reporting or comparison.

        Examples
        --------
        >>> result = {'mean': 42.0, 'std': 5.5}
        >>> DataManager.save_analysis_result('statistical', result)
        """
        try:
            timestamp = datetime.now().isoformat()

            # Prepare record
            record = {
                'timestamp': timestamp,
                'analysis_type': analysis_type,
                'result': result,
                'metadata': metadata or {},
                'dataset_name': st.session_state.get(SESSION_KEYS['dataset_name'])
            }

            # Store in appropriate location
            if analysis_type == 'statistical':
                key = SESSION_KEYS['data_analysis']
                if key not in st.session_state:
                    st.session_state[key] = {}
                st.session_state[key][timestamp] = record

            elif analysis_type == 'ml':
                key = SESSION_KEYS['ml_results']
                if key not in st.session_state:
                    st.session_state[key] = {}
                st.session_state[key][timestamp] = record

            # Add to history
            if SESSION_KEYS['analysis_history'] in st.session_state:
                st.session_state[SESSION_KEYS['analysis_history']].append(record)

            logger.info(f"Saved {analysis_type} analysis result")

        except Exception as e:
            logger.error(f"Error saving analysis result: {str(e)}")

    @staticmethod
    def get_analysis_history() -> List[Dict[str, Any]]:
        """
        Retrieve history of all analyses performed.

        Returns
        -------
        List[Dict[str, Any]]
            List of analysis records with timestamps and results.

        Examples
        --------
        >>> history = DataManager.get_analysis_history()
        >>> for analysis in history:
        ...     print(f"{analysis['timestamp']}: {analysis['analysis_type']}")
        """
        return st.session_state.get(SESSION_KEYS['analysis_history'], [])

    @staticmethod
    def export_session_data() -> Dict[str, Any]:
        """
        Export all session data for backup or download.

        Returns
        -------
        Dict[str, Any]
            Complete session state snapshot (excluding dataset for size).

        Notes
        -----
        Useful for:
        - Session backup
        - Results export
        - Reproducibility documentation
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'dataset_info': DataManager.get_dataset_info(),
                'analysis_history': DataManager.get_analysis_history(),
                'ml_results': st.session_state.get(SESSION_KEYS['ml_results'], {}),
                'experiments': st.session_state.get(SESSION_KEYS['experiments'], {})
            }

            return export_data

        except Exception as e:
            logger.error(f"Error exporting session data: {str(e)}")
            return {}


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def validate_dataset() -> bool:
    """Backward compatibility wrapper for DataManager.validate_dataset()."""
    return DataManager.validate_dataset()


def get_dataset_info() -> Dict[str, Any]:
    """Backward compatibility wrapper for DataManager.get_dataset_info()."""
    return DataManager.get_dataset_info()


def clear_active_dataset():
    """Backward compatibility wrapper for DataManager.clear_active_dataset()."""
    return DataManager.clear_active_dataset()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'DataManager',
    'validate_dataset',
    'get_dataset_info',
    'clear_active_dataset',
    'SESSION_KEYS',
    'MAX_DATASET_SIZE_MB'
]
