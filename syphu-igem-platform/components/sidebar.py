# components/sidebar.py
"""
Sidebar Navigation Component for Scientific Research Platform
==============================================================

This module provides a standardized sidebar component with navigation,
data status monitoring, and platform information display.

Author: SYPHU-CHINA iGEM Team
Date: 2025-10-01
License: MIT

Dependencies:
    - streamlit >= 1.0.0
    - Custom modules: utils.data_loader, utils.data_manager

Notes
-----
This component follows Nature journal style guidelines for scientific
software interfaces, emphasizing clarity and functional organization.
"""

import streamlit as st
import os
import sys
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root directory to path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom utilities with error handling
try:
    from utils.data_loader import DataLoader
    from utils.data_manager import DataManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    st.error(f"⚠️ Module import failed: {e}")
    st.stop()

# ============================================================================
# Constants and Configuration
# ============================================================================

PLATFORM_CONFIG = {
    "name": "SYPHU iGEM",
    "subtitle": "Research Platform",
    "icon": "🧬",
    "version": "2.0.0",
    "year": "2024"
}

NAVIGATION_PAGES = {
    "🏠 Project Overview": "1_🏠_Project_Overview.py",
    "📂 Data Management": "2_📂_Data_Hub.py",
    "🔬 Analysis Modules": "3_🔬_Analysis_Modules.py",
    "🖼️ AI Image Analysis": "4_🖼️_AI_Image_Analysis.py",
    "🤖 Machine Learning": "5_🤖_Machine_Learning.py",
    "🧪 Experiment Management": "6_🧪_Experiment_Hub.py",
    "🧬 Bioinformatics": "7_🧬_Bioinformatics.py",
    "📈 Results Visualization": "8_📈_Results.py",
    "📚 Documentation": "9_📚_Documentation.py"
}

STYLE_CONFIG = {
    "primary_color": "#2E86AB",
    "secondary_color": "#A9D6E5",
    "success_color": "#28A745",
    "warning_color": "#FFC107",
    "text_color": "#666",
    "background_color": "#f8f9fa",
    "border_color": "#eee"
}


# ============================================================================
# Main Sidebar Rendering Function
# ============================================================================

def render_sidebar() -> None:
    """
    Render the complete sidebar with all components.

    This function orchestrates the display of the platform header,
    data status monitor, navigation menu, and footer in the Streamlit
    sidebar interface.

    Returns
    -------
    None
        Renders directly to the Streamlit sidebar.

    Examples
    --------
    >>> render_sidebar()

    Notes
    -----
    - All sub-components are modular and can be called independently
    - Errors in individual components won't crash the entire sidebar
    - Logging is implemented for debugging purposes
    """

    with st.sidebar:
        try:
            # Render platform header
            render_platform_header()

            # Render data status monitor
            render_data_status()

            # Render navigation menu
            render_navigation()

            # Render footer information
            render_footer()

        except Exception as e:
            logger.error(f"Sidebar rendering error: {e}")
            st.error("⚠️ Sidebar component error. Please refresh the page.")


# ============================================================================
# Component Rendering Functions
# ============================================================================

def render_platform_header() -> None:
    """
    Render the platform title and branding section.

    Displays the platform name, icon, and subtitle following
    professional scientific software design principles.

    Returns
    -------
    None
        Renders directly to the Streamlit sidebar.

    Notes
    -----
    - Uses centralized PLATFORM_CONFIG for easy customization
    - Styling follows Nature journal web interface guidelines
    """

    st.markdown(
        f"""
        <div style='
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid {STYLE_CONFIG['border_color']};
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, 
                {STYLE_CONFIG['primary_color']}15, 
                {STYLE_CONFIG['secondary_color']}15);
            border-radius: 8px;
        '>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>
                {PLATFORM_CONFIG['icon']}
            </div>
            <h2 style='
                margin: 0;
                color: {STYLE_CONFIG['primary_color']};
                font-weight: 700;
                letter-spacing: -0.02em;
            '>
                {PLATFORM_CONFIG['name']}
            </h2>
            <p style='
                margin: 0.5rem 0 0 0;
                font-size: 0.9em;
                color: {STYLE_CONFIG['text_color']};
                font-weight: 500;
            '>
                {PLATFORM_CONFIG['subtitle']}
            </p>
            <p style='
                margin: 0.25rem 0 0 0;
                font-size: 0.75em;
                color: {STYLE_CONFIG['text_color']};
                opacity: 0.7;
            '>
                v{PLATFORM_CONFIG['version']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_data_status() -> None:
    """
    Render data availability and dataset status monitor.

    Displays real-time information about data availability, file counts,
    and active dataset properties. Provides visual feedback using
    color-coded status indicators.

    Returns
    -------
    None
        Renders directly to the Streamlit sidebar.

    Notes
    -----
    - Integrates with DataLoader and DataManager utilities
    - Uses try-except blocks for robust error handling
    - Status colors follow standard UX conventions

    See Also
    --------
    DataLoader.check_data_availability : Check data file availability
    DataManager.validate_dataset : Validate active dataset
    DataManager.get_dataset_info : Retrieve dataset metadata
    """

    st.markdown(
        f"""
        <h3 style='
            color: {STYLE_CONFIG['primary_color']};
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        '>
            📊 Data Status
        </h3>
        """,
        unsafe_allow_html=True
    )

    try:
        # Check data availability
        has_data, file_count = DataLoader.check_data_availability()

        if has_data:
            st.success(f"✅ Data Ready ({file_count} files)")
        else:
            st.warning("⚠️ No data available")
            st.info("💡 Please upload data files in the Data Management page.")

        # Display active dataset information if available
        if DataManager.validate_dataset():
            dataset_info = DataManager.get_dataset_info()

            st.markdown("---")
            st.markdown(
                f"""
                <div style='
                    background: {STYLE_CONFIG['background_color']};
                    padding: 0.75rem;
                    border-radius: 6px;
                    border-left: 3px solid {STYLE_CONFIG['success_color']};
                '>
                    <p style='
                        margin: 0 0 0.5rem 0;
                        font-weight: 600;
                        color: {STYLE_CONFIG['primary_color']};
                        font-size: 0.9rem;
                    '>
                        📁 Active Dataset
                    </p>
                    <code style='
                        display: block;
                        padding: 0.5rem;
                        background: white;
                        border-radius: 4px;
                        font-size: 0.85rem;
                        margin-bottom: 0.5rem;
                        word-break: break-all;
                    '>
                        {dataset_info['name']}
                    </code>
                    <p style='
                        margin: 0;
                        font-size: 0.85rem;
                        color: {STYLE_CONFIG['text_color']};
                    '>
                        Dimensions: {dataset_info['shape'][0]} × {dataset_info['shape'][1]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        logger.error(f"Data status rendering error: {e}")
        st.error("⚠️ Unable to load data status")

    st.markdown("---")


def render_navigation() -> None:
    """
    Render the navigation menu with page links.

    Creates an interactive navigation menu with buttons for all
    platform pages. Buttons are styled for optimal usability and
    follow accessibility guidelines.

    Returns
    -------
    None
        Renders directly to the Streamlit sidebar.

    Notes
    -----
    - Page configuration is stored in NAVIGATION_PAGES constant
    - Uses st.switch_page for seamless navigation
    - Button styling ensures consistent user experience
    - Each button has a unique key to prevent conflicts

    Examples
    --------
    To add a new page to navigation:
    >>> NAVIGATION_PAGES["🔧 New Module"] = "10_🔧_New_Module.py"
    """

    st.markdown(
        f"""
        <h3 style='
            color: {STYLE_CONFIG['primary_color']};
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        '>
            🧭 Navigation
        </h3>
        """,
        unsafe_allow_html=True
    )

    # Render navigation buttons
    for page_name, page_file in NAVIGATION_PAGES.items():
        # Extract emoji and clean name for better display
        button_label = page_name

        # Create navigation button with custom styling
        if st.button(
                button_label,
                use_container_width=True,
                key=f"nav_{page_file}",
                help=f"Navigate to {page_name}"
        ):
            try:
                st.switch_page(f"pages/{page_file}")
            except Exception as e:
                logger.error(f"Navigation error to {page_file}: {e}")
                st.error(f"⚠️ Unable to navigate to {page_name}")


def render_footer() -> None:
    """
    Render the footer with platform information and credits.

    Displays copyright information, team credits, and version details
    in a visually subtle footer section.

    Returns
    -------
    None
        Renders directly to the Streamlit sidebar.

    Notes
    -----
    - Footer styling minimizes visual distraction
    - Information updates automatically from PLATFORM_CONFIG
    - Includes team attribution as per iGEM requirements
    """

    st.markdown("---")
    st.markdown(
        f"""
        <div style='
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            background: {STYLE_CONFIG['background_color']};
            border-radius: 8px;
            border: 1px solid {STYLE_CONFIG['border_color']};
        '>
            <p style='
                margin: 0 0 0.5rem 0;
                font-size: 0.85em;
                color: {STYLE_CONFIG['text_color']};
                font-weight: 600;
            '>
                {PLATFORM_CONFIG['icon']} SYPHU-CHINA iGEM {PLATFORM_CONFIG['year']}
            </p>
            <p style='
                margin: 0;
                font-size: 0.75em;
                color: {STYLE_CONFIG['text_color']};
                opacity: 0.8;
            '>
                Scientific Research Platform
            </p>
            <p style='
                margin: 0.5rem 0 0 0;
                font-size: 0.7em;
                color: {STYLE_CONFIG['text_color']};
                opacity: 0.6;
            '>
                Version {PLATFORM_CONFIG['version']} | Built with Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# Utility Functions
# ============================================================================

def get_current_page() -> Optional[str]:
    """
    Get the currently active page name.

    Returns
    -------
    str or None
        The name of the current page, or None if unavailable.

    Notes
    -----
    This function can be used for conditional rendering based on
    the active page context.
    """

    try:
        return st.session_state.get('current_page', None)
    except Exception as e:
        logger.error(f"Error getting current page: {e}")
        return None


def highlight_active_page(page_name: str) -> bool:
    """
    Check if a given page is currently active.

    Parameters
    ----------
    page_name : str
        The name of the page to check.

    Returns
    -------
    bool
        True if the page is active, False otherwise.

    Notes
    -----
    Can be used to apply special styling to the active navigation item.
    """

    current = get_current_page()
    return current == page_name if current else False


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'render_sidebar',
    'render_platform_header',
    'render_data_status',
    'render_navigation',
    'render_footer',
    'PLATFORM_CONFIG',
    'NAVIGATION_PAGES',
    'STYLE_CONFIG'
]
