# components/headers.py
"""
Header Components for Scientific Data Visualization
====================================================

This module provides standardized header components for scientific data
visualization applications, following Nature journal style guidelines.

Author: [Your Name]
Date: 2025-10-01
License: MIT

Dependencies:
    - streamlit >= 1.0.0
"""

import streamlit as st
from typing import Optional


def render_page_header(
        title: str,
        icon: str,
        subtitle: str = "",
        background_gradient: tuple = ("#2E86AB", "#A9D6E5")
) -> None:
    """
    Render a professional page header with gradient background.

    This function creates a visually appealing header section suitable for
    scientific applications and data visualization dashboards, following
    Nature journal aesthetic guidelines.

    Parameters
    ----------
    title : str
        Main title text to display in the header.
    icon : str
        Unicode emoji or icon character to display alongside the title.
    subtitle : str, optional
        Additional descriptive text below the title (default: "").
    background_gradient : tuple of str, optional
        Two-color gradient tuple in hex format (default: ("#2E86AB", "#A9D6E5")).

    Returns
    -------
    None
        Renders directly to the Streamlit interface.

    Examples
    --------
    >>> render_page_header(
    ...     title="Data Analysis Dashboard",
    ...     icon="📊",
    ...     subtitle="Advanced Statistical Visualization"
    ... )

    Notes
    -----
    - Colors are chosen to be colorblind-friendly
    - Font sizes follow Nature journal typography standards
    - Responsive design adapts to different screen sizes
    """

    # Validate inputs
    if not title or not isinstance(title, str):
        raise ValueError("Title must be a non-empty string")
    if not icon or not isinstance(icon, str):
        raise ValueError("Icon must be a non-empty string")

    # Extract gradient colors
    color_start, color_end = background_gradient

    # Construct subtitle HTML if provided
    subtitle_html = ""
    if subtitle:
        subtitle_html = f"""
        <p style='
            margin: 0.5rem 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 400;
            line-height: 1.4;
        '>{subtitle}</p>
        """

    # Render header with improved styling
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg, {color_start}, {color_end});
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        '>
            <div style='
                display: flex;
                align-items: center;
                gap: 1rem;
            '>
                <span style='
                    font-size: 3rem;
                    line-height: 1;
                '>{icon}</span>
                <div style='flex: 1;'>
                    <h1 style='
                        margin: 0;
                        font-size: 2.5rem;
                        font-weight: 700;
                        letter-spacing: -0.02em;
                        line-height: 1.2;
                    '>{title}</h1>
                    {subtitle_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section_header(
        title: str,
        icon: str = "",
        border_color: str = "#2E86AB",
        heading_level: int = 2
) -> None:
    """
    Render a section header with optional icon and left border.

    Creates a clearly defined section separator suitable for organizing
    scientific content and analysis results. Follows Nature journal
    section formatting guidelines.

    Parameters
    ----------
    title : str
        Section title text to display.
    icon : str, optional
        Unicode emoji or icon character (default: "").
    border_color : str, optional
        Hex color code for the left border (default: "#2E86AB").
    heading_level : int, optional
        HTML heading level (2-6), where 2 is <h2> (default: 2).

    Returns
    -------
    None
        Renders directly to the Streamlit interface.

    Examples
    --------
    >>> render_section_header(
    ...     title="Statistical Analysis",
    ...     icon="📈",
    ...     heading_level=2
    ... )

    Notes
    -----
    - Heading levels follow semantic HTML structure
    - Border color should maintain sufficient contrast
    - Icon is optional for cleaner academic appearance

    Raises
    ------
    ValueError
        If title is empty or heading_level is out of valid range (2-6).
    """

    # Validate inputs
    if not title or not isinstance(title, str):
        raise ValueError("Title must be a non-empty string")
    if not (2 <= heading_level <= 6):
        raise ValueError("Heading level must be between 2 and 6")

    # Construct icon HTML if provided
    icon_html = ""
    if icon:
        icon_html = f"""
        <span style='
            font-size: 1.5rem;
            margin-right: 0.5rem;
            vertical-align: middle;
        '>{icon}</span>
        """

    # Determine font size based on heading level
    font_sizes = {
        2: "2rem",
        3: "1.75rem",
        4: "1.5rem",
        5: "1.25rem",
        6: "1.1rem"
    }
    font_size = font_sizes[heading_level]

    # Render section header
    st.markdown(
        f"""
        <div style='
            border-left: 4px solid {border_color};
            padding-left: 1rem;
            margin: 2rem 0 1rem 0;
        '>
            <h{heading_level} style='
                margin: 0;
                color: {border_color};
                display: flex;
                align-items: center;
                font-size: {font_size};
                font-weight: 600;
                line-height: 1.3;
            '>
                {icon_html}{title}
            </h{heading_level}>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_info_box(
        content: str,
        box_type: str = "info",
        title: Optional[str] = None
) -> None:
    """
    Render an informational box for notes, warnings, or tips.

    Provides a visually distinct container for important information,
    following Nature journal callout box styling.

    Parameters
    ----------
    content : str
        Text content to display in the box.
    box_type : str, optional
        Type of box: 'info', 'warning', 'success', 'error' (default: 'info').
    title : str, optional
        Optional title for the info box (default: None).

    Returns
    -------
    None
        Renders directly to the Streamlit interface.

    Examples
    --------
    >>> render_info_box(
    ...     content="Statistical significance: p < 0.05",
    ...     box_type="info",
    ...     title="Note"
    ... )
    """

    # Define box styles based on type
    box_styles = {
        "info": {"color": "#2E86AB", "bg": "#E8F4F8", "icon": "ℹ️"},
        "warning": {"color": "#F4A460", "bg": "#FFF4E6", "icon": "⚠️"},
        "success": {"color": "#28A745", "bg": "#E8F5E9", "icon": "✅"},
        "error": {"color": "#DC3545", "bg": "#FFEBEE", "icon": "❌"}
    }

    style = box_styles.get(box_type, box_styles["info"])

    title_html = ""
    if title:
        title_html = f"""
        <div style='
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: {style['color']};
        '>
            {style['icon']} {title}
        </div>
        """

    st.markdown(
        f"""
        <div style='
            background-color: {style['bg']};
            border-left: 4px solid {style['color']};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        '>
            {title_html}
            <div style='color: #333; line-height: 1.6;'>
                {content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Module-level constants for consistent styling
NATURE_COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A9D6E5",
    "accent": "#F4A460",
    "success": "#28A745",
    "warning": "#FFC107",
    "error": "#DC3545",
    "text": "#333333",
    "background": "#FFFFFF"
}

# Export public API
__all__ = [
    'render_page_header',
    'render_section_header',
    'render_info_box',
    'NATURE_COLORS'
]
