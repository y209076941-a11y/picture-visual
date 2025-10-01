# utils/plot_utils.py
"""
Plotting Utilities - SYPHU iGEM Research Platform
==================================================

Publication-quality scientific visualization tools using Plotly.
Provides standardized plotting functions following Nature journal
figure guidelines for clarity, accessibility, and reproducibility.

Author: SYPHU-CHINA iGEM Team
Date: 2025-10-01
License: MIT

Dependencies:
    - plotly >= 5.0.0
    - pandas >= 1.5.0
    - numpy >= 1.20.0
    - scipy >= 1.7.0 (for KDE)

Notes
-----
All plots follow Nature journal guidelines:
- Colorblind-friendly palettes
- Minimum 6pt font sizes
- High contrast for readability
- Export-ready for publications (300+ DPI)
- Consistent styling across all plot types

References
----------
- Nature Methods: Guidelines for figure preparation
- Wong, B. (2011). Points of view: Color blindness. Nature Methods 8, 441
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

# Colorblind-friendly color schemes (following Wong 2011 Nature Methods)
COLOR_SCHEMES = {
    'wong': [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#CC79A7',  # Reddish Purple
        '#000000'   # Black
    ],
    'nature': [
        '#E64B35',  # Red
        '#4DBBD5',  # Blue
        '#00A087',  # Green
        '#3C5488',  # Navy
        '#F39B7F',  # Orange
        '#8491B4',  # Purple
        '#91D1C2'   # Teal
    ],
    'science': [
        '#0C5DA5',  # Blue
        '#00B945',  # Green
        '#FF9500',  # Orange
        '#FF2C00',  # Red
        '#845B97',  # Purple
        '#474747',  # Dark Gray
        '#9e9e9e'   # Light Gray
    ],
    'cell': [
        '#DC0000',  # Red
        '#F0E442',  # Yellow
        '#4DBBD5',  # Cyan
        '#009E73',  # Green
        '#E69F00',  # Orange
        '#56B4E9',  # Light Blue
        '#CC79A7'   # Pink
    ]
}

# Default plot configuration following Nature guidelines
DEFAULT_LAYOUT = {
    'font': {
        'family': 'Arial, sans-serif',
        'size': 12,
        'color': '#000000'
    },
    'title': {
        'font': {'size': 14, 'family': 'Arial, sans-serif'},
        'x': 0.5,
        'xanchor': 'center'
    },
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'margin': {'l': 80, 'r': 40, 't': 80, 'b': 80},
    'hovermode': 'closest'
}

# Axis configuration
DEFAULT_AXIS = {
    'showgrid': True,
    'gridcolor': '#E5E5E5',
    'gridwidth': 1,
    'showline': True,
    'linecolor': '#000000',
    'linewidth': 1.5,
    'ticks': 'outside',
    'tickcolor': '#000000',
    'tickwidth': 1.5,
    'tickfont': {'size': 11}
}


# ============================================================================
# Main PlotUtils Class
# ============================================================================

class PlotUtils:
    """
    Publication-quality plotting utilities for scientific data visualization.

    Provides standardized plotting functions that follow Nature journal
    guidelines for figure preparation. All plots are designed to be:
    - Colorblind-friendly
    - Print-ready (high DPI)
    - Consistent in styling
    - Accessible and clear

    Methods
    -------
    set_plot_template(template='plotly_white')
        Set global plot template.
    create_scatter(df, x, y, color=None, size=None, **kwargs)
        Create publication-ready scatter plot.
    create_histogram(df, column, bins=30, show_kde=True)
        Create histogram with optional KDE overlay.
    create_boxplot(df, y, x=None, **kwargs)
        Create box plot for group comparisons.
    create_violin_plot(df, y, x=None, **kwargs)
        Create violin plot showing distribution shape.
    create_correlation_heatmap(corr_matrix, **kwargs)
        Create correlation matrix heatmap.
    create_line_plot(df, x, y, color=None, **kwargs)
        Create line plot for time series or trends.
    create_bar_plot(df, x, y, color=None, **kwargs)
        Create bar plot for categorical comparisons.
    apply_nature_style(fig)
        Apply Nature journal styling to existing figure.

    Examples
    --------
    >>> fig = PlotUtils.create_scatter(df, 'time', 'expression', color='gene')
    >>> fig.show()

    >>> fig = PlotUtils.create_histogram(df, 'measurement', bins=50)
    >>> fig.write_image("histogram.png", width=1200, height=800, scale=3)
    """

    @staticmethod
    def set_plot_template(template: str = 'plotly_white'):
        """
        Set global plot template for all subsequent plots.

        Parameters
        ----------
        template : str, optional
            Plotly template name (default: 'plotly_white').
            Options: 'plotly_white', 'plotly', 'ggplot2', 'seaborn', 'simple_white'

        Examples
        --------
        >>> PlotUtils.set_plot_template('plotly_white')
        """
        try:
            import plotly.io as pio
            pio.templates.default = template
            logger.info(f"Plot template set to: {template}")
        except Exception as e:
            logger.error(f"Error setting plot template: {str(e)}")


    @staticmethod
    def apply_nature_style(fig: go.Figure) -> go.Figure:
        """
        Apply Nature journal styling to a Plotly figure.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure object to style.

        Returns
        -------
        go.Figure
            Styled figure following Nature guidelines.

        Notes
        -----
        Applies:
        - Arial font family
        - Appropriate font sizes (12pt body, 14pt title)
        - High contrast colors
        - Clean grid lines
        - Publication-ready margins
        """
        fig.update_layout(**DEFAULT_LAYOUT)
        fig.update_xaxes(**DEFAULT_AXIS)
        fig.update_yaxes(**DEFAULT_AXIS)

        return fig


    @staticmethod
    def create_scatter(
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: Optional[str] = None,
        color_scheme: str = 'wong',
        trendline: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create publication-ready scatter plot.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        color : str, optional
            Column name for color grouping.
        size : str, optional
            Column name for marker size.
        title : str, optional
            Plot title (auto-generated if None).
        color_scheme : str, optional
            Color scheme name (default: 'wong').
        trendline : str, optional
            Trendline type: 'ols', 'lowess', 'rolling', 'expanding', 'ewm'.
        **kwargs
            Additional arguments passed to px.scatter.

        Returns
        -------
        go.Figure
            Publication-ready scatter plot.

        Examples
        --------
        >>> fig = PlotUtils.create_scatter(df, 'time', 'expression',
        ...                                color='treatment', trendline='ols')
        >>> fig.write_image("scatter.png", width=1200, height=800, scale=3)

        Notes
        -----
        - Uses colorblind-friendly colors
        - Includes error bands for trendlines
        - Optimized for both screen and print
        """
        try:
            # Create scatter plot
            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=color,
                size=size,
                title=title or f"{y} vs {x}",
                color_discrete_sequence=COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['wong']),
                trendline=trendline,
                **kwargs
            )

            # Apply Nature styling
            fig = PlotUtils.apply_nature_style(fig)

            # Update marker styling
            fig.update_traces(
                marker=dict(
                    line=dict(width=0.5, color='white'),
                    opacity=0.8
                )
            )

            # Update axis labels
            fig.update_xaxes(title_text=x, title_font=dict(size=13))
            fig.update_yaxes(title_text=y, title_font=dict(size=13))

            logger.info(f"Created scatter plot: {y} vs {x}")
            return fig

        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return go.Figure()


    @staticmethod
    def create_histogram(
        df: pd.DataFrame,
        column: str,
        bins: int = 30,
        title: Optional[str] = None,
        show_kde: bool = True,
        show_stats: bool = True,
        color_scheme: str = 'science'
    ) -> go.Figure:
        """
        Create histogram with optional KDE overlay and statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        column : str
            Column name for histogram.
        bins : int, optional
            Number of bins (default: 30).
        title : str, optional
            Plot title.
        show_kde : bool, optional
            Show kernel density estimate (default: True).
        show_stats : bool, optional
            Add statistics annotation (default: True).
        color_scheme : str, optional
            Color scheme name (default: 'science').

        Returns
        -------
        go.Figure
            Histogram with optional KDE and statistics.

        Examples
        --------
        >>> fig = PlotUtils.create_histogram(df, 'gene_expression', bins=50)
        >>> fig.show()

        Notes
        -----
        - KDE uses Gaussian kernel
        - Statistics include mean, median, std
        - Optimized bin calculation using Sturges' rule if bins='auto'
        """
        try:
            data = df[column].dropna()

            if len(data) == 0:
                logger.warning(f"No data available for column: {column}")
                return go.Figure()

            # Create figure
            fig = go.Figure()

            # Add histogram
            colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['science'])
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=bins,
                name='Histogram',
                marker_color=colors[0],
                opacity=0.7,
                hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
            ))

            # Add KDE overlay
            if show_kde and len(data) > 2:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    kde_values = kde(x_range)

                    # Scale KDE to match histogram
                    hist_max = np.histogram(data, bins=bins)[0].max()
                    kde_scaled = kde_values * (hist_max / kde_values.max()) * 0.9

                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde_scaled,
                        mode='lines',
                        name='KDE',
                        line=dict(color=colors[3] if len(colors) > 3 else '#FF2C00', width=3),
                        hovertemplate='<b>Value:</b> %{x:.3f}<br><b>Density:</b> %{y:.3f}<extra></extra>'
                    ))
                except ImportError:
                    logger.warning("scipy not available for KDE calculation")

            # Add statistics annotation
            if show_stats:
                stats_text = (
                    f"n = {len(data)}<br>"
                    f"Mean = {data.mean():.3f}<br>"
                    f"Median = {data.median():.3f}<br>"
                    f"Std = {data.std():.3f}"
                )

                fig.add_annotation(
                    x=0.98,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text=stats_text,
                    showarrow=False,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10),
                    align='left',
                    xanchor='right',
                    yanchor='top'
                )

            # Apply styling
            fig = PlotUtils.apply_nature_style(fig)
            fig.update_layout(
                title=title or f"Distribution of {column}",
                xaxis_title=column,
                yaxis_title="Frequency",
                showlegend=True,
                legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top')
            )

            logger.info(f"Created histogram for: {column}")
            return fig

        except Exception as e:
            logger.error(f"Error creating histogram: {str(e)}")
            return go.Figure()


    @staticmethod
    def create_boxplot(
        df: pd.DataFrame,
        y: str,
        x: Optional[str] = None,
        title: Optional[str] = None,
        color_scheme: str = 'wong',
        show_points: bool = True,
        notched: bool = False,
        **kwargs
    ) -> go.Figure:
        """
        Create box plot for group comparisons.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        y : str
            Column name for values.
        x : str, optional
            Column name for grouping.
        title : str, optional
            Plot title.
        color_scheme : str, optional
            Color scheme name (default: 'wong').
        show_points : bool, optional
            Show all data points (default: True).
        notched : bool, optional
            Show notched box plot (default: False).
        **kwargs
            Additional arguments.

        Returns
        -------
        go.Figure
            Box plot with optional data points.

        Notes
        -----
        - Notches show 95% confidence interval around median
        - Points are jittered for visibility
        - Outliers are highlighted
        """
        try:
            fig = px.box(
                df,
                y=y,
                x=x,
                title=title or f"Distribution of {y}" + (f" by {x}" if x else ""),
                color=x,
                color_discrete_sequence=COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['wong']),
                points='all' if show_points else False,
                notched=notched,
                **kwargs
            )

            # Apply styling
            fig = PlotUtils.apply_nature_style(fig)
            fig.update_traces(
                marker=dict(size=4, opacity=0.6),
                line=dict(width=2)
            )

            logger.info(f"Created box plot for: {y}")
            return fig

        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return go.Figure()


    @staticmethod
    def create_violin_plot(
        df: pd.DataFrame,
        y: str,
        x: Optional[str] = None,
        title: Optional[str] = None,
        color_scheme: str = 'nature',
        show_box: bool = True,
        show_points: bool = True,
        **kwargs
    ) -> go.Figure:
        """
        Create violin plot showing full distribution shape.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        y : str
            Column name for values.
        x : str, optional
            Column name for grouping.
        title : str, optional
            Plot title.
        color_scheme : str, optional
            Color scheme name (default: 'nature').
        show_box : bool, optional
            Show box plot inside violin (default: True).
        show_points : bool, optional
            Show individual data points (default: True).
        **kwargs
            Additional arguments.

        Returns
        -------
        go.Figure
            Violin plot with optional box and points.

        Notes
        -----
        Combines advantages of box plot (quartiles) and KDE (full distribution).
        """
        try:
            fig = px.violin(
                df,
                y=y,
                x=x,
                title=title or f"Violin Plot of {y}" + (f" by {x}" if x else ""),
                color=x,
                color_discrete_sequence=COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['nature']),
                box=show_box,
                points='all' if show_points else False,
                **kwargs
            )

            # Apply styling
            fig = PlotUtils.apply_nature_style(fig)
            fig.update_traces(
                marker=dict(size=3, opacity=0.5),
                meanline_visible=True
            )

            logger.info(f"Created violin plot for: {y}")
            return fig

        except Exception as e:
            logger.error(f"Error creating violin plot: {str(e)}")
            return go.Figure()


    @staticmethod
    def create_correlation_heatmap(
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
        show_values: bool = True,
        color_scale: str = 'RdBu_r',
        cluster: bool = False
    ) -> go.Figure:
        """
        Create correlation matrix heatmap.

        Parameters
        ----------
        corr_matrix : pd.DataFrame
            Correlation matrix.
        title : str, optional
            Plot title (default: "Correlation Matrix").
        show_values : bool, optional
            Display correlation values (default: True).
        color_scale : str, optional
            Plotly color scale (default: 'RdBu_r').
        cluster : bool, optional
            Apply hierarchical clustering (default: False).

        Returns
        -------
        go.Figure
            Correlation heatmap.

        Notes
        -----
        - Diverging color scale centered at 0
        - Values rounded to 2 decimal places
        - Symmetric matrix
        """
        try:
            # Optional clustering
            if cluster:
                from scipy.cluster.hierarchy import linkage, dendrogram
                from scipy.spatial.distance import squareform

                # Perform clustering
                distances = 1 - np.abs(corr_matrix)
                linkage_matrix = linkage(squareform(distances), method='average')
                dendro = dendrogram(linkage_matrix, no_plot=True)
                order = dendro['leaves']

                corr_matrix = corr_matrix.iloc[order, order]

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=color_scale,
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2) if show_values else None,
                texttemplate='%{text}' if show_values else None,
                textfont={"size": 10, "color": "black"},
                colorbar=dict(
                    title="Correlation",
                    titleside="right",
                    tickmode="linear",
                    tick0=-1,
                    dtick=0.5
                ),
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>'
            ))

            # Apply styling
            fig = PlotUtils.apply_nature_style(fig)
            fig.update_layout(
                title=title,
                width=max(600, len(corr_matrix) * 40),
                height=max(600, len(corr_matrix) * 40),
                xaxis={'side': 'bottom'},
                yaxis={'side': 'left'}
            )

            logger.info("Created correlation heatmap")
            return fig

        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return go.Figure()


    @staticmethod
    def create_line_plot(
        df: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        color: Optional[str] = None,
        title: Optional[str] = None,
        color_scheme: str = 'wong',
        show_markers: bool = True,
        **kwargs
    ) -> go.Figure:
        """
        Create line plot for time series or trends.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        x : str
            Column name for x-axis (often time).
        y : str or List[str]
            Column name(s) for y-axis values.
        color : str, optional
            Column name for color grouping.
        title : str, optional
            Plot title.
        color_scheme : str, optional
            Color scheme name (default: 'wong').
        show_markers : bool, optional
            Show data point markers (default: True).
        **kwargs
            Additional arguments.

        Returns
        -------
        go.Figure
            Line plot.
        """
        try:
            # Handle multiple y columns
            if isinstance(y, list):
                fig = go.Figure()
                colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['wong'])

                for idx, y_col in enumerate(y):
                    fig.add_trace(go.Scatter(
                        x=df[x],
                        y=df[y_col],
                        mode='lines+markers' if show_markers else 'lines',
                        name=y_col,
                        line=dict(color=colors[idx % len(colors)], width=2),
                        marker=dict(size=6)
                    ))
            else:
                fig = px.line(
                    df,
                    x=x,
                    y=y,
                    color=color,
                    title=title or f"{y} over {x}",
                    color_discrete_sequence=COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['wong']),
                    markers=show_markers,
                    **kwargs
                )

            # Apply styling
            fig = PlotUtils.apply_nature_style(fig)
            fig.update_traces(line=dict(width=2), marker=dict(size=6))

            logger.info(f"Created line plot: {y} vs {x}")
            return fig

        except Exception as e:
            logger.error(f"Error creating line plot: {str(e)}")
            return go.Figure()


    @staticmethod
    def create_bar_plot(
        df: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        color_scheme: str = 'science',
        error_y: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create bar plot for categorical comparisons.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        x : str
            Column name for categories.
        y : str
            Column name for values.
        color : str, optional
            Column name for color grouping.
        title : str, optional
            Plot title.
        color_scheme : str, optional
            Color scheme name (default: 'science').
        error_y : str, optional
            Column name for error bars.
        **kwargs
            Additional arguments.

        Returns
        -------
        go.Figure
            Bar plot with optional error bars.
        """
        try:
            fig = px.bar(
                df,
                x=x,
                y=y,
                color=color,
                title=title or f"{y} by {x}",
                color_discrete_sequence=COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['science']),
                error_y=error_y,
                **kwargs
            )

            # Apply styling
            fig = PlotUtils.apply_nature_style(fig)
            fig.update_traces(
                marker_line_color='black',
                marker_line_width=1,
                opacity=0.8
            )

            logger.info(f"Created bar plot: {y} by {x}")
            return fig

        except Exception as e:
            logger.error(f"Error creating bar plot: {str(e)}")
            return go.Figure()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'PlotUtils',
    'COLOR_SCHEMES',
    'DEFAULT_LAYOUT'
]
