# utils/analysis_utils.py
"""
Statistical Analysis Utilities - SYPHU iGEM Research Platform
==============================================================

Comprehensive statistical analysis tools for biological data analysis,
including descriptive statistics, hypothesis testing, correlation analysis,
and data quality assessment.

Author: SYPHU-CHINA iGEM Team
Date: 2025-10-01
License: MIT

Dependencies:
    - pandas >= 1.5.0
    - numpy >= 1.20.0
    - scipy >= 1.7.0
    - statsmodels >= 0.13.0 (optional, for advanced tests)

Notes
-----
This module implements statistical methods following best practices
recommended by Nature Methods and Nature Protocols for biological
data analysis, including proper multiple testing correction and
effect size reporting.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple, Union
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

# Default significance level
ALPHA = 0.05

# Effect size interpretation thresholds (Cohen's d)
EFFECT_SIZE_SMALL = 0.2
EFFECT_SIZE_MEDIUM = 0.5
EFFECT_SIZE_LARGE = 0.8

# Correlation strength thresholds
CORR_WEAK = 0.3
CORR_MODERATE = 0.5
CORR_STRONG = 0.7


# ============================================================================
# Main Analysis Class
# ============================================================================

class AnalysisUtils:
    """
    Statistical analysis utility class for biological data.

    Provides comprehensive statistical methods including:
    - Descriptive statistics
    - Hypothesis testing (t-tests, ANOVA, chi-square)
    - Correlation analysis
    - Normality tests
    - Effect size calculations
    - Multiple testing corrections

    All methods follow Nature journal guidelines for statistical reporting.

    Methods
    -------
    descriptive_stats(df, columns=None)
        Calculate comprehensive descriptive statistics.
    correlation_analysis(df, method='pearson', threshold=0.7)
        Perform correlation analysis with significance testing.
    t_test_independent(group1, group2, alpha=0.05)
        Independent samples t-test with effect size.
    t_test_paired(group1, group2, alpha=0.05)
        Paired samples t-test.
    anova_oneway(df, value_col, group_col, alpha=0.05)
        One-way ANOVA with post-hoc tests.
    normality_test(data, method='shapiro')
        Test for normal distribution.
    chi_square_test(observed, expected=None)
        Chi-square test of independence.
    effect_size_cohens_d(group1, group2)
        Calculate Cohen's d effect size.
    multiple_testing_correction(p_values, method='bonferroni')
        Correct for multiple comparisons.

    Examples
    --------
    >>> analyzer = AnalysisUtils()
    >>> stats = analyzer.descriptive_stats(df)
    >>> print(stats['mean'])

    >>> corr = analyzer.correlation_analysis(df)
    >>> print(corr['correlation_matrix'])
    """

    @staticmethod
    def descriptive_stats(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        columns : List[str], optional
            Specific columns to analyze (default: all numeric columns).

        Returns
        -------
        pd.DataFrame
            Descriptive statistics including:
            - count: sample size
            - mean, std, min, max: basic statistics
            - quartiles: 25%, 50%, 75%
            - skewness, kurtosis: distribution shape
            - missing: count and percentage of missing values
            - coefficient_of_variation: relative variability

        Examples
        --------
        >>> stats = AnalysisUtils.descriptive_stats(df)
        >>> print(stats.loc['gene_expression', 'mean'])

        Notes
        -----
        - Automatically excludes non-numeric columns
        - Skewness > 0: right-skewed, < 0: left-skewed
        - Kurtosis > 0: heavy-tailed, < 0: light-tailed
        - CV (coefficient of variation) = std/mean, useful for comparing variability
        """
        try:
            # Select numeric columns
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            if not columns:
                logger.warning("No numeric columns found for analysis")
                return pd.DataFrame()

            # Calculate basic statistics
            stats_df = df[columns].describe().T

            # Add distribution measures
            stats_df['skewness'] = df[columns].skew()
            stats_df['kurtosis'] = df[columns].kurtosis()

            # Add missing value info
            stats_df['missing_count'] = df[columns].isnull().sum()
            stats_df['missing_pct'] = (df[columns].isnull().sum() / len(df)) * 100

            # Add coefficient of variation (CV)
            stats_df['cv'] = (stats_df['std'] / stats_df['mean']).abs()

            # Add confidence interval for mean (95%)
            n = stats_df['count']
            se = stats_df['std'] / np.sqrt(n)
            ci_95 = 1.96 * se
            stats_df['ci_lower'] = stats_df['mean'] - ci_95
            stats_df['ci_upper'] = stats_df['mean'] + ci_95

            logger.info(f"Calculated descriptive statistics for {len(columns)} variables")
            return stats_df

        except Exception as e:
            logger.error(f"Error calculating descriptive statistics: {str(e)}")
            return pd.DataFrame()


    @staticmethod
    def correlation_analysis(
        df: pd.DataFrame,
        method: str = 'pearson',
        threshold: float = 0.7,
        return_pvalues: bool = True
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis with significance testing.

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        method : str, optional
            Correlation method: 'pearson', 'spearman', or 'kendall' (default: 'pearson').
        threshold : float, optional
            Threshold for identifying strong correlations (default: 0.7).
        return_pvalues : bool, optional
            Whether to calculate p-values (default: True).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - correlation_matrix: correlation coefficients
            - p_value_matrix: significance p-values (if requested)
            - high_correlations: list of variable pairs exceeding threshold
            - method: correlation method used
            - n_significant: number of significant correlations

        Examples
        --------
        >>> corr_results = AnalysisUtils.correlation_analysis(df, method='spearman')
        >>> print(corr_results['high_correlations'])

        Notes
        -----
        - Pearson: assumes linear relationship and normal distribution
        - Spearman: rank-based, robust to outliers
        - Kendall: similar to Spearman, better for small samples
        - P-values are corrected for multiple comparisons (Bonferroni)
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                logger.warning("No numeric columns for correlation analysis")
                return {}

            # Calculate correlation matrix
            corr_matrix = numeric_df.corr(method=method)

            result = {
                'correlation_matrix': corr_matrix,
                'method': method,
                'n_variables': len(corr_matrix)
            }

            # Calculate p-values if requested
            if return_pvalues:
                p_matrix = pd.DataFrame(
                    np.zeros_like(corr_matrix),
                    index=corr_matrix.index,
                    columns=corr_matrix.columns
                )

                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]

                        if method == 'pearson':
                            _, p_val = stats.pearsonr(
                                numeric_df[col1].dropna(),
                                numeric_df[col2].dropna()
                            )
                        elif method == 'spearman':
                            _, p_val = stats.spearmanr(
                                numeric_df[col1].dropna(),
                                numeric_df[col2].dropna()
                            )
                        else:  # kendall
                            _, p_val = stats.kendalltau(
                                numeric_df[col1].dropna(),
                                numeric_df[col2].dropna()
                            )

                        p_matrix.iloc[i, j] = p_val
                        p_matrix.iloc[j, i] = p_val

                result['p_value_matrix'] = p_matrix

            # Find high correlations
            high_corr_pairs = []
            n_comparisons = len(corr_matrix) * (len(corr_matrix) - 1) / 2

            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_val = corr_matrix.iloc[i, j]

                    if abs(corr_val) > threshold:
                        pair_info = {
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_val),
                            'abs_correlation': float(abs(corr_val))
                        }

                        if return_pvalues:
                            p_val = p_matrix.iloc[i, j]
                            # Bonferroni correction
                            adj_p = min(p_val * n_comparisons, 1.0)
                            pair_info['p_value'] = float(p_val)
                            pair_info['adj_p_value'] = float(adj_p)
                            pair_info['significant'] = adj_p < ALPHA

                        high_corr_pairs.append(pair_info)

            # Sort by absolute correlation
            high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

            result['high_correlations'] = high_corr_pairs
            result['n_high_correlations'] = len(high_corr_pairs)

            logger.info(
                f"Correlation analysis complete: {len(high_corr_pairs)} "
                f"correlations > {threshold}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {}


    @staticmethod
    def t_test_independent(
        group1: Union[pd.Series, np.ndarray],
        group2: Union[pd.Series, np.ndarray],
        alpha: float = ALPHA,
        equal_var: bool = True
    ) -> Dict[str, Any]:
        """
        Independent samples t-test with effect size.

        Parameters
        ----------
        group1 : pd.Series or np.ndarray
            First group data.
        group2 : pd.Series or np.ndarray
            Second group data.
        alpha : float, optional
            Significance level (default: 0.05).
        equal_var : bool, optional
            Assume equal variances (default: True).

        Returns
        -------
        Dict[str, Any]
            Test results including:
            - t_statistic: t-test statistic
            - p_value: two-tailed p-value
            - significant: whether result is significant
            - cohens_d: effect size
            - effect_interpretation: small/medium/large
            - descriptive_stats: means and SDs for both groups

        Examples
        --------
        >>> result = AnalysisUtils.t_test_independent(control, treatment)
        >>> if result['significant']:
        ...     print(f"Effect size: {result['cohens_d']:.3f}")

        Notes
        -----
        - Use equal_var=False (Welch's t-test) if variances differ
        - Cohen's d: 0.2=small, 0.5=medium, 0.8=large effect
        - Always report effect sizes alongside p-values
        """
        try:
            # Clean data
            g1 = pd.Series(group1).dropna()
            g2 = pd.Series(group2).dropna()

            if len(g1) < 2 or len(g2) < 2:
                logger.warning("Insufficient data for t-test")
                return {}

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=equal_var)

            # Calculate effect size (Cohen's d)
            cohens_d = AnalysisUtils.effect_size_cohens_d(g1, g2)

            # Interpret effect size
            abs_d = abs(cohens_d['cohens_d'])
            if abs_d < EFFECT_SIZE_SMALL:
                effect_interp = "negligible"
            elif abs_d < EFFECT_SIZE_MEDIUM:
                effect_interp = "small"
            elif abs_d < EFFECT_SIZE_LARGE:
                effect_interp = "medium"
            else:
                effect_interp = "large"

            result = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'cohens_d': cohens_d['cohens_d'],
                'effect_interpretation': effect_interp,
                'equal_variance_assumed': equal_var,
                'descriptive_stats': {
                    'group1': {
                        'n': len(g1),
                        'mean': float(g1.mean()),
                        'std': float(g1.std()),
                        'se': float(g1.sem())
                    },
                    'group2': {
                        'n': len(g2),
                        'mean': float(g2.mean()),
                        'std': float(g2.std()),
                        'se': float(g2.sem())
                    }
                }
            }

            # Add confidence interval for difference
            diff = g1.mean() - g2.mean()
            pooled_se = np.sqrt(g1.var()/len(g1) + g2.var()/len(g2))
            df = len(g1) + len(g2) - 2
            t_crit = stats.t.ppf(1 - alpha/2, df)

            result['mean_difference'] = float(diff)
            result['ci_lower'] = float(diff - t_crit * pooled_se)
            result['ci_upper'] = float(diff + t_crit * pooled_se)

            logger.info(f"T-test completed: p={p_value:.4f}, d={cohens_d['cohens_d']:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error in t-test: {str(e)}")
            return {}


    @staticmethod
    def t_test_paired(
        group1: Union[pd.Series, np.ndarray],
        group2: Union[pd.Series, np.ndarray],
        alpha: float = ALPHA
    ) -> Dict[str, Any]:
        """
        Paired samples t-test.

        Parameters
        ----------
        group1 : pd.Series or np.ndarray
            First measurement (e.g., pre-treatment).
        group2 : pd.Series or np.ndarray
            Second measurement (e.g., post-treatment).
        alpha : float, optional
            Significance level (default: 0.05).

        Returns
        -------
        Dict[str, Any]
            Test results including t-statistic, p-value, and effect size.

        Notes
        -----
        Use when comparing two measurements from the same subjects.
        """
        try:
            g1 = pd.Series(group1).dropna()
            g2 = pd.Series(group2).dropna()

            if len(g1) != len(g2):
                logger.warning("Groups must have equal length for paired t-test")
                return {}

            t_stat, p_value = stats.ttest_rel(g1, g2)

            # Calculate paired effect size
            differences = g1 - g2
            cohens_d = differences.mean() / differences.std()

            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'cohens_d': float(cohens_d),
                'mean_difference': float(differences.mean()),
                'n_pairs': len(g1)
            }

        except Exception as e:
            logger.error(f"Error in paired t-test: {str(e)}")
            return {}


    @staticmethod
    def anova_oneway(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        alpha: float = ALPHA
    ) -> Dict[str, Any]:
        """
        One-way analysis of variance (ANOVA).

        Parameters
        ----------
        df : pd.DataFrame
            Input data frame.
        value_col : str
            Column name for dependent variable.
        group_col : str
            Column name for grouping variable.
        alpha : float, optional
            Significance level (default: 0.05).

        Returns
        -------
        Dict[str, Any]
            ANOVA results including:
            - f_statistic: F-test statistic
            - p_value: significance p-value
            - eta_squared: effect size (proportion of variance explained)
            - group_stats: descriptive statistics per group

        Examples
        --------
        >>> result = AnalysisUtils.anova_oneway(df, 'expression', 'treatment')
        >>> if result['significant']:
        ...     print(f"η² = {result['eta_squared']:.3f}")

        Notes
        -----
        - Tests if means differ across ≥3 groups
        - Post-hoc tests (Tukey HSD) recommended if significant
        - η² (eta-squared): 0.01=small, 0.06=medium, 0.14=large effect
        """
        try:
            # Extract groups
            groups = []
            group_names = []
            group_stats = {}

            for name, group in df.groupby(group_col):
                data = group[value_col].dropna()
                if len(data) > 0:
                    groups.append(data)
                    group_names.append(name)
                    group_stats[str(name)] = {
                        'n': len(data),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'se': float(data.sem())
                    }

            if len(groups) < 2:
                logger.warning("Need at least 2 groups for ANOVA")
                return {}

            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)

            # Calculate effect size (eta-squared)
            grand_mean = np.concatenate(groups).mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            result = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'eta_squared': float(eta_squared),
                'num_groups': len(groups),
                'group_names': group_names,
                'group_stats': group_stats,
                'total_n': sum(len(g) for g in groups)
            }

            logger.info(f"ANOVA completed: F={f_stat:.3f}, p={p_value:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error in ANOVA: {str(e)}")
            return {}


    @staticmethod
    def normality_test(
        data: Union[pd.Series, np.ndarray],
        method: str = 'shapiro',
        alpha: float = ALPHA
    ) -> Dict[str, Any]:
        """
        Test for normal distribution.

        Parameters
        ----------
        data : pd.Series or np.ndarray
            Data to test.
        method : str, optional
            Test method: 'shapiro' or 'kstest' (default: 'shapiro').
        alpha : float, optional
            Significance level (default: 0.05).

        Returns
        -------
        Dict[str, Any]
            Test results including statistic, p-value, and interpretation.

        Notes
        -----
        - Shapiro-Wilk: recommended for n < 5000
        - Kolmogorov-Smirnov: for larger samples
        - Rejection (p < 0.05) indicates non-normality
        """
        try:
            clean_data = pd.Series(data).dropna()

            if len(clean_data) < 3:
                logger.warning("Insufficient data for normality test (n < 3)")
                return {}

            if method == 'shapiro':
                if len(clean_data) > 5000:
                    logger.warning("Shapiro-Wilk not recommended for n > 5000, using KS test")
                    method = 'kstest'
                else:
                    stat, p_value = stats.shapiro(clean_data)
                    test_name = "Shapiro-Wilk Test"

            if method == 'kstest':
                stat, p_value = stats.kstest(clean_data, 'norm')
                test_name = "Kolmogorov-Smirnov Test"

            return {
                'test': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > alpha,
                'alpha': alpha,
                'sample_size': len(clean_data),
                'interpretation': f"Data is {'normally' if p_value > alpha else 'not normally'} distributed"
            }

        except Exception as e:
            logger.error(f"Error in normality test: {str(e)}")
            return {}


    @staticmethod
    def chi_square_test(
        observed: pd.DataFrame,
        expected: Optional[pd.DataFrame] = None,
        alpha: float = ALPHA
    ) -> Dict[str, Any]:
        """
        Chi-square test of independence.

        Parameters
        ----------
        observed : pd.DataFrame
            Observed frequency table (contingency table).
        expected : pd.DataFrame, optional
            Expected frequencies (default: calculated from marginals).
        alpha : float, optional
            Significance level (default: 0.05).

        Returns
        -------
        Dict[str, Any]
            Test results including chi-square statistic, p-value, and effect size.

        Notes
        -----
        Tests whether two categorical variables are independent.
        """
        try:
            chi2, p_value, dof, expected_freq = stats.chi2_contingency(observed)

            # Calculate Cramér's V (effect size)
            n = observed.sum().sum()
            min_dim = min(observed.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

            return {
                'chi_square': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < alpha,
                'cramers_v': float(cramers_v),
                'expected_frequencies': expected_freq
            }

        except Exception as e:
            logger.error(f"Error in chi-square test: {str(e)}")
            return {}


    @staticmethod
    def effect_size_cohens_d(
        group1: Union[pd.Series, np.ndarray],
        group2: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate Cohen's d effect size.

        Parameters
        ----------
        group1, group2 : pd.Series or np.ndarray
            Two groups to compare.

        Returns
        -------
        Dict[str, float]
            Dictionary with 'cohens_d' and 'interpretation'.

        Notes
        -----
        Cohen's d interpretation:
        - |d| < 0.2: negligible
        - |d| < 0.5: small
        - |d| < 0.8: medium
        - |d| ≥ 0.8: large
        """
        try:
            g1 = pd.Series(group1).dropna()
            g2 = pd.Series(group2).dropna()

            mean_diff = g1.mean() - g2.mean()
            pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)

            if pooled_std == 0:
                return {'cohens_d': 0.0, 'interpretation': 'undefined (zero variance)'}

            d = mean_diff / pooled_std

            return {
                'cohens_d': float(d),
                'absolute_d': float(abs(d))
            }

        except Exception as e:
            logger.error(f"Error calculating Cohen's d: {str(e)}")
            return {'cohens_d': np.nan}


    @staticmethod
    def multiple_testing_correction(
        p_values: List[float],
        method: str = 'bonferroni',
        alpha: float = ALPHA
    ) -> Dict[str, Any]:
        """
        Correct p-values for multiple comparisons.

        Parameters
        ----------
        p_values : List[float]
            List of p-values to correct.
        method : str, optional
            Correction method: 'bonferroni', 'holm', 'fdr_bh' (default: 'bonferroni').
        alpha : float, optional
            Family-wise error rate (default: 0.05).

        Returns
        -------
        Dict[str, Any]
            Corrected p-values and significance decisions.

        Notes
        -----
        - Bonferroni: most conservative, controls family-wise error rate
        - Holm: less conservative than Bonferroni
        - FDR (Benjamini-Hochberg): controls false discovery rate
        """
        try:
            from statsmodels.stats.multitest import multipletests

            reject, pvals_corrected, _, _ = multipletests(
                p_values,
                alpha=alpha,
                method=method
            )

            return {
                'original_pvalues': p_values,
                'corrected_pvalues': pvals_corrected.tolist(),
                'reject_null': reject.tolist(),
                'method': method,
                'alpha': alpha,
                'n_significant': int(reject.sum())
            }

        except ImportError:
            logger.warning("statsmodels not installed, using simple Bonferroni")
            corrected = [min(p * len(p_values), 1.0) for p in p_values]
            return {
                'original_pvalues': p_values,
                'corrected_pvalues': corrected,
                'reject_null': [p < alpha for p in corrected],
                'method': 'bonferroni',
                'alpha': alpha
            }
        except Exception as e:
            logger.error(f"Error in multiple testing correction: {str(e)}")
            return {}


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'AnalysisUtils',
    'ALPHA',
    'EFFECT_SIZE_SMALL',
    'EFFECT_SIZE_MEDIUM',
    'EFFECT_SIZE_LARGE'
]
