# agent/tool_registry.py

from tool.parametric_test import (
    independent_t_test, paired_t_test, mann_whitney_u_test, wilcoxon_test,
    one_way_anova, kruskal_wallis_test, chi2_independence_test,
    fisher_exact_test, ks_2samp_test
)
from tool.time_series_analysis import (
    granger_causality, var_model, time_series_correlation, compute_trend
)
from tool.bayes_analysis import (
    bayes_factor_ttest, bayesian_simple_regression, bayesian_hierarchical_regression
)
from tool.mixed_effects_models import fit_mixedlm
from tool.multi_factor_analysis import factorial_anova, manova
from tool.repeated_measures_and_covariate_analysis import (
    repeated_measures_anova, friedman_test, ancova, multiple_comparisons
)

TOOLS = {
    # ------------------ Parametric & Non-parametric Tests ------------------
    "independent_t_test": {
        "func": independent_t_test,
        "description": "Independent samples t-test: Compare means of two independent groups. Args: sample1, sample2, equal_var (default True).",
        "args": ["sample1", "sample2"],
        "arg_types": {"sample1": "list of float", "sample2": "list of float", "equal_var": "bool"}
    },
    "paired_t_test": {
        "func": paired_t_test,
        "description": "Paired samples t-test: Compare means of two related groups. Args: sample1, sample2.",
        "args": ["sample1", "sample2"],
        "arg_types": {"sample1": "list of float", "sample2": "list of float"}
    },
    "mann_whitney_u_test": {
        "func": mann_whitney_u_test,
        "description": "Mann-Whitney U test: Non-parametric comparison of two independent samples. Args: sample1, sample2, alternative.",
        "args": ["sample1", "sample2"],
        "arg_types": {"sample1": "list of float", "sample2": "list of float", "alternative": "str"}
    },
    "wilcoxon_test": {
        "func": wilcoxon_test,
        "description": "Wilcoxon signed-rank test: Non-parametric comparison of two related samples. Args: sample1, sample2, alternative.",
        "args": ["sample1", "sample2"],
        "arg_types": {"sample1": "list of float", "sample2": "list of float", "alternative": "str"}
    },
    "one_way_anova": {
        "func": one_way_anova,
        "description": "One-way ANOVA: Compare means across three or more independent groups. Args: *samples.",
        "args": ["*samples"],
        "arg_types": {"*samples": "multiple lists of float"}
    },
    "kruskal_wallis_test": {
        "func": kruskal_wallis_test,
        "description": "Kruskal-Wallis H test: Non-parametric comparison for three or more independent groups. Args: *samples.",
        "args": ["*samples"],
        "arg_types": {"*samples": "multiple lists of float"}
    },
    "chi2_independence_test": {
        "func": chi2_independence_test,
        "description": "Chi-square test of independence: Test association between two categorical variables. Args: contingency_table.",
        "args": ["contingency_table"],
        "arg_types": {"contingency_table": "2D list or array"}
    },
    "fisher_exact_test": {
        "func": fisher_exact_test,
        "description": "Fisher's exact test: Exact test for 2x2 contingency tables. Args: contingency_table.",
        "args": ["contingency_table"],
        "arg_types": {"contingency_table": "2x2 list or array"}
    },
    "ks_2samp_test": {
        "func": ks_2samp_test,
        "description": "Two-sample Kolmogorov-Smirnov test: Compare two distributions. Args: sample1, sample2, alternative.",
        "args": ["sample1", "sample2"],
        "arg_types": {"sample1": "list of float", "sample2": "list of float", "alternative": "str"}
    },

    # ------------------ Time Series Analysis ------------------
    "granger_causality": {
        "func": granger_causality,
        "description": "Granger causality test: Test if one time series can forecast another. Args: data (2D array/DataFrame), maxlag, verbose.",
        "args": ["data"],
        "arg_types": {"data": "2D array/DataFrame", "maxlag": "int", "verbose": "bool"}
    },
    "var_model": {
        "func": var_model,
        "description": "Vector autoregression (VAR): Fit VAR model to multiple time series. Args: data (DataFrame), lags.",
        "args": ["data"],
        "arg_types": {"data": "DataFrame", "lags": "int"}
    },
    "time_series_correlation": {
        "func": time_series_correlation,
        "description": "Time series correlation: Calculate Pearson correlation between two time series. Args: x, y.",
        "args": ["x", "y"],
        "arg_types": {"x": "list or array", "y": "list or array"}
    },
    "compute_trend": {
        "func": compute_trend,
        "description": "Trend analysis: Estimate linear trend (slope and intercept) for a time series. Args: series.",
        "args": ["series"],
        "arg_types": {"series": "list or array"}
    },

    # ------------------ Bayesian Analysis ------------------
    "bayes_factor_ttest": {
        "func": bayes_factor_ttest,
        "description": "Bayesian factor t-test: Quantify evidence for mean difference between two groups. Args: x, y, paired.",
        "args": ["x", "y"],
        "arg_types": {"x": "list of float", "y": "list of float", "paired": "bool"}
    },
    "bayesian_simple_regression": {
        "func": bayesian_simple_regression,
        "description": "Bayesian simple linear regression. Args: x, y.",
        "args": ["x", "y"],
        "arg_types": {"x": "list or array", "y": "list or array"}
    },
    "bayesian_hierarchical_regression": {
        "func": bayesian_hierarchical_regression,
        "description": "Bayesian hierarchical regression: Multilevel modeling with groupings. Args: df (DataFrame).",
        "args": ["df"],
        "arg_types": {"df": "DataFrame with 'x', 'y', and 'group' columns"}
    },

    # ------------------ Mixed Effects Models ------------------
    "fit_mixedlm": {
        "func": fit_mixedlm,
        "description": "Fit a linear mixed effects model (LMM) with fixed and random effects. Args: data, formula, groups.",
        "args": ["data", "formula", "groups"],
        "arg_types": {"data": "DataFrame", "formula": "str", "groups": "str"}
    },

    # ------------------ Multifactor Analysis ------------------
    "factorial_anova": {
        "func": factorial_anova,
        "description": "Factorial ANOVA: Analyze main and interaction effects of two or more factors. Args: data, formula.",
        "args": ["data", "formula"],
        "arg_types": {"data": "DataFrame", "formula": "str"}
    },
    "manova": {
        "func": manova,
        "description": "Multivariate analysis of variance (MANOVA): Test differences across groups for multiple dependent variables. Args: data, formula.",
        "args": ["data", "formula"],
        "arg_types": {"data": "DataFrame", "formula": "str"}
    },

    # ------------------ Repeated Measures and Covariate Analysis ------------------
    "repeated_measures_anova": {
        "func": repeated_measures_anova,
        "description": "Repeated measures ANOVA: Analyze within-subject effects across conditions or time points. Args: data, dv, subject, within.",
        "args": ["data", "dv", "subject", "within"],
        "arg_types": {"data": "DataFrame", "dv": "str", "subject": "str", "within": "str"}
    },
    "friedman_test": {
        "func": friedman_test,
        "description": "Friedman test: Non-parametric alternative to repeated measures ANOVA. Args: *groups.",
        "args": ["*groups"],
        "arg_types": {"*groups": "multiple lists/arrays, same subjects"}
    },
    "ancova": {
        "func": ancova,
        "description": "Analysis of covariance (ANCOVA): Compare group means while controlling for covariate(s). Args: data, dv, covar, between.",
        "args": ["data", "dv", "covar", "between"],
        "arg_types": {"data": "DataFrame", "dv": "str", "covar": "str", "between": "str"}
    },
    "multiple_comparisons": {
        "func": multiple_comparisons,
        "description": "Multiple comparison correction: Adjust p-values for multiple group comparisons. Args: data, group_col, value_col, method.",
        "args": ["data", "group_col", "value_col", "method"],
        "arg_types": {"data": "DataFrame", "group_col": "str", "value_col": "str", "method": "str"}
    },
}
