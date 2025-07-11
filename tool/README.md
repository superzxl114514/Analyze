# Statistical Analysis Toolkit

This is a comprehensive Python toolkit for statistical analysis, providing a variety of commonly used statistical tests and analysis methods. The toolkit covers a complete range of methods from basic parametric tests to advanced Bayesian analysis.

## 📋 Table of Contents

* [Features Overview](#features-overview)
* [Installation](#installation)
* [Module Descriptions](#module-descriptions)
* [Usage Examples](#usage-examples)
* [API Reference](#api-reference)
* [Notes](#notes)

## 🎯 Features Overview

This toolkit includes the following main modules:

* **Parametric Tests** (`parametric_test.py`) – Basic statistical testing methods
* **Time Series Analysis** (`time_series_analysis.py`) – Time series correlation and analysis
* **Bayesian Analysis** (`bayes_analysis.py`) – Bayesian statistical methods
* **Mixed Effects Models** (`mixed_effects_models.py`) – Linear mixed effects models
* **Multifactor Analysis** (`multi_factor_analysis.py`) – Multifactorial ANOVA
* **Repeated Measures and Covariate Analysis** (`repeated_measures_and_covariate_analysis.py`) – Advanced statistical methods

## 📦 Installation

Before using this toolkit, please make sure you have installed the following Python packages:

```bash
pip install numpy pandas scipy statsmodels pingouin pymc
```

### Package Descriptions

* `numpy`: Fundamental package for numerical computation
* `pandas`: Data processing and analysis
* `scipy`: Scientific computing and statistical functions
* `statsmodels`: Statistical modeling
* `pingouin`: Statistical testing library
* `pymc`: Bayesian statistical modeling

## 📚 Module Descriptions

### 1. Parametric Test Module (`parametric_test.py`)

Provides basic statistical test methods, including:

* **Independent samples t-test** (`independent_t_test`)
* **Paired samples t-test** (`paired_t_test`)
* **Mann-Whitney U test** (`mann_whitney_u_test`)
* **Wilcoxon signed-rank test** (`wilcoxon_test`)
* **One-way ANOVA** (`one_way_anova`)
* **Kruskal-Wallis H test** (`kruskal_wallis_test`)
* **Chi-square test of independence** (`chi2_independence_test`)
* **Fisher's exact test** (`fisher_exact_test`)
* **Two-sample Kolmogorov-Smirnov test** (`ks_2samp_test`)

### 2. Time Series Analysis Module (`time_series_analysis.py`)

Specialized for analyzing time series data:

* **Granger causality test** (`granger_causality`)
* **Vector autoregression model** (`var_model`)
* **Time series correlation analysis** (`time_series_correlation`)
* **Trend analysis** (`compute_trend`)

### 3. Bayesian Analysis Module (`bayes_analysis.py`)

Provides Bayesian statistical analysis methods:

* **Bayesian factor t-test** (`bayes_factor_ttest`)
* **Bayesian simple regression** (`bayesian_simple_regression`)
* **Bayesian hierarchical regression** (`bayesian_hierarchical_regression`)

### 4. Mixed Effects Models Module (`mixed_effects_models.py`)

For data with both fixed and random effects:

* **Linear mixed effects model** (`fit_mixedlm`)

### 5. Multifactor Analysis Module (`multi_factor_analysis.py`)

For analyzing multifactorial experimental designs:

* **Factorial ANOVA** (`factorial_anova`)
* **Multivariate analysis of variance (MANOVA)** (`manova`)

### 6. Repeated Measures and Covariate Analysis Module (`repeated_measures_and_covariate_analysis.py`)

Advanced statistical analysis methods:

* **Repeated measures ANOVA** (`repeated_measures_anova`)
* **Friedman test** (`friedman_test`)
* **Analysis of covariance (ANCOVA)** (`ancova`)
* **Multiple comparison correction** (`multiple_comparisons`)

## 💡 Usage Examples

### Basic Statistical Test

```python
from parametric_test import independent_t_test, paired_t_test
import numpy as np

# Generate example data
sample1 = np.random.normal(0, 1, 30)
sample2 = np.random.normal(0.5, 1, 30)

# Independent samples t-test
result = independent_t_test(sample1, sample2)
print(f"t-statistic: {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.4f}")
```

### Time Series Analysis

```python
from time_series_analysis import granger_causality, var_model
import pandas as pd

# Create time series data
data = pd.DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.normal(0, 1, 100)
})

# Granger causality test
result = granger_causality(data, maxlag=2)
```

### Bayesian Analysis

```python
from bayes_analysis import bayes_factor_ttest, bayesian_simple_regression

# Bayesian factor t-test
bf = bayes_factor_ttest(sample1, sample2)
print(f"Bayes factor: {bf:.3f}")

# Bayesian regression
x = np.linspace(0, 10, 100)
y = 2.5 * x + 3 + np.random.normal(0, 2, 100)
trace = bayesian_simple_regression(x, y)
```

### Mixed Effects Model

```python
from mixed_effects_models import fit_mixedlm

# Fit mixed effects model
formula = "score ~ group + time"
model_result = fit_mixedlm(data, formula, groups='subject')
print(model_result.summary())
```

## 📖 API Reference

### Parametric Test Functions

#### `independent_t_test(sample1, sample2, equal_var=True)`

* **Purpose**: Compare means of two independent groups
* **Parameters**:

  * `sample1`: First group data
  * `sample2`: Second group data
  * `equal_var`: Assume equal variances
* **Returns**: (t-statistic, p-value)

#### `paired_t_test(sample1, sample2)`

* **Purpose**: Compare two related samples
* **Parameters**:

  * `sample1`: Data before intervention
  * `sample2`: Data after intervention
* **Returns**: (t-statistic, p-value)

### Time Series Analysis Functions

#### `granger_causality(data, maxlag=2, verbose=True)`

* **Purpose**: Test whether one time series can predict another
* **Parameters**:

  * `data`: 2D array or DataFrame, columns as time series
  * `maxlag`: Maximum number of lags
  * `verbose`: Print detailed output
* **Returns**: Test result dictionary

#### `var_model(data, lags=1)`

* **Purpose**: Fit a vector autoregression model
* **Parameters**:

  * `data`: DataFrame, each column is a variable/time series
  * `lags`: Number of lags
* **Returns**: Fitted VARResults object

### Bayesian Analysis Functions

#### `bayes_factor_ttest(x, y, paired=False)`

* **Purpose**: Bayesian factor t-test
* **Parameters**:

  * `x`: First group sample
  * `y`: Second group sample
  * `paired`: Whether samples are paired
* **Returns**: Bayes factor

#### `bayesian_simple_regression(x, y)`

* **Purpose**: Bayesian simple linear regression
* **Parameters**:

  * `x`: Predictor
  * `y`: Outcome
* **Returns**: Posterior samples

### Mixed Effects Model Functions

#### `fit_mixedlm(data, formula, groups)`

* **Purpose**: Fit a linear mixed effects model
* **Parameters**:

  * `data`: pandas DataFrame
  * `formula`: Model formula string
  * `groups`: Grouping variable for random effects
* **Returns**: Fitted model result

## ⚠️ Notes

1. **Data format**: Ensure input data format is correct, especially column names and data types for DataFrames.
2. **Sample size**: Some tests have sample size requirements; ensure adequate data.
3. **Normality assumption**: Parametric tests generally assume normality. If not satisfied, consider non-parametric alternatives.
4. **Multiple comparisons**: Correct for multiple testing where necessary.
5. **Bayesian analysis**: Set proper priors; interpret results with care.

## 🔧 Running Examples

Each module contains example functions that can be run directly:

```python
# Run parametric test examples
python parametric_test.py

# Run time series analysis examples
python time_series_analysis.py

# Run Bayesian analysis examples
python bayes_analysis.py
```

## 📄 License

This toolkit is open source and may be used and modified freely.

## 🤝 Contributing

Issues and suggestions for improvement are welcome!

---

**Last updated:** 2024
**Version:** 1.0.0
