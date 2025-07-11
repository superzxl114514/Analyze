# Project Structure Overview

## 📁 Directory Structure

```
Test/sum/
├── README.md                                # Main documentation
├── QUICKSTART.md                            # Quick start guide
├── PROJECT_STRUCTURE.md                     # Project structure description (this file)
├── requirements.txt                         # Python dependency list
├── __init__.py                              # Python package initializer
├── parametric_test.py                       # Parametric tests module
├── time_series_analysis.py                  # Time series analysis module
├── bayes_analysis.py                        # Bayesian analysis module
├── mixed_effects_models.py                  # Mixed effects models module
├── multi_factor_analysis.py                 # Multifactor analysis module
└── repeated_measures_and_covariate_analysis.py  # Repeated measures and covariate analysis module
```

## 🔧 Module Overview

### Core Statistical Test Modules

#### `parametric_test.py` – Parametric Test Module

```
Main functions:
├── independent_t_test()      # Independent samples t-test
├── paired_t_test()           # Paired samples t-test
├── mann_whitney_u_test()     # Mann-Whitney U test
├── wilcoxon_test()           # Wilcoxon signed-rank test
├── one_way_anova()           # One-way ANOVA
├── kruskal_wallis_test()     # Kruskal-Wallis H test
├── chi2_independence_test()  # Chi-square test of independence
├── fisher_exact_test()       # Fisher's exact test
└── ks_2samp_test()           # Two-sample Kolmogorov-Smirnov test
```

#### `time_series_analysis.py` – Time Series Analysis Module

```
Main functions:
├── granger_causality()        # Granger causality test
├── var_model()                # Vector autoregression (VAR) model
├── time_series_correlation()  # Time series correlation analysis
└── compute_trend()            # Trend analysis
```

#### `bayes_analysis.py` – Bayesian Analysis Module

```
Main functions:
├── bayes_factor_ttest()               # Bayesian factor t-test
├── bayesian_simple_regression()       # Bayesian simple regression
└── bayesian_hierarchical_regression() # Bayesian hierarchical regression
```

### Advanced Statistical Modeling Modules

#### `mixed_effects_models.py` – Mixed Effects Models Module

```
Main function:
└── fit_mixedlm()                      # (Linear) mixed effects model
```

#### `multi_factor_analysis.py` – Multifactor Analysis Module

```
Main functions:
├── factorial_anova()                  # Factorial ANOVA
└── manova()                           # Multivariate analysis of variance (MANOVA)
```

#### `repeated_measures_and_covariate_analysis.py` – Repeated Measures and Covariate Analysis Module

```
Main functions:
├── repeated_measures_anova()          # Repeated measures ANOVA
├── friedman_test()                    # Friedman test
├── ancova()                           # Analysis of covariance (ANCOVA)
└── multiple_comparisons()             # Multiple comparison correction
```

## 📊 Function Classification

### Basic Statistical Tests

* **Parametric tests**: t-test, ANOVA, chi-square test
* **Non-parametric tests**: Mann-Whitney U, Wilcoxon, Kruskal-Wallis
* **Distribution tests**: Kolmogorov-Smirnov test

### Time Series Analysis

* **Causality**: Granger causality test
* **Modeling**: Vector autoregression (VAR)
* **Correlation**: Time series correlation analysis
* **Trend**: Linear trend analysis

### Bayesian Statistics

* **Hypothesis testing**: Bayesian factor t-test
* **Regression analysis**: Bayesian simple regression
* **Hierarchical modeling**: Bayesian hierarchical regression

### Advanced Modeling

* **Mixed effects**: Mixed effects model
* **Multifactor design**: Factorial ANOVA, MANOVA
* **Repeated measures**: Repeated measures ANOVA
* **Covariate analysis**: ANCOVA
* **Multiple comparisons**: Multiple comparison correction

## 🎯 Application Scenarios

### Experimental Design Analysis

* **Completely randomized design**: Use ANOVA in `parametric_test.py`
* **Randomized block design**: Use `mixed_effects_models.py`
* **Factorial design**: Use `multi_factor_analysis.py`
* **Repeated measures design**: Use `repeated_measures_and_covariate_analysis.py`

### Data Exploration

* **Descriptive statistics**: Basic statistical tests
* **Correlation analysis**: Time series correlation
* **Trend analysis**: Time series trend analysis

### Hypothesis Testing

* **Parametric methods**: When data meet normality assumptions
* **Non-parametric methods**: When data do not meet normality assumptions
* **Bayesian methods**: When quantifying strength of evidence is required

## 🔄 Module Relationships

```
Data input
   ↓
Basic tests (parametric_test.py)
   ↓
Advanced analysis
   ├── Time series (time_series_analysis.py)
   ├── Bayesian (bayes_analysis.py)
   ├── Mixed effects (mixed_effects_models.py)
   ├── Multifactor (multi_factor_analysis.py)
   └── Repeated measures (repeated_measures_and_covariate_analysis.py)
   ↓
Result output
```

---

**Note:**
Each module is independent and can be used separately or in combination as needed.
