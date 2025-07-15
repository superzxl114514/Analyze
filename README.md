# Statistical Analysis Agent Test Project

A comprehensive statistical analysis framework with LLM-powered intelligent tool selection and reasoning capabilities.

## Overview

This project implements an enhanced statistical analysis agent that combines traditional statistical tools with modern LLM reasoning capabilities. It provides intelligent tool selection, parameter extraction, and comprehensive statistical analysis across various domains including parametric tests, non-parametric tests, time series analysis, Bayesian methods, and more.

## Features

### 🧠 Intelligent Tool Selection
- **LLM-powered reasoning**: Advanced tool selection based on query analysis
- **Parameter extraction**: Automatic extraction of numerical data from natural language queries
- **Confidence scoring**: Confidence levels for tool selection decisions
- **Fallback mechanisms**: Robust fallback when LLM reasoning fails

### 📊 Comprehensive Statistical Tools
- **Parametric Tests**: t-tests, ANOVA, chi-square tests
- **Non-parametric Tests**: Mann-Whitney U, Wilcoxon, Kruskal-Wallis
- **Time Series Analysis**: Granger causality, VAR models, trend analysis
- **Bayesian Methods**: Bayes factor t-tests, hierarchical regression
- **Mixed Effects Models**: Linear mixed effects modeling
- **Multifactor Analysis**: Factorial ANOVA, MANOVA
- **Repeated Measures**: Repeated measures ANOVA, Friedman tests

### 🔧 Advanced Capabilities
- **Async support**: Concurrent processing of multiple queries
- **Parameter validation**: Automatic validation of tool parameters
- **History tracking**: Tool selection and usage history
- **Statistics reporting**: Usage statistics and performance metrics

## Project Structure

```
Test/
├── main.py                          # Main entry point
├── example_enhanced_usage.py        # Usage examples and demonstrations
├── test_fixed_adapter.py           # Test suite for the enhanced adapter
├── agent/
│   ├── agent_core.py               # Core routing and execution logic
│   ├── stat_agent_llm_adapter.py   # Basic LLM adapter
│   ├── enhanced_llm_adapter.py     # Enhanced adapter with reasoning
│   ├── tool_registry.py            # Tool registry and metadata
│   ├── tool_reasoning_engine.py    # Statistical reasoning engine
│   └── integrated_reasoning_adapter.py # Integrated reasoning system
└── tool/
    ├── parametric_test.py          # Parametric statistical tests
    ├── time_series_analysis.py     # Time series analysis tools
    ├── bayes_analysis.py          # Bayesian analysis methods
    ├── mixed_effects_models.py     # Mixed effects modeling
    ├── multi_factor_analysis.py    # Multifactor analysis tools
    └── repeated_measures_and_covariate_analysis.py # Repeated measures
```

## Quick Start

### Prerequisites

```bash
pip install openai numpy pandas scipy statsmodels pingouin pymc
```

### Basic Usage

```python
from agent.enhanced_llm_adapter import EnhancedStatAgentLLMAdapter

# Initialize the enhanced agent
agent = EnhancedStatAgentLLMAdapter(
    model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
    api_key="EMPTY",
    client_args={"base_url": "http://your-server:port/v1/"},
    generate_args={"temperature": 0.1}
)

# Perform statistical analysis
result = agent("Perform an independent t-test on [1,2,3,4,5] and [5,6,7,8,9]")
print(result)
```

### Advanced Usage

```python
import asyncio

# Async processing
async def analyze_multiple_queries():
    queries = [
        "Compare three groups using ANOVA: [1,2,3], [4,5,6], [7,8,9]",
        "Test correlation between [1,2,3,4,5] and [2,4,6,8,10]",
        "Perform Bayesian t-test on [1,2,3,4,5] and [5,6,7,8,9]"
    ]
    
    tasks = [agent.acall(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

# Run async analysis
results = asyncio.run(analyze_multiple_queries())
```

## Available Statistical Tools

### Parametric Tests
- `independent_t_test`: Compare means of two independent groups
- `paired_t_test`: Compare means of two related groups
- `one_way_anova`: Compare means across three or more groups
- `chi2_independence_test`: Test association between categorical variables

### Non-parametric Tests
- `mann_whitney_u_test`: Non-parametric comparison of two groups
- `wilcoxon_test`: Non-parametric comparison of related samples
- `kruskal_wallis_test`: Non-parametric comparison of multiple groups

### Time Series Analysis
- `granger_causality`: Test if one series predicts another
- `var_model`: Vector autoregression modeling
- `time_series_correlation`: Pearson correlation between series
- `compute_trend`: Linear trend analysis

### Bayesian Methods
- `bayes_factor_ttest`: Bayesian factor t-test
- `bayesian_simple_regression`: Bayesian linear regression
- `bayesian_hierarchical_regression`: Hierarchical regression

### Advanced Analysis
- `fit_mixedlm`: Linear mixed effects models
- `factorial_anova`: Factorial ANOVA with interactions
- `manova`: Multivariate analysis of variance
- `repeated_measures_anova`: Repeated measures analysis

## Examples

### Tool Selection with Reasoning

```python
# The agent automatically selects the appropriate tool
result = agent("I want to compare the means of three groups")
print(f"Selected tool: {result['tool_selection'].selected_tool}")
print(f"Confidence: {result['tool_selection'].confidence}")
print(f"Reasoning: {result['tool_selection'].reasoning}")
```

### Parameter Validation

```python
# Automatic parameter extraction and validation
result = agent("Perform t-test on [1,2,3,4,5] and [6,7,8,9,10]")
if result['execution_result'].get('success'):
    print("Analysis completed successfully")
else:
    print(f"Error: {result['execution_result'].get('error')}")
```

### History and Statistics

```python
# View tool selection history
history = agent.get_tool_selection_history()
for entry in history:
    print(f"Query: {entry['query'][:50]}...")
    print(f"Tool: {entry['selected_tool']}")

# View usage statistics
stats = agent.get_tool_usage_statistics()
for tool, count in stats.items():
    print(f"{tool}: {count} times")
```

## Configuration

### Model Configuration

```python
agent = EnhancedStatAgentLLMAdapter(
    model_name="your-model-name",
    api_key="your-api-key",
    client_args={
        "base_url": "http://your-server:port/v1/",
        "timeout": 30
    },
    generate_args={
        "temperature": 0.1,
        "max_tokens": 1000
    },
    tool_selection_strategy=ToolSelectionStrategy.REASONING
)
```

### Tool Selection Strategies

- `DIRECT`: Direct tool selection
- `RANKING`: Ranked tool selection
- `ENSEMBLE`: Ensemble selection
- `REASONING`: LLM-powered reasoning (default)

## Testing

Run the test suite to verify functionality:

```bash
cd /data/wujinchao/Test
python test_fixed_adapter.py
```

Run the comprehensive example:

```bash
python example_enhanced_usage.py
```

## Error Handling

The system includes robust error handling:

- **Parameter validation**: Automatic validation of tool parameters
- **Fallback mechanisms**: Keyword-based fallback when LLM fails
- **Async error handling**: Proper exception handling in async operations
- **Graceful degradation**: System continues operation even with partial failures

## Performance Considerations

- **Async processing**: Support for concurrent query processing
- **Caching**: Tool selection history for performance optimization
- **Resource management**: Efficient memory and CPU usage
- **Timeout handling**: Configurable timeouts for long-running operations

## Contributing

To contribute to this project:

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings for new functions
3. Include example usage in docstrings
4. Update the tool registry for new statistical tools
5. Add appropriate tests for new functionality

## License

This project is provided as-is for research and educational purposes.

## Support

For issues and questions:
- Check the example files for usage patterns
- Review the test suite for expected behavior
- Examine the tool registry for available statistical methods 