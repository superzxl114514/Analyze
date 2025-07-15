#!/usr/bin/env python3
"""
Enhanced LLM Adapter Usage Example
Demonstrates intelligent tool selection reasoning functionality
"""

import asyncio
from agent.enhanced_llm_adapter import EnhancedStatAgentLLMAdapter, ToolSelectionStrategy


def sync_example():
    """Synchronous usage example"""
    print("=== Synchronous Usage Example ===")
    
    # Initialize enhanced adapter
    enhanced_agent = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1},
        tool_selection_strategy=ToolSelectionStrategy.REASONING
    )
    
    # Test queries
    test_queries = [
        "Perform an independent t-test on [1,2,3,4,5] and [5,6,7,8,9].",
        "Compare three groups using ANOVA: group1=[1,2,3], group2=[4,5,6], group3=[7,8,9].",
        "Test if there's a correlation between time series [1,2,3,4,5] and [2,4,6,8,10].",
        "Perform a Bayesian t-test on [1,2,3,4,5] and [5,6,7,8,9]."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        try:
            result = enhanced_agent(query)
            
            print(f"Selected tool: {result['tool_selection'].selected_tool}")
            print(f"Confidence: {result['tool_selection'].confidence:.3f}")
            print(f"Reasoning: {result['tool_selection'].reasoning}")
            print(f"Execution plan: {result['tool_selection'].execution_plan}")
            
            if result['execution_result'].get('success'):
                print(f"Execution result: {result['execution_result']['execution_result']}")
            else:
                print(f"Execution error: {result['execution_result'].get('error')}")
                
        except Exception as e:
            print(f"Error processing query: {e}")


async def async_example():
    """Asynchronous usage example"""
    print("\n=== Asynchronous Usage Example ===")
    
    # Initialize enhanced adapter
    enhanced_agent = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1},
        tool_selection_strategy=ToolSelectionStrategy.REASONING
    )
    
    # Process multiple queries concurrently
    queries = [
        "Perform a paired t-test on [1,2,3,4,5] and [2,3,4,5,6].",
        "Test for normality of [1,2,3,4,5,6,7,8,9,10].",
        "Perform a Mann-Whitney U test on [1,2,3] and [4,5,6]."
    ]
    
    tasks = [enhanced_agent.acall(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, (query, result) in enumerate(zip(queries, results), 1):
        print(f"\n--- Async Query {i}: {query} ---")
        
        if isinstance(result, Exception):
            print(f"Error processing query: {result}")
        else:
            print(f"Selected tool: {result['tool_selection'].selected_tool}")
            print(f"Confidence: {result['tool_selection'].confidence:.3f}")
            print(f"Reasoning: {result['tool_selection'].reasoning[:200]}...")
            
            if result['execution_result'].get('success'):
                print(f"Execution result: {result['execution_result']['execution_result']}")
            else:
                print(f"Execution error: {result['execution_result'].get('error')}")


def demonstrate_tool_categories():
    """Demonstrate tool categorization functionality"""
    print("\n=== Tool Categorization Demo ===")
    
    enhanced_agent = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1}
    )
    
    print("Tool categories:")
    for category, tools in enhanced_agent.tool_categories.items():
        if tools:
            print(f"\n{category.replace('_', ' ').title()}:")
            for tool in tools:
                print(f"  - {tool}")


def demonstrate_fallback_mechanism():
    """Demonstrate fallback mechanism"""
    print("\n=== Fallback Mechanism Demo ===")
    
    enhanced_agent = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1}
    )
    
    # Test fallback mechanism
    fallback_queries = [
        "I want to do a t-test",
        "Compare means with ANOVA",
        "Test correlation",
        "Unknown statistical test"
    ]
    
    for query in fallback_queries:
        print(f"\nQuery: {query}")
        result = enhanced_agent._fallback_tool_selection(query)
        print(f"Selected tool: {result.selected_tool}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")


def demonstrate_parameter_validation():
    """Demonstrate parameter validation functionality"""
    print("\n=== Parameter Validation Demo ===")
    
    enhanced_agent = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1}
    )
    
    # Test parameter validation
    from agent.tool_registry import TOOLS
    
    test_cases = [
        ("independent_t_test", {"sample1": [1,2,3], "sample2": [4,5,6]}),
        ("independent_t_test", {"sample1": [1,2,3]}),  # Missing parameter
        ("unknown_tool", {"sample1": [1,2,3]}),  # Unknown tool
    ]
    
    for tool_name, parameters in test_cases:
        is_valid, message = enhanced_agent.validate_tool_parameters(tool_name, parameters)
        print(f"Tool: {tool_name}")
        print(f"Parameters: {parameters}")
        print(f"Validation result: {'Pass' if is_valid else 'Fail'}")
        print(f"Message: {message}")
        print()


def demonstrate_history_and_stats():
    """Demonstrate history and statistics functionality"""
    print("\n=== History and Statistics Demo ===")
    
    enhanced_agent = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1}
    )
    
    # Execute some queries
    test_queries = [
        "Perform t-test on [1,2,3] and [4,5,6]",
        "Compare groups with ANOVA",
        "Test correlation between [1,2,3] and [4,5,6]"
    ]
    
    for query in test_queries:
        try:
            enhanced_agent(query)
        except Exception as e:
            print(f"Query '{query}' failed: {e}")
    
    # Display history
    print("\nTool selection history:")
    history = enhanced_agent.get_tool_selection_history()
    for entry in history:
        print(f"  - Query: {entry['query'][:50]}...")
        print(f"    Tool: {entry['selected_tool']}")
        print(f"    Confidence: {entry['confidence']:.3f}")
    
    # Display usage statistics
    print("\nTool usage statistics:")
    stats = enhanced_agent.get_tool_usage_statistics()
    for tool, count in stats.items():
        print(f"  - {tool}: {count} times")


def main():
    """Main function"""
    print("Enhanced LLM Adapter Demo")
    print("=" * 50)
    
    # Demonstrate tool categories
    demonstrate_tool_categories()
    
    # Demonstrate fallback mechanism
    demonstrate_fallback_mechanism()
    
    # Demonstrate parameter validation
    demonstrate_parameter_validation()
    
    # Synchronous usage example
    sync_example()
    
    # Asynchronous usage example
    asyncio.run(async_example())
    
    # History and statistics demo
    demonstrate_history_and_stats()


if __name__ == "__main__":
    main() 