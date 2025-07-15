#!/usr/bin/env python3
"""
Test the fixed enhanced LLM adapter
"""

from agent.enhanced_llm_adapter import EnhancedStatAgentLLMAdapter, ToolSelectionStrategy


def test_basic_functionality():
    """Test basic functionality"""
    print("=== Test Basic Functionality ===")
    
    # Initialize adapter
    adapter = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1}
    )
    
    # Test parameter extraction
    test_queries = [
        "Perform t-test on [1,2,3] and [4,5,6]",
        "Compare three groups: [1,2,3], [4,5,6], [7,8,9]",
        "Test correlation between [1,2,3,4,5] and [2,4,6,8,10]"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        params = adapter.extract_parameters_from_query(query)
        print(f"Extracted parameters: {params}")
        
        # Test tool selection
        try:
            result = adapter.select_tools_with_reasoning(query)
            print(f"Selected tool: {result.selected_tool}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Parameters: {result.parameters}")
            
            # Test execution
            execution_result = adapter.execute_with_validation(result)
            if execution_result.get('success'):
                print(f"Execution successful: {execution_result['execution_result']}")
            else:
                print(f"Execution failed: {execution_result.get('error')}")
                
        except Exception as e:
            print(f"Processing failed: {e}")


def test_parameter_validation():
    """Test parameter validation"""
    print("\n=== Test Parameter Validation ===")
    
    adapter = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1}
    )
    
    # Test valid parameters
    valid_params = {
        "sample1": [1.0, 2.0, 3.0],
        "sample2": [4.0, 5.0, 6.0]
    }
    
    is_valid, message = adapter.validate_tool_parameters("independent_t_test", valid_params)
    print(f"Valid parameter test: {'Pass' if is_valid else 'Fail'} - {message}")
    
    # Test invalid parameters
    invalid_params = {
        "sample1": [1.0, 2.0, 3.0]
        # Missing sample2
    }
    
    is_valid, message = adapter.validate_tool_parameters("independent_t_test", invalid_params)
    print(f"Invalid parameter test: {'Pass' if is_valid else 'Fail'} - {message}")


def test_tool_categories():
    """Test tool categorization"""
    print("\n=== Test Tool Categorization ===")
    
    adapter = EnhancedStatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},
        generate_args={"temperature": 0.1}
    )
    
    for category, tools in adapter.tool_categories.items():
        if tools:
            print(f"\n{category.replace('_', ' ').title()}:")
            for tool in tools[:3]:  # Only show first 3
                print(f"  - {tool}")


if __name__ == "__main__":
    test_basic_functionality()
    test_parameter_validation()
    test_tool_categories()
    print("\nTest completed!") 