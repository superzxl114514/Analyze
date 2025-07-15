import json
import asyncio
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

try:
    import openai
except ImportError:
    raise ImportError("Please install openai>=1.0.0: pip install openai")

from agent.tool_registry import TOOLS
from agent.agent_core import route_and_run

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolSelectionStrategy(Enum):
    """Tool selection strategy enumeration"""
    DIRECT = "direct"  # Direct selection
    RANKING = "ranking"  # Ranking selection
    ENSEMBLE = "ensemble"  # Ensemble selection
    REASONING = "reasoning"  # Reasoning selection


@dataclass
class ToolCandidate:
    """Tool candidate"""
    name: str
    description: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]


@dataclass
class ToolSelectionResult:
    """Tool selection result"""
    selected_tool: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]
    alternatives: List[ToolCandidate]
    execution_plan: str


class EnhancedStatAgentLLMAdapter:
    """
    Enhanced LLM-driven Statistical Analysis Agent Adapter
    Includes intelligent tool selection reasoning functionality
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client_args: dict = None,
        generate_args: dict = None,
        tool_selection_strategy: ToolSelectionStrategy = ToolSelectionStrategy.REASONING,
        **kwargs
    ):
        """
        Args:
            model_name: Model name
            api_key: API key
            client_args: Client arguments
            generate_args: Generation arguments
            tool_selection_strategy: Tool selection strategy
        """
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"
        self.generate_args = generate_args or {}
        self.client_args = client_args or {}
        self.tool_selection_strategy = tool_selection_strategy
        
        if "base_url" not in self.client_args:
            self.client_args["base_url"] = "http://localhost:8000/v1"
            logger.warning(f"No base_url provided, using default: {self.client_args['base_url']}")

        # OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            **self.client_args
        )
        
        # Async client
        try:
            from openai import AsyncOpenAI
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                **self.client_args
            )
        except ImportError:
            logger.warning("AsyncOpenAI not found, async will fallback to thread.")
            self.async_client = None

        # Tool selection history
        self.tool_selection_history = []
        
        # Initialize tool categorization
        self._categorize_tools()

    def _categorize_tools(self):
        """Categorize tools by functionality"""
        self.tool_categories = {
            "parametric_tests": [],
            "non_parametric_tests": [],
            "time_series": [],
            "bayesian": [],
            "mixed_effects": [],
            "multifactor": [],
            "repeated_measures": []
        }
        
        for tool_name, tool_info in TOOLS.items():
            if "t_test" in tool_name or "anova" in tool_name:
                self.tool_categories["parametric_tests"].append(tool_name)
            elif "mann_whitney" in tool_name or "wilcoxon" in tool_name or "kruskal" in tool_name:
                self.tool_categories["non_parametric_tests"].append(tool_name)
            elif "granger" in tool_name or "var" in tool_name or "trend" in tool_name:
                self.tool_categories["time_series"].append(tool_name)
            elif "bayes" in tool_name:
                self.tool_categories["bayesian"].append(tool_name)
            elif "mixed" in tool_name:
                self.tool_categories["mixed_effects"].append(tool_name)
            elif "factorial" in tool_name or "manova" in tool_name:
                self.tool_categories["multifactor"].append(tool_name)
            elif "repeated" in tool_name or "friedman" in tool_name or "ancova" in tool_name:
                self.tool_categories["repeated_measures"].append(tool_name)

    def build_tool_selection_prompt(self, user_query: str) -> str:
        """Build system prompt for tool selection"""
        tools_desc = "\n".join([
            f"- {name}: {tool['description']}" for name, tool in TOOLS.items()
        ])
        
        categories_desc = "\n".join([
            f"## {category.replace('_', ' ').title()}:\n" + 
            "\n".join([f"  - {tool}: {TOOLS[tool]['description']}" for tool in tools])
            for category, tools in self.tool_categories.items() if tools
        ])
        
        system_prompt = f"""You are an expert statistical analysis assistant. Your task is to analyze user queries and select the most appropriate statistical tools.

## Available Statistical Tools by Category:

{categories_desc}

## Tool Selection Guidelines:

1. **Data Type Analysis**: First determine if the data is continuous, categorical, or time series
2. **Sample Size Consideration**: For small samples (<30), prefer non-parametric tests
3. **Distribution Assumptions**: Check if normality and homogeneity of variance assumptions are met
4. **Research Design**: Consider if it's between-subjects, within-subjects, or mixed design
5. **Number of Groups**: One group (descriptive), two groups (t-tests), three+ groups (ANOVA)
6. **Dependent Variables**: Single (univariate) vs multiple (multivariate)

## Parameter Extraction Rules:
- Extract numerical data from user query using regex patterns
- For t-tests: extract two lists as sample1 and sample2
- For ANOVA: extract multiple lists as *samples
- For time series: extract data as 2D array
- For Bayesian tests: extract two lists as x and y
- Convert all numbers to float type
- Handle missing parameters gracefully

## Response Format:
Return a JSON object with the following structure:
{{
    "selected_tool": "tool_name",
    "confidence": 0.95,
    "reasoning": "Detailed explanation of why this tool was chosen",
    "parameters": {{
        "sample1": [1.0, 2.0, 3.0],
        "sample2": [4.0, 5.0, 6.0]
    }},
    "alternatives": [
        {{
            "name": "alternative_tool",
            "confidence": 0.7,
            "reasoning": "Why this alternative was considered"
        }}
    ],
    "execution_plan": "Step-by-step execution plan"
}}

## Important Notes:
- Always consider the research question and data characteristics
- Provide confidence scores (0-1) for your selection
- Include alternative tools that could also be appropriate
- Explain your reasoning clearly
- Ensure parameters match the tool's requirements
- Extract data from user query using regex patterns
"""
        return system_prompt

    def build_reasoning_prompt(self, user_query: str, selected_tool: str) -> str:
        """Build reasoning prompt for detailed tool selection analysis"""
        tool_info = TOOLS.get(selected_tool, {})
        
        reasoning_prompt = f"""You are a statistical expert. Analyze the following query and explain why the selected tool is appropriate.

**User Query**: {user_query}
**Selected Tool**: {selected_tool}
**Tool Description**: {tool_info.get('description', 'N/A')}

Please provide a detailed analysis including:

1. **Data Characteristics**: What type of data is being analyzed?
2. **Statistical Assumptions**: What assumptions does this test make and are they met?
3. **Research Design**: What type of experimental design is this?
4. **Alternative Considerations**: What other tools could be used and why weren't they chosen?
5. **Interpretation Guidance**: How should the results be interpreted?

Provide a comprehensive analysis that demonstrates deep statistical knowledge.
"""
        return reasoning_prompt

    def extract_parameters_from_query(self, user_query: str) -> Dict[str, Any]:
        """Extract parameters from user query"""
        parameters = {}
        
        # Extract number list patterns
        number_pattern = r'\[([\d\.,\s]+)\]'
        lists = re.findall(number_pattern, user_query)
        
        # Convert to float lists
        float_lists = []
        for list_str in lists:
            try:
                # Split and convert to float
                numbers = [float(x.strip()) for x in list_str.split(',')]
                float_lists.append(numbers)
            except ValueError:
                continue
        
        # Assign parameters based on tool type
        if len(float_lists) >= 2:
            query_lower = user_query.lower()
            
            if "paired" in query_lower or "dependent" in query_lower:
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
            elif "anova" in query_lower or "three" in query_lower or "multiple" in query_lower or "groups" in query_lower:
                parameters["*samples"] = float_lists
            elif "bayes" in query_lower:
                parameters["x"] = float_lists[0]
                parameters["y"] = float_lists[1]
            elif "time" in query_lower or "series" in query_lower or "granger" in query_lower:
                # Time series data
                if len(float_lists) == 2:
                    import numpy as np
                    data = np.column_stack(float_lists)
                    parameters["data"] = data.tolist()
            elif "mann" in query_lower or "whitney" in query_lower:
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
            elif "wilcoxon" in query_lower:
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
            elif "kruskal" in query_lower:
                parameters["*samples"] = float_lists
            else:
                # Default t-test
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
        
        return parameters

    def select_tools_with_reasoning(self, user_query: str) -> ToolSelectionResult:
        """Tool selection with reasoning"""
        logger.info("Performing tool selection with reasoning...")
        
        # Extract parameters
        extracted_params = self.extract_parameters_from_query(user_query)
        
        # Build selection prompt
        selection_prompt = self.build_tool_selection_prompt(user_query)
        
        messages = [
            {"role": "system", "content": selection_prompt},
            {"role": "user", "content": user_query}
        ]
        
        call_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        call_kwargs.update(self.generate_args)
        
        try:
            resp = self.client.chat.completions.create(**call_kwargs)
            content = resp.choices[0].message.content
            
            try:
                selection_result = json.loads(content)
            except json.JSONDecodeError:
                selection_result = self._fallback_tool_selection(user_query)
            
            # Merge extracted parameters with LLM-generated parameters
            if extracted_params:
                selection_result["parameters"] = {**selection_result.get("parameters", {}), **extracted_params}
            
            # Detailed reasoning
            reasoning_prompt = self.build_reasoning_prompt(
                user_query, 
                selection_result.get("selected_tool", "")
            )
            
            reasoning_messages = [
                {"role": "system", "content": reasoning_prompt},
                {"role": "user", "content": user_query}
            ]
            
            reasoning_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=reasoning_messages,
                temperature=0.3
            )
            
            detailed_reasoning = reasoning_resp.choices[0].message.content
            
            result = ToolSelectionResult(
                selected_tool=selection_result.get("selected_tool", ""),
                confidence=selection_result.get("confidence", 0.5),
                reasoning=detailed_reasoning,
                parameters=selection_result.get("parameters", {}),
                alternatives=[
                    ToolCandidate(
                        name=alt.get("name", ""),
                        description=TOOLS.get(alt.get("name", ""), {}).get("description", ""),
                        confidence=alt.get("confidence", 0.0),
                        reasoning=alt.get("reasoning", ""),
                        parameters=alt.get("parameters", {})
                    )
                    for alt in selection_result.get("alternatives", [])
                ],
                execution_plan=selection_result.get("execution_plan", "")
            )
            
            # Record to history
            self.tool_selection_history.append({
                "query": user_query,
                "selected_tool": result.selected_tool,
                "confidence": result.confidence,
                "reasoning": result.reasoning[:200] + "..." if len(result.reasoning) > 200 else result.reasoning
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return self._fallback_tool_selection(user_query)

    def _fallback_tool_selection(self, user_query: str) -> ToolSelectionResult:
        """Fallback tool selection"""
        logger.warning("Using fallback tool selection")
        
        # Simple keyword matching
        query_lower = user_query.lower()
        
        if "t-test" in query_lower or "t test" in query_lower:
            if "paired" in query_lower or "dependent" in query_lower:
                selected_tool = "paired_t_test"
            else:
                selected_tool = "independent_t_test"
        elif "anova" in query_lower:
            selected_tool = "one_way_anova"
        elif "mann" in query_lower or "whitney" in query_lower:
            selected_tool = "mann_whitney_u_test"
        elif "wilcoxon" in query_lower:
            selected_tool = "wilcoxon_test"
        elif "kruskal" in query_lower:
            selected_tool = "kruskal_wallis_test"
        elif "bayes" in query_lower:
            selected_tool = "bayes_factor_ttest"
        elif "granger" in query_lower or "causality" in query_lower:
            selected_tool = "granger_causality"
        else:
            selected_tool = "independent_t_test"
        
        # Extract parameters
        extracted_params = self.extract_parameters_from_query(user_query)
        
        return ToolSelectionResult(
            selected_tool=selected_tool,
            confidence=0.5,
            reasoning=f"Fallback selection based on keywords in query: {user_query}",
            parameters=extracted_params,
            alternatives=[],
            execution_plan="Execute the selected tool with extracted parameters"
        )

    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate tool parameters"""
        if tool_name not in TOOLS:
            return False, f"Unknown tool: {tool_name}"
        
        tool_info = TOOLS[tool_name]
        required_params = tool_info.get("args", [])
        
        # Check required parameters
        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter: {param} ({tool_info.get('arg_types', {}).get(param, 'unknown type')})"
        
        return True, "Parameters validated successfully"

    def execute_with_validation(self, tool_selection: ToolSelectionResult) -> Dict[str, Any]:
        """Execute and validate"""
        # Validate parameters
        is_valid, validation_msg = self.validate_tool_parameters(
            tool_selection.selected_tool, 
            tool_selection.parameters
        )
        
        if not is_valid:
            return {
                "error": validation_msg,
                "tool_selection": tool_selection
            }
        
        # Execute tool
        try:
            tool_info = TOOLS[tool_selection.selected_tool]
            func = tool_info["func"]
            
            # Build call parameters based on parameter type
            if "*samples" in tool_selection.parameters:
                # Handle variable arguments
                samples = tool_selection.parameters["*samples"]
                result = func(*samples)
            elif "data" in tool_selection.parameters:
                # Handle data parameters
                data = tool_selection.parameters["data"]
                result = func(data)
            else:
                # Handle regular parameters
                result = func(**tool_selection.parameters)
            
            return {
                "success": True,
                "tool_selection": tool_selection,
                "execution_result": result,
                "validation": validation_msg
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "tool_selection": tool_selection,
                "validation": validation_msg
            }

    def __call__(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """Sync call: user_query -> tool selection reasoning -> execution -> result"""
        logger.info(f"Processing query: {user_query}")
        
        # Tool selection reasoning
        tool_selection = self.select_tools_with_reasoning(user_query)
        
        # Execute and validate
        execution_result = self.execute_with_validation(tool_selection)
        
        return {
            "query": user_query,
            "tool_selection": tool_selection,
            "execution_result": execution_result,
            "confidence": tool_selection.confidence,
            "reasoning": tool_selection.reasoning
        }

    async def acall(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """Async call"""
        logger.info(f"Async processing query: {user_query}")
        
        # Check if in main thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Async tool selection reasoning
        tool_selection = await self._async_select_tools_with_reasoning(user_query)
        
        # Async execute and validate
        execution_result = await self._async_execute_with_validation(tool_selection)
        
        return {
            "query": user_query,
            "tool_selection": tool_selection,
            "execution_result": execution_result,
            "confidence": tool_selection.confidence,
            "reasoning": tool_selection.reasoning
        }

    async def _async_select_tools_with_reasoning(self, user_query: str) -> ToolSelectionResult:
        """Async tool selection reasoning"""
        # Extract parameters
        extracted_params = self.extract_parameters_from_query(user_query)
        
        selection_prompt = self.build_tool_selection_prompt(user_query)
        
        messages = [
            {"role": "system", "content": selection_prompt},
            {"role": "user", "content": user_query}
        ]
        
        call_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        call_kwargs.update(self.generate_args)
        
        if self.async_client:
            resp = await self.async_client.chat.completions.create(**call_kwargs)
        else:
            # Use thread pool to execute sync calls
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, 
                lambda: self.client.chat.completions.create(**call_kwargs)
            )
        
        content = resp.choices[0].message.content
        
        try:
            selection_result = json.loads(content)
        except json.JSONDecodeError:
            selection_result = self._fallback_tool_selection(user_query)
        
        # Merge extracted parameters with LLM-generated parameters
        if extracted_params:
            selection_result["parameters"] = {**selection_result.get("parameters", {}), **extracted_params}
        
        # Async detailed reasoning
        reasoning_prompt = self.build_reasoning_prompt(
            user_query, 
            selection_result.get("selected_tool", "")
        )
        
        reasoning_messages = [
            {"role": "system", "content": reasoning_prompt},
            {"role": "user", "content": user_query}
        ]
        
        if self.async_client:
            reasoning_resp = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=reasoning_messages,
                temperature=0.3
            )
        else:
            loop = asyncio.get_event_loop()
            reasoning_resp = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=reasoning_messages,
                    temperature=0.3
                )
            )
        
        detailed_reasoning = reasoning_resp.choices[0].message.content
        
        result = ToolSelectionResult(
            selected_tool=selection_result.get("selected_tool", ""),
            confidence=selection_result.get("confidence", 0.5),
            reasoning=detailed_reasoning,
            parameters=selection_result.get("parameters", {}),
            alternatives=[
                ToolCandidate(
                    name=alt.get("name", ""),
                    description=TOOLS.get(alt.get("name", ""), {}).get("description", ""),
                    confidence=alt.get("confidence", 0.0),
                    reasoning=alt.get("reasoning", ""),
                    parameters=alt.get("parameters", {})
                )
                for alt in selection_result.get("alternatives", [])
            ],
            execution_plan=selection_result.get("execution_plan", "")
        )
        
        # Record to history
        self.tool_selection_history.append({
            "query": user_query,
            "selected_tool": result.selected_tool,
            "confidence": result.confidence,
            "reasoning": result.reasoning[:200] + "..." if len(result.reasoning) > 200 else result.reasoning
        })
        
        return result

    async def _async_execute_with_validation(self, tool_selection: ToolSelectionResult) -> Dict[str, Any]:
        """Async execute and validate"""
        # Validate parameters
        is_valid, validation_msg = self.validate_tool_parameters(
            tool_selection.selected_tool, 
            tool_selection.parameters
        )
        
        if not is_valid:
            return {
                "error": validation_msg,
                "tool_selection": tool_selection
            }
        
        # Async execute tool
        try:
            tool_info = TOOLS[tool_selection.selected_tool]
            func = tool_info["func"]
            
            # Use thread pool to execute sync functions
            loop = asyncio.get_event_loop()
            
            if "*samples" in tool_selection.parameters:
                # Handle variable arguments
                samples = tool_selection.parameters["*samples"]
                result = await loop.run_in_executor(None, lambda: func(*samples))
            elif "data" in tool_selection.parameters:
                # Handle data parameters
                data = tool_selection.parameters["data"]
                result = await loop.run_in_executor(None, lambda: func(data))
            else:
                # Handle regular parameters
                result = await loop.run_in_executor(None, lambda: func(**tool_selection.parameters))
            
            return {
                "success": True,
                "tool_selection": tool_selection,
                "execution_result": result,
                "validation": validation_msg
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "tool_selection": tool_selection,
                "validation": validation_msg
            }

    def get_tool_selection_history(self) -> List[Dict[str, Any]]:
        """Get tool selection history"""
        return self.tool_selection_history

    def get_tool_usage_statistics(self) -> Dict[str, int]:
        """Get tool usage statistics"""
        stats = {}
        for entry in self.tool_selection_history:
            tool = entry["selected_tool"]
            stats[tool] = stats.get(tool, 0) + 1
        return stats 