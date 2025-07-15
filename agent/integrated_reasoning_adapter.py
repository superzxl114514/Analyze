"""
集成推理适配器
结合统计推理引擎和LLM推理的增强适配器
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import openai
except ImportError:
    raise ImportError("Please install openai>=1.0.0: pip install openai")

from agent.tool_registry import TOOLS
from agent.agent_core import route_and_run
from agent.tool_reasoning_engine import ToolReasoningEngine, ToolRecommendation, ReasoningLevel
from agent.enhanced_llm_adapter import ToolSelectionStrategy


@dataclass
class IntegratedToolSelection:
    """集成工具选择结果"""
    selected_tool: str
    confidence: float
    reasoning: str
    statistical_reasoning: str
    llm_reasoning: str
    parameters: Dict[str, Any]
    alternatives: List[str]
    warnings: List[str]
    execution_plan: str
    reasoning_method: str  # "statistical", "llm", "hybrid"


class IntegratedReasoningAdapter:
    """
    集成推理适配器
    结合统计推理引擎和LLM推理
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client_args: dict = None,
        generate_args: dict = None,
        reasoning_level: ReasoningLevel = ReasoningLevel.ADVANCED,
        use_hybrid_reasoning: bool = True,
        **kwargs
    ):
        """
        Args:
            model_name: 模型名称
            api_key: API密钥
            client_args: 客户端参数
            generate_args: 生成参数
            reasoning_level: 推理级别
            use_hybrid_reasoning: 是否使用混合推理
        """
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"
        self.generate_args = generate_args or {}
        self.client_args = client_args or {}
        self.reasoning_level = reasoning_level
        self.use_hybrid_reasoning = use_hybrid_reasoning
        
        if "base_url" not in self.client_args:
            self.client_args["base_url"] = "http://localhost:8000/v1"
            logger.warning(f"No base_url provided, using default: {self.client_args['base_url']}")

        # OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            **self.client_args
        )
        
        # 异步客户端
        try:
            from openai import AsyncOpenAI
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                **self.client_args
            )
        except ImportError:
            logger.warning("AsyncOpenAI not found, async will fallback to thread.")
            self.async_client = None

        # 初始化推理引擎
        self.reasoning_engine = ToolReasoningEngine(reasoning_level)
        
        # 工具选择历史
        self.tool_selection_history = []
        
        # 初始化工具分类
        self._categorize_tools()

    def _categorize_tools(self):
        """将工具按功能分类"""
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

    def build_hybrid_prompt(self, user_query: str, statistical_recommendation: ToolRecommendation) -> str:
        """构建混合推理提示"""
        tools_desc = "\n".join([
            f"- {name}: {tool['description']}" for name, tool in TOOLS.items()
        ])
        
        categories_desc = "\n".join([
            f"## {category.replace('_', ' ').title()}:\n" + 
            "\n".join([f"  - {tool}: {TOOLS[tool]['description']}" for tool in tools])
            for category, tools in self.tool_categories.items() if tools
        ])
        
        system_prompt = f"""You are an expert statistical analysis assistant. You have access to both statistical reasoning and LLM reasoning capabilities.

## Available Statistical Tools by Category:

{categories_desc}

## Statistical Reasoning Result:
The statistical reasoning engine has analyzed the query and provided the following recommendation:
- Recommended Tool: {statistical_recommendation.tool_name}
- Confidence: {statistical_recommendation.confidence:.3f}
- Reasoning: {statistical_recommendation.reasoning}
- Warnings: {', '.join(statistical_recommendation.warnings) if statistical_recommendation.warnings else 'None'}
- Alternatives: {', '.join(statistical_recommendation.alternatives) if statistical_recommendation.alternatives else 'None'}

## Your Task:
Consider both the statistical reasoning and the user's specific requirements to make the final tool selection. You can:
1. Accept the statistical recommendation if it's appropriate
2. Choose an alternative if the user's context suggests a different approach
3. Provide additional reasoning based on the user's specific needs

## Response Format:
Return a JSON object with the following structure:
{{
    "selected_tool": "tool_name",
    "confidence": 0.95,
    "reasoning": "Your reasoning for the final selection",
    "statistical_agreement": true/false,
    "parameters": {{
        "sample1": [1, 2, 3],
        "sample2": [4, 5, 6],
        "equal_var": true
    }},
    "alternatives": [
        {{
            "name": "alternative_tool",
            "confidence": 0.7,
            "reasoning": "Why this alternative was considered"
        }}
    ],
    "execution_plan": "Step-by-step execution plan",
    "additional_considerations": "Any additional considerations or warnings"
}}

## Important Guidelines:
- Always consider the statistical reasoning as a strong baseline
- Provide confidence scores (0-1) for your selection
- Explain why you agree or disagree with the statistical recommendation
- Include any additional considerations based on the user's context
- Ensure parameters match the tool's requirements
"""
        return system_prompt

    def integrate_reasoning(self, user_query: str) -> IntegratedToolSelection:
        """集成推理：结合统计推理和LLM推理"""
        logger.info("Starting integrated reasoning process...")
        
        # 第一步：统计推理
        logger.info("Step 1: Statistical reasoning")
        statistical_recommendation = self.reasoning_engine.get_best_tool(user_query)
        
        # 第二步：LLM推理（考虑统计推理结果）
        logger.info("Step 2: LLM reasoning with statistical context")
        hybrid_prompt = self.build_hybrid_prompt(user_query, statistical_recommendation)
        
        messages = [
            {"role": "system", "content": hybrid_prompt},
            {"role": "user", "content": user_query}
        ]
        
        call_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        call_kwargs.update(self.generate_args)
        
        resp = self.client.chat.completions.create(**call_kwargs)
        content = resp.choices[0].message.content
        
        # 解析LLM响应
        try:
            llm_result = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {content}")
            # 回退到统计推理结果
            llm_result = {
                "selected_tool": statistical_recommendation.tool_name,
                "confidence": statistical_recommendation.confidence,
                "reasoning": "Using statistical reasoning due to LLM parsing error",
                "statistical_agreement": True,
                "parameters": {},
                "alternatives": [],
                "execution_plan": "Execute based on statistical recommendation",
                "additional_considerations": "LLM parsing failed, using statistical reasoning"
            }
        
        # 第三步：整合结果
        logger.info("Step 3: Integrating results")
        
        # 确定最终选择
        final_tool = llm_result.get("selected_tool", statistical_recommendation.tool_name)
        final_confidence = llm_result.get("confidence", statistical_recommendation.confidence)
        
        # 确定推理方法
        statistical_agreement = llm_result.get("statistical_agreement", False)
        if statistical_agreement:
            reasoning_method = "hybrid" if self.use_hybrid_reasoning else "statistical"
        else:
            reasoning_method = "llm"
        
        # 构建最终结果
        integrated_result = IntegratedToolSelection(
            selected_tool=final_tool,
            confidence=final_confidence,
            reasoning=llm_result.get("reasoning", ""),
            statistical_reasoning=statistical_recommendation.reasoning,
            llm_reasoning=llm_result.get("reasoning", ""),
            parameters=llm_result.get("parameters", {}),
            alternatives=[
                alt.get("name", "") for alt in llm_result.get("alternatives", [])
            ],
            warnings=statistical_recommendation.warnings,
            execution_plan=llm_result.get("execution_plan", ""),
            reasoning_method=reasoning_method
        )
        
        # 记录选择历史
        self.tool_selection_history.append({
            "query": user_query,
            "selected_tool": integrated_result.selected_tool,
            "confidence": integrated_result.confidence,
            "reasoning_method": integrated_result.reasoning_method,
            "statistical_agreement": statistical_agreement,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return integrated_result

    def validate_and_execute(self, tool_selection: IntegratedToolSelection) -> Dict[str, Any]:
        """验证并执行工具"""
        # 验证参数
        is_valid, validation_msg = self._validate_tool_parameters(
            tool_selection.selected_tool, 
            tool_selection.parameters
        )
        
        if not is_valid:
            return {
                "error": validation_msg,
                "tool_selection": tool_selection
            }
        
        # 执行工具
        try:
            execution_request = {
                "tool": tool_selection.selected_tool,
                "args": list(tool_selection.parameters.values()),
                "kwargs": {}
            }
            
            result = route_and_run(execution_request)
            
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

    def _validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """验证工具参数"""
        tool_info = TOOLS.get(tool_name)
        if not tool_info:
            return False, f"Unknown tool: {tool_name}"
        
        # 基本参数验证
        required_params = tool_info.get("parameters", [])
        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"
        
        return True, "Parameters valid"

    def __call__(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """同步调用：集成推理 -> 验证执行 -> 结果"""
        logger.info(f"Processing query with integrated reasoning: {user_query}")
        
        # 集成推理
        tool_selection = self.integrate_reasoning(user_query)
        
        # 验证并执行
        execution_result = self.validate_and_execute(tool_selection)
        
        return {
            "query": user_query,
            "tool_selection": tool_selection,
            "execution_result": execution_result,
            "confidence": tool_selection.confidence,
            "reasoning_method": tool_selection.reasoning_method
        }

    async def acall(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """异步调用"""
        logger.info(f"Async processing query with integrated reasoning: {user_query}")
        
        # 异步集成推理
        tool_selection = await self._async_integrate_reasoning(user_query)
        
        # 异步验证并执行
        execution_result = await self._async_validate_and_execute(tool_selection)
        
        return {
            "query": user_query,
            "tool_selection": tool_selection,
            "execution_result": execution_result,
            "confidence": tool_selection.confidence,
            "reasoning_method": tool_selection.reasoning_method
        }

    async def _async_integrate_reasoning(self, user_query: str) -> IntegratedToolSelection:
        """异步集成推理"""
        # 统计推理（同步）
        statistical_recommendation = self.reasoning_engine.get_best_tool(user_query)
        
        # 异步LLM推理
        hybrid_prompt = self.build_hybrid_prompt(user_query, statistical_recommendation)
        
        messages = [
            {"role": "system", "content": hybrid_prompt},
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
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, 
                lambda: self.client.chat.completions.create(**call_kwargs)
            )
        
        content = resp.choices[0].message.content
        
        try:
            llm_result = json.loads(content)
        except json.JSONDecodeError:
            llm_result = {
                "selected_tool": statistical_recommendation.tool_name,
                "confidence": statistical_recommendation.confidence,
                "reasoning": "Using statistical reasoning due to LLM parsing error",
                "statistical_agreement": True,
                "parameters": {},
                "alternatives": [],
                "execution_plan": "Execute based on statistical recommendation",
                "additional_considerations": "LLM parsing failed, using statistical reasoning"
            }
        
        # 整合结果
        final_tool = llm_result.get("selected_tool", statistical_recommendation.tool_name)
        final_confidence = llm_result.get("confidence", statistical_recommendation.confidence)
        statistical_agreement = llm_result.get("statistical_agreement", False)
        
        reasoning_method = "hybrid" if statistical_agreement and self.use_hybrid_reasoning else "llm"
        
        integrated_result = IntegratedToolSelection(
            selected_tool=final_tool,
            confidence=final_confidence,
            reasoning=llm_result.get("reasoning", ""),
            statistical_reasoning=statistical_recommendation.reasoning,
            llm_reasoning=llm_result.get("reasoning", ""),
            parameters=llm_result.get("parameters", {}),
            alternatives=[
                alt.get("name", "") for alt in llm_result.get("alternatives", [])
            ],
            warnings=statistical_recommendation.warnings,
            execution_plan=llm_result.get("execution_plan", ""),
            reasoning_method=reasoning_method
        )
        
        return integrated_result

    async def _async_validate_and_execute(self, tool_selection: IntegratedToolSelection) -> Dict[str, Any]:
        """异步验证并执行"""
        is_valid, validation_msg = self._validate_tool_parameters(
            tool_selection.selected_tool, 
            tool_selection.parameters
        )
        
        if not is_valid:
            return {
                "error": validation_msg,
                "tool_selection": tool_selection
            }
        
        try:
            execution_request = {
                "tool": tool_selection.selected_tool,
                "args": list(tool_selection.parameters.values()),
                "kwargs": {}
            }
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: route_and_run(execution_request)
            )
            
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

    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        stats = {
            "total_queries": len(self.tool_selection_history),
            "reasoning_methods": {},
            "tool_usage": {},
            "statistical_agreement_rate": 0.0
        }
        
        if not self.tool_selection_history:
            return stats
        
        # 统计推理方法
        for entry in self.tool_selection_history:
            method = entry.get("reasoning_method", "unknown")
            stats["reasoning_methods"][method] = stats["reasoning_methods"].get(method, 0) + 1
            
            tool = entry.get("selected_tool", "unknown")
            stats["tool_usage"][tool] = stats["tool_usage"].get(tool, 0) + 1
        
        # 计算统计一致性率
        agreement_count = sum(1 for entry in self.tool_selection_history 
                           if entry.get("statistical_agreement", False))
        stats["statistical_agreement_rate"] = agreement_count / len(self.tool_selection_history)
        
        return stats

    def explain_reasoning_process(self, user_query: str) -> str:
        """解释推理过程"""
        # 获取统计推理
        statistical_recommendation = self.reasoning_engine.get_best_tool(user_query)
        
        # 获取集成推理结果
        integrated_result = self.integrate_reasoning(user_query)
        
        explanation = f"""
## 集成推理过程解释

### 用户查询
{user_query}

### 第一步：统计推理
- 推荐工具：{statistical_recommendation.tool_name}
- 置信度：{statistical_recommendation.confidence:.3f}
- 推理：{statistical_recommendation.reasoning}
- 警告：{', '.join(statistical_recommendation.warnings) if statistical_recommendation.warnings else '无'}

### 第二步：LLM推理
- 最终选择：{integrated_result.selected_tool}
- 最终置信度：{integrated_result.confidence:.3f}
- LLM推理：{integrated_result.llm_reasoning}
- 推理方法：{integrated_result.reasoning_method}

### 第三步：结果整合
- 统计一致性：{'是' if integrated_result.selected_tool == statistical_recommendation.tool_name else '否'}
- 执行计划：{integrated_result.execution_plan}
- 替代方案：{', '.join(integrated_result.alternatives) if integrated_result.alternatives else '无'}
"""
        
        return explanation 