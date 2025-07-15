"""
工具推理引擎
提供高级的工具选择推理功能
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from agent.tool_registry import TOOLS


class ReasoningLevel(Enum):
    """推理级别"""
    BASIC = "basic"  # 基础推理
    INTERMEDIATE = "intermediate"  # 中级推理
    ADVANCED = "advanced"  # 高级推理
    EXPERT = "expert"  # 专家级推理


@dataclass
class StatisticalContext:
    """统计上下文"""
    data_type: str  # continuous, categorical, time_series
    sample_size: int
    number_of_groups: int
    research_design: str  # between_subjects, within_subjects, mixed
    dependent_variables: int  # 1 for univariate, >1 for multivariate
    normality_assumption: Optional[bool] = None
    homogeneity_assumption: Optional[bool] = None
    missing_data: bool = False
    outliers: bool = False


@dataclass
class ToolRecommendation:
    """工具推荐"""
    tool_name: str
    confidence: float
    reasoning: str
    assumptions_check: Dict[str, bool]
    alternatives: List[str]
    warnings: List[str]
    sample_size_requirement: Optional[int] = None


class ToolReasoningEngine:
    """
    工具推理引擎
    基于统计原理和最佳实践进行工具选择
    """
    
    def __init__(self, reasoning_level: ReasoningLevel = ReasoningLevel.ADVANCED):
        self.reasoning_level = reasoning_level
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """初始化知识库"""
        self.tool_knowledge = {
            "independent_t_test": {
                "data_type": ["continuous"],
                "sample_size_min": 2,
                "number_of_groups": 2,
                "research_design": ["between_subjects"],
                "assumptions": ["normality", "homogeneity"],
                "alternatives": ["mann_whitney_u_test"],
                "description": "比较两个独立组的均值"
            },
            "paired_t_test": {
                "data_type": ["continuous"],
                "sample_size_min": 2,
                "number_of_groups": 2,
                "research_design": ["within_subjects"],
                "assumptions": ["normality"],
                "alternatives": ["wilcoxon_test"],
                "description": "比较两个相关组的均值"
            },
            "one_way_anova": {
                "data_type": ["continuous"],
                "sample_size_min": 3,
                "number_of_groups": [3, "many"],
                "research_design": ["between_subjects"],
                "assumptions": ["normality", "homogeneity"],
                "alternatives": ["kruskal_wallis_test"],
                "description": "比较多个独立组的均值"
            },
            "mann_whitney_u_test": {
                "data_type": ["continuous", "ordinal"],
                "sample_size_min": 2,
                "number_of_groups": 2,
                "research_design": ["between_subjects"],
                "assumptions": [],
                "alternatives": ["independent_t_test"],
                "description": "非参数比较两个独立组"
            },
            "wilcoxon_test": {
                "data_type": ["continuous", "ordinal"],
                "sample_size_min": 2,
                "number_of_groups": 2,
                "research_design": ["within_subjects"],
                "assumptions": [],
                "alternatives": ["paired_t_test"],
                "description": "非参数比较两个相关组"
            },
            "kruskal_wallis_test": {
                "data_type": ["continuous", "ordinal"],
                "sample_size_min": 3,
                "number_of_groups": [3, "many"],
                "research_design": ["between_subjects"],
                "assumptions": [],
                "alternatives": ["one_way_anova"],
                "description": "非参数比较多个独立组"
            },
            "chi2_independence_test": {
                "data_type": ["categorical"],
                "sample_size_min": 5,
                "number_of_groups": [2, "many"],
                "research_design": ["between_subjects"],
                "assumptions": ["expected_frequencies"],
                "alternatives": ["fisher_exact_test"],
                "description": "检验分类变量间的独立性"
            },
            "fisher_exact_test": {
                "data_type": ["categorical"],
                "sample_size_min": 2,
                "number_of_groups": 2,
                "research_design": ["between_subjects"],
                "assumptions": [],
                "alternatives": ["chi2_independence_test"],
                "description": "小样本分类变量独立性检验"
            },
            "granger_causality": {
                "data_type": ["time_series"],
                "sample_size_min": 10,
                "number_of_groups": 2,
                "research_design": ["time_series"],
                "assumptions": ["stationarity"],
                "alternatives": ["time_series_correlation"],
                "description": "时间序列因果关系检验"
            },
            "bayes_factor_ttest": {
                "data_type": ["continuous"],
                "sample_size_min": 2,
                "number_of_groups": 2,
                "research_design": ["between_subjects", "within_subjects"],
                "assumptions": [],
                "alternatives": ["independent_t_test", "paired_t_test"],
                "description": "贝叶斯因子t检验"
            }
        }
    
    def extract_statistical_context(self, query: str) -> StatisticalContext:
        """从查询中提取统计上下文"""
        query_lower = query.lower()
        
        # 数据类型识别
        data_type = "continuous"  # 默认
        if "categorical" in query_lower or "category" in query_lower:
            data_type = "categorical"
        elif "time series" in query_lower or "time" in query_lower:
            data_type = "time_series"
        
        # 样本大小估计
        sample_size = self._estimate_sample_size(query)
        
        # 组数识别
        number_of_groups = self._extract_number_of_groups(query)
        
        # 研究设计识别
        research_design = "between_subjects"  # 默认
        if any(word in query_lower for word in ["paired", "related", "same", "within"]):
            research_design = "within_subjects"
        elif "mixed" in query_lower:
            research_design = "mixed"
        
        # 因变量数量
        dependent_variables = 1  # 默认单变量
        if any(word in query_lower for word in ["multivariate", "multiple dependent", "several variables"]):
            dependent_variables = 2
        
        return StatisticalContext(
            data_type=data_type,
            sample_size=sample_size,
            number_of_groups=number_of_groups,
            research_design=research_design,
            dependent_variables=dependent_variables
        )
    
    def _estimate_sample_size(self, query: str) -> int:
        """估计样本大小"""
        # 从查询中提取数字
        numbers = re.findall(r'\d+', query)
        if numbers:
            return max(map(int, numbers))
        return 30  # 默认值
    
    def _extract_number_of_groups(self, query: str) -> int:
        """提取组数"""
        query_lower = query.lower()
        
        if "two" in query_lower or "2" in query_lower:
            return 2
        elif "three" in query_lower or "3" in query_lower:
            return 3
        elif any(word in query_lower for word in ["multiple", "several", "many"]):
            return 4  # 多个组
        else:
            return 2  # 默认两组
    
    def find_compatible_tools(self, context: StatisticalContext) -> List[Tuple[str, float]]:
        """找到兼容的工具"""
        compatible_tools = []
        
        for tool_name, tool_info in self.tool_knowledge.items():
            compatibility_score = 0.0
            
            # 数据类型匹配
            if context.data_type in tool_info["data_type"]:
                compatibility_score += 0.3
            
            # 组数匹配
            if isinstance(tool_info["number_of_groups"], list):
                if context.number_of_groups in tool_info["number_of_groups"]:
                    compatibility_score += 0.3
            elif context.number_of_groups == tool_info["number_of_groups"]:
                compatibility_score += 0.3
            
            # 研究设计匹配
            if context.research_design in tool_info["research_design"]:
                compatibility_score += 0.2
            
            # 样本大小检查
            if context.sample_size >= tool_info["sample_size_min"]:
                compatibility_score += 0.1
            
            # 因变量数量匹配
            if context.dependent_variables == 1:  # 单变量
                compatibility_score += 0.1
            elif context.dependent_variables > 1 and "multivariate" in tool_name:
                compatibility_score += 0.1
            
            if compatibility_score > 0.3:  # 最低兼容性阈值
                compatible_tools.append((tool_name, compatibility_score))
        
        # 按兼容性分数排序
        compatible_tools.sort(key=lambda x: x[1], reverse=True)
        return compatible_tools
    
    def assess_assumptions(self, tool_name: str, context: StatisticalContext) -> Dict[str, bool]:
        """评估统计假设"""
        tool_info = self.tool_knowledge.get(tool_name, {})
        assumptions = tool_info.get("assumptions", [])
        
        assumption_results = {}
        
        for assumption in assumptions:
            if assumption == "normality":
                # 基于样本大小的正态性假设评估
                if context.sample_size >= 30:
                    assumption_results[assumption] = True
                elif context.sample_size >= 15:
                    assumption_results[assumption] = None  # 需要进一步检验
                else:
                    assumption_results[assumption] = False
            
            elif assumption == "homogeneity":
                # 方差齐性假设
                if context.sample_size >= 20:
                    assumption_results[assumption] = True
                else:
                    assumption_results[assumption] = None
            
            elif assumption == "expected_frequencies":
                # 期望频数假设（卡方检验）
                if context.sample_size >= 20:
                    assumption_results[assumption] = True
                else:
                    assumption_results[assumption] = False
            
            elif assumption == "stationarity":
                # 平稳性假设（时间序列）
                if context.sample_size >= 50:
                    assumption_results[assumption] = True
                else:
                    assumption_results[assumption] = None
        
        return assumption_results
    
    def generate_recommendations(self, query: str) -> List[ToolRecommendation]:
        """生成工具推荐"""
        context = self.extract_statistical_context(query)
        compatible_tools = self.find_compatible_tools(context)
        
        recommendations = []
        
        for tool_name, compatibility_score in compatible_tools:
            # 评估假设
            assumptions_check = self.assess_assumptions(tool_name, context)
            
            # 生成推理
            reasoning = self._generate_reasoning(tool_name, context, assumptions_check)
            
            # 生成警告
            warnings = self._generate_warnings(tool_name, context, assumptions_check)
            
            # 计算置信度
            confidence = self._calculate_confidence(compatibility_score, assumptions_check)
            
            # 获取替代方案
            tool_info = self.tool_knowledge.get(tool_name, {})
            alternatives = tool_info.get("alternatives", [])
            
            recommendation = ToolRecommendation(
                tool_name=tool_name,
                confidence=confidence,
                reasoning=reasoning,
                assumptions_check=assumptions_check,
                alternatives=alternatives,
                warnings=warnings,
                sample_size_requirement=tool_info.get("sample_size_min")
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_reasoning(self, tool_name: str, context: StatisticalContext, 
                           assumptions_check: Dict[str, bool]) -> str:
        """生成推理说明"""
        tool_info = self.tool_knowledge.get(tool_name, {})
        
        reasoning_parts = []
        
        # 基本推理
        reasoning_parts.append(f"选择 {tool_name} 的原因：")
        reasoning_parts.append(f"- 数据类型：{context.data_type}")
        reasoning_parts.append(f"- 组数：{context.number_of_groups}")
        reasoning_parts.append(f"- 研究设计：{context.research_design}")
        reasoning_parts.append(f"- 样本大小：{context.sample_size}")
        
        # 假设评估
        if assumptions_check:
            reasoning_parts.append("\n假设评估：")
            for assumption, status in assumptions_check.items():
                if status is True:
                    reasoning_parts.append(f"- {assumption}：满足")
                elif status is False:
                    reasoning_parts.append(f"- {assumption}：不满足")
                else:
                    reasoning_parts.append(f"- {assumption}：需要进一步检验")
        
        # 替代方案
        alternatives = tool_info.get("alternatives", [])
        if alternatives:
            reasoning_parts.append(f"\n替代方案：{', '.join(alternatives)}")
        
        return "\n".join(reasoning_parts)
    
    def _generate_warnings(self, tool_name: str, context: StatisticalContext,
                          assumptions_check: Dict[str, bool]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        # 样本大小警告
        tool_info = self.tool_knowledge.get(tool_name, {})
        min_sample_size = tool_info.get("sample_size_min", 0)
        
        if context.sample_size < min_sample_size:
            warnings.append(f"样本大小({context.sample_size})小于推荐值({min_sample_size})")
        
        # 假设警告
        for assumption, status in assumptions_check.items():
            if status is False:
                warnings.append(f"不满足{assumption}假设")
            elif status is None:
                warnings.append(f"需要检验{assumption}假设")
        
        # 数据类型警告
        if context.data_type not in tool_info.get("data_type", []):
            warnings.append(f"数据类型({context.data_type})可能不适合此工具")
        
        return warnings
    
    def _calculate_confidence(self, compatibility_score: float, 
                            assumptions_check: Dict[str, bool]) -> float:
        """计算置信度"""
        confidence = compatibility_score
        
        # 基于假设满足情况调整置信度
        satisfied_assumptions = sum(1 for status in assumptions_check.values() if status is True)
        total_assumptions = len(assumptions_check)
        
        if total_assumptions > 0:
            assumption_ratio = satisfied_assumptions / total_assumptions
            confidence *= (0.7 + 0.3 * assumption_ratio)
        
        return min(confidence, 1.0)  # 确保不超过1.0
    
    def get_best_tool(self, query: str) -> ToolRecommendation:
        """获取最佳工具推荐"""
        recommendations = self.generate_recommendations(query)
        
        if not recommendations:
            # 如果没有找到合适的工具，返回默认推荐
            return ToolRecommendation(
                tool_name="independent_t_test",
                confidence=0.5,
                reasoning="未找到完全匹配的工具，使用默认推荐",
                assumptions_check={},
                alternatives=[],
                warnings=["这是默认推荐，请根据具体情况调整"]
            )
        
        # 返回置信度最高的推荐
        return max(recommendations, key=lambda x: x.confidence)
    
    def explain_tool_selection(self, query: str, selected_tool: str) -> str:
        """详细解释工具选择"""
        context = self.extract_statistical_context(query)
        recommendations = self.generate_recommendations(query)
        
        # 找到选中的工具
        selected_recommendation = None
        for rec in recommendations:
            if rec.tool_name == selected_tool:
                selected_recommendation = rec
                break
        
        if not selected_recommendation:
            return f"未找到工具 '{selected_tool}' 的推荐信息"
        
        explanation = f"""
## 工具选择详细解释

### 查询分析
- 原始查询：{query}
- 数据类型：{context.data_type}
- 样本大小：{context.sample_size}
- 组数：{context.number_of_groups}
- 研究设计：{context.research_design}
- 因变量数量：{context.dependent_variables}

### 选择的工具：{selected_tool}
- 置信度：{selected_recommendation.confidence:.3f}
- 推理：{selected_recommendation.reasoning}

### 假设评估
"""
        
        for assumption, status in selected_recommendation.assumptions_check.items():
            status_text = "满足" if status else "不满足" if status is False else "需要检验"
            explanation += f"- {assumption}：{status_text}\n"
        
        if selected_recommendation.warnings:
            explanation += "\n### 警告\n"
            for warning in selected_recommendation.warnings:
                explanation += f"- {warning}\n"
        
        if selected_recommendation.alternatives:
            explanation += f"\n### 替代方案\n"
            for alt in selected_recommendation.alternatives:
                explanation += f"- {alt}\n"
        
        return explanation


# 使用示例
if __name__ == "__main__":
    engine = ToolReasoningEngine()
    
    test_queries = [
        "Perform t-test on two groups with 20 samples each",
        "Compare three groups using ANOVA with 15 samples per group",
        "Test correlation between time series data",
        "Analyze categorical data with small sample size"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print(f"{'='*50}")
        
        best_tool = engine.get_best_tool(query)
        print(f"推荐工具: {best_tool.tool_name}")
        print(f"置信度: {best_tool.confidence:.3f}")
        print(f"推理: {best_tool.reasoning}")
        
        if best_tool.warnings:
            print("警告:")
            for warning in best_tool.warnings:
                print(f"  - {warning}")
        
        # 详细解释
        explanation = engine.explain_tool_selection(query, best_tool.tool_name)
        print(f"\n详细解释:\n{explanation}") 