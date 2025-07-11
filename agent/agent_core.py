# agent/agent_core.py
from agent.tool_registry import TOOLS

def route_and_run(user_request: dict):
    """
    路由请求到正确工具并调用
    user_request 示例:
        {
            "tool": "independent_t_test",
            "args": [[1,2,3], [4,5,6]],   # 位置参数
            # 或者 {"sample1": [...], "sample2": [...]}  # 关键字参数
        }
    """
    tool_name = user_request["tool"]
    tool_info = TOOLS.get(tool_name)
    if not tool_info:
        raise ValueError(f"Unknown tool: {tool_name}")
    func = tool_info["func"]
    args = user_request.get("args", [])
    kwargs = user_request.get("kwargs", {})
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        return {"error": str(e)}