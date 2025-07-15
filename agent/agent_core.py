# agent/agent_core.py
from agent.tool_registry import TOOLS

def route_and_run(user_request: dict):
    """
    Route request to the correct tool and execute it
    user_request example:
        {
            "tool": "independent_t_test",
            "args": [[1,2,3], [4,5,6]],   # positional arguments
            # or {"sample1": [...], "sample2": [...]}  # keyword arguments
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