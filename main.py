# agent/main.py
from agent.agent_core import route_and_run
from agent.stat_agent_llm_adapter import StatAgentLLMAdapter


if __name__ == "__main__":
    # 按你的配置初始化 vLLMChatAdapter
    llm_agent = StatAgentLLMAdapter(
        model_name="/data/pretrain_dir/Qwen2.5-14B-Instruct",
        api_key="EMPTY",  # vLLM随便填
        client_args={"base_url": "http://111.6.167.248:9887/v1/"},  # 不要末尾斜杠！
        generate_args={"temperature": 0}
    )
    result = llm_agent("Perform an independent t-test on [1,2,3,4,5] and [5,6,7,8,9].")
    print(result)
