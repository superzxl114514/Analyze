import json
import asyncio
import logging
from typing import Any, Dict, List, Optional

from agent.tool_registry import TOOLS
from agent.agent_core import route_and_run

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    raise ImportError("Please install openai>=1.0.0: pip install openai")

class StatAgentLLMAdapter:
    """
    LLM-powered Statistical Analysis Agent Adapter.
    Compatible with vLLM/OpenAI API (>=1.0.0) via client object,
    supporting base_url and async.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        client_args: dict = None,
        generate_args: dict = None,
        **kwargs
    ):
        """
        Args:
            model_name: Model name as shown in vLLM server, e.g. "Qwen2.5-14B-Instruct".
            api_key: Optional, not needed for vLLM, but required by openai client.
            client_args: dict, must include 'base_url': your vllm api, e.g. "http://host:port/v1"
            generate_args: dict, e.g. {'temperature':0}
        """
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"
        self.generate_args = generate_args or {}
        self.client_args = client_args or {}
        if "base_url" not in self.client_args:
            self.client_args["base_url"] = "http://localhost:8000/v1"
            logger.warning(f"No base_url provided, using default: {self.client_args['base_url']}")

        # New openai client object (>=1.0)
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

    def build_prompt(self, user_query: str) -> str:
        """Build LLM system prompt with tool registry description."""
        tools_desc = "\n".join([
            f"- {name}: {tool['description']}" for name, tool in TOOLS.items()
        ])
        system_prompt = (
            "You are an AI statistical analysis assistant. "
            "You have access to the following statistical tools:\n"
            f"{tools_desc}\n"
            "When the user asks a question, analyze their request and generate a JSON object like:\n"
            '{ "tool": "tool_name", "args": [...], "kwargs": {...} }\n'
            "The args and kwargs should match the function parameters. "
            "Only return the JSON object. If unsure, try to infer the best tool and reasonable parameters."
        )
        return system_prompt

    def parse_llm_output(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM output is not valid JSON: {content}")
            raise

    def __call__(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """
        Sync: user_query->LLM->parse->route_and_run->result
        """
        messages = [
            {"role": "system", "content": self.build_prompt(user_query)},
            {"role": "user", "content": user_query},
        ]
        call_kwargs = dict(
            model=self.model_name,
            messages=messages,
        )
        call_kwargs.update(self.generate_args)
        call_kwargs.update(kwargs)

        logger.info(f"Calling LLM (model={self.model_name}) via base_url={self.client_args['base_url']}")
        resp = self.client.chat.completions.create(**call_kwargs)
        content = resp.choices[0].message.content
        logger.info(f"LLM raw output: {content}")

        parsed = self.parse_llm_output(content)
        result = route_and_run(parsed)
        return {
            "llm_interpretation": parsed,
            "analysis_result": result
        }

    async def acall(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """
        Async: same logic, but with async openai client if possible.
        """
        messages = [
            {"role": "system", "content": self.build_prompt(user_query)},
            {"role": "user", "content": user_query},
        ]
        call_kwargs = dict(
            model=self.model_name,
            messages=messages,
        )
        call_kwargs.update(self.generate_args)
        call_kwargs.update(kwargs)

        logger.info(f"Async calling LLM (model={self.model_name}) via base_url={self.client_args['base_url']}")
        if self.async_client:
            resp = await self.async_client.chat.completions.create(**call_kwargs)
            content = resp.choices[0].message.content
        else:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(**call_kwargs))
            content = resp.choices[0].message.content

        logger.info(f"LLM raw output: {content}")

        parsed = self.parse_llm_output(content)
        result = route_and_run(parsed)
        return {
            "llm_interpretation": parsed,
            "analysis_result": result
        }
