from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from langgraph.types import Command


@tool(description="Add two numbers.")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Multiply two numbers.")
def multiply(a: float, b: float) -> float:
    return a * b


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("\n" + "=" * 20 + "before_agent" + "=" * 20)
    print("Agent starts")
    return None


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("\n" + "=" * 20 + "before_model" + "=" * 20)
    print(f"Message count: {len(state['messages'])}")
    return None


@wrap_model_call
def log_model_call(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    print("\n" + "=" * 20 + "wrap_model_call before" + "=" * 20)
    response = handler(request)
    print("\n" + "=" * 20 + "wrap_model_call after" + "=" * 20)
    return response


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("\n" + "=" * 20 + "after_model" + "=" * 20)
    message = state["messages"][-1]
    print(f"Message type: {type(message).__name__}")
    print(f"Content: {message.content}")
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {[tool_call['name'] for tool_call in message.tool_calls]}")
    return None


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("\n" + "=" * 20 + "after_agent" + "=" * 20)
    print("Agent finished")
    return None


@wrap_tool_call
def log_tool_call(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    print("\n" + "=" * 20 + "wrap_tool_call before" + "=" * 20)
    print(f"Tool: {request.tool_call['name']}")
    print(f"Args: {request.tool_call['args']}")
    response = handler(request)
    print("\n" + "=" * 20 + "wrap_tool_call after" + "=" * 20)
    return response


chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)
agent = create_agent(
    model=chat_llm,
    tools=[add, multiply],
    system_prompt="You are a calculator assistant. Always use the provided tools for calculation. Call only one tool at a time.",
    middleware=[log_before_agent, log_before_model, log_model_call, log_after_model, log_tool_call, log_after_agent],
)
response = agent.invoke(
    input={"messages": [{"role": "user", "content": "First add 10 and 20, then multiply the result by 3."}]}
)
print("\n" + "=" * 20 + "Answer" + "=" * 20)
print(response["messages"][-1].content)
