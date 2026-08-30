from collections.abc import Callable
from typing import Any

from agent.context import UserContext
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import PromptKey, load_prompt


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    logger.info("[monitor_tool] Tools: %s", tool_name)
    logger.info("[monitor_tool] Args: %s", request.tool_call["args"])
    response = handler(request)
    logger.info("[monitor_tool] Tools %s call successfully.", tool_name)
    if (
        tool_name == "prepare_energy_report_context"
        and isinstance(response, ToolMessage)
        and response.status == "success"
    ):
        context = request.runtime.context
        if context is not None:
            context.report = True
            logger.info("[monitor_tool] Switched to report prompt.")
    return response


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    logger.info("[log_before_model] Before model, message count: %d", len(state["messages"]))
    return None


MAIN_PROMPT = load_prompt(PromptKey.MAIN)
REPORT_PROMPT = load_prompt(PromptKey.REPORT)


@dynamic_prompt
def switch_report_prompt(request: ModelRequest[UserContext]) -> str:
    if request.runtime is None or request.runtime.context is None:
        return MAIN_PROMPT
    if request.runtime.context.report:
        return REPORT_PROMPT
    return MAIN_PROMPT


MIDDLEWARE = [monitor_tool, log_before_model, switch_report_prompt]
