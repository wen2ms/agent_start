from collections.abc import Iterator

from agent.context import UserContext
from agent.tools.agent_tools import TOOLS
from agent.tools.middleware import MIDDLEWARE
from langchain.agents import create_agent
from langchain.agents.middleware import InputAgentState
from model.factory import chat_llm


class ReactAgent:
    def __init__(self) -> None:
        self.agent = create_agent(
            model=chat_llm,
            tools=TOOLS,
            middleware=MIDDLEWARE,
            context_schema=UserContext,
        )

    def execute_stream(self, query: str, user_id: str) -> Iterator[str]:
        agent_input: InputAgentState = {"messages": [{"role": "user", "content": query}]}
        context = UserContext(user_id=user_id)
        for state in self.agent.stream(input=agent_input, context=context, stream_mode="values"):
            latest_message = state["messages"][-1]
            if latest_message.content:
                yield latest_message.content + "\n"


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream(
        query=" 请生成我本月的家庭能源报告，并根据能源知识库，针对耗电量最高的类别给出有依据的节能建议。同时告诉我西安的今天天气怎么样？请用中文",
        user_id="user_001",
    ):
        print(chunk, end="", flush=True)
    print()
