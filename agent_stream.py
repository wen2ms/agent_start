from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI


@tool(description="Get the demo stock price for a ticker symbol.")
def get_stock_price(symbol: str, reason: str) -> str:
    prices = {"AAPL": 230.50, "NVDA": 180.25, "TSLA": 340.80}
    symbol = symbol.upper()
    price = prices.get(symbol)
    if price is None:
        return f"No demo price data for {symbol}."
    return f"{symbol} stock price: ${price:.2f}."


@tool(description="Get basic demo company information for a ticker symbol.")
def get_company_info(symbol: str, reason: str) -> str:
    companies = {
        "AAPL": "Apple Inc. is a technology company known for the iPhone, Mac, and other consumer electronics.",
        "NVDA": "NVIDIA is a technology company specializing in GPUs and AI computing.",
        "TSLA": "Tesla is an electric vehicle and energy technology company.",
    }
    symbol = symbol.upper()
    info = companies.get(symbol)
    if info is None:
        return f"No demo company information for {symbol}."
    return info


chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)
agent = create_agent(
    model=chat_llm,
    tools=[get_stock_price, get_company_info],
    system_prompt=(
        "You are a stock research assistant. "
        "When calling a tool, always provide a brief reason argument explaining why the tool is needed."
        "Use the provided tools when stock data is required. The tool data is demo data."
    ),
)
question = "Check AAPL's stock price and company information using the tools, then give me a short memory."
input_message = HumanMessage(question)

for chunk in agent.stream(input={"messages": [input_message]}, stream_mode="updates"):
    for update in chunk.values():
        for message in update["messages"]:
            if isinstance(message, AIMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    print("\n" + "=" * 20 + "Reason" + "=" * 20)
                    print(tool_call["args"]["reason"])
                    print("\n" + "=" * 20 + "Action" + "=" * 20)
                    print(f"Tool: {tool_call['name']}")
            elif isinstance(message, ToolMessage):
                print("\n" + "=" * 20 + "Observation" + "=" * 20)
                print(message.content)
            elif isinstance(message, AIMessage):
                print("\n" + "=" * 20 + "Final" + "=" * 20)
                print(message.content)
