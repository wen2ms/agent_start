from urllib.parse import quote

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    url = f"https://wttr.in/{quote(city)}"
    response = requests.get(url, params={"format": "j1"}, timeout=10)
    response.raise_for_status()
    data = response.json()
    curr = data["current_condition"][0]
    temperature = curr["temp_C"]
    feels_like = curr["FeelsLikeC"]
    description = curr["weatherDesc"][0]["value"]

    return f"Weather in {city}: {description}, temperature {temperature}°C, feels like {feels_like}°C."


chat_llm = ChatOpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
)
agent = create_agent(model=chat_llm, tools=[get_weather], system_prompt="You are a helpful weather assistant.")

response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Xi'an?"}]})
for message in response["messages"]:
    print("\n" + "=" * 20 + type(message).__name__ + "=" * 20)
    print(message.content)
