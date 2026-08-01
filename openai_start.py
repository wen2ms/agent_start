from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

client = OpenAI(base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a mathematics expert."},
    {"role": "user", "content": "Hello, what can you help me with?"},
    {
        "role": "assistant",
        "content": (
            "Hello wen2ms. I can help you understand mathematical concepts, "
            "derive formulas, solve equations, and explain numerical methods."
        ),
    },
    {"role": "user", "content": "Give me the Newton-Raphson method."},
]

stream = client.chat.completions.create(
    model="deepseek-v4-pro", messages=messages, reasoning_effort="high", stream=True
)
output_path = Path("openai_start_response.md")
with output_path.open("w") as outfile:
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        outfile.write(delta)
        outfile.flush()
