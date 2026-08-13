import json

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

schema = [
    "Date",
    "Stock Name",
    "Opening Price",
    "Closing Price",
    "Trading Volume",
]

examples_data = [
    {
        "content": (
            "On July 31, 2026, NVIDIA Corporation opened at USD 198.49. "
            "The stock reached an intraday high of USD 201.90 and fell to "
            "an intraday low of USD 195.09. It closed at USD 200.75, "
            "with a trading volume of 139,961,153 shares."
        ),
        "answers": {
            "Date": "2026-07-31",
            "Stock Name": "NVIDIA Corporation",
            "Opening Price": "USD 198.49",
            "Closing Price": "USD 200.75",
            "Trading Volume": "139,961,153 shares",
        },
    },
    {
        "content": (
            "On July 31, 2026, Apple Inc. opened at USD 304.69. "
            "During the trading session, its price rose to USD 311.87 "
            "and dropped to USD 300.015. The stock eventually closed at "
            "USD 308.91, with a trading volume of 132,489,137 shares."
        ),
        "answers": {
            "Date": "2026-07-31",
            "Stock Name": "Apple Inc.",
            "Opening Price": "USD 304.69",
            "Closing Price": "USD 308.91",
            "Trading Volume": "132,489,137 shares",
        },
    },
]

questions = [
    (
        "On July 31, 2026, Microsoft Corporation opened at USD 450.00. "
        "Its share price reached a high of USD 466.75 and a low of USD 445.82. "
        "It closed at USD 464.72, with a trading volume of 60,845,972 shares."
    ),
    (
        "On July 31, 2026, Tesla Inc. opened at USD 309.70. "
        "The stock reached USD 317.20 before falling to USD 302.16. "
        "It eventually closed at USD 311.21."
    ),
]

client = OpenAI(base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": (
            "You are an information extraction assistant. "
            f"Extract the following fields from each input text: {schema}. "
            "If a field is not mentioned in the source text, use "
            '"Not mentioned in the source text" as its value. '
        ),
    }
]

for example in examples_data:
    messages.append({"role": "user", "content": f"{example['content']}"})
    messages.append({"role": "assistant", "content": json.dumps(example["answers"])})

for question in questions:
    completion = client.chat.completions.create(
        model="glm-5", messages=[*messages, {"role": "user", "content": f"{question}"}]
    )
    print(completion.choices[0].message.content, flush=True)
