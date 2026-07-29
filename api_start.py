from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

client = OpenAI(
    base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Who are you?"}]
completion = client.chat.completions.create(
    model="qwen3.7-max", messages=messages, extra_body={"enable_thinking": True}, stream=True
)
is_answering = False
print("\n" + "=" * 20 + "Thinking" + "=" * 20)
for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    reasoning_content = getattr(delta, "reasoning_content", None)
    if reasoning_content and not is_answering:
        print(reasoning_content, end="", flush=True)
    if delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Answer" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)
