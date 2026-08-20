from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def invoke() -> None:
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="glm-5"
    )
    response = chat_llm.invoke(input="Who are you")
    print(response)


def stream() -> None:
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="glm-5"
    )
    response = chat_llm.stream(input="Who are you")
    for chunk in response:
        print(chunk.content, end="", flush=True)


def chat() -> None:
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="glm-5"
    )
    # messages = [
    #     SystemMessage(content="You are an English poet."),
    #     HumanMessage(content="Write a short poem about autumn."),
    #     AIMessage(
    #         content=(
    #             "Golden leaves drift through the air,\n"
    #             "Cool winds wander everywhere.\n"
    #             "Fields grow quiet, skies turn gray,\n"
    #             "Summer softly fades away."
    #         )
    #     ),
    #     HumanMessage(content="Following the style of your previous response, write a short poem about winter."),
    # ]

    # messages = [
    #     ("system", "You are an English poet."),
    #     ("human", "Write a short poem about autumn."),
    #     (
    #         "ai",
    #         (
    #             "Golden leaves drift through the air,\n"
    #             "Cool winds wander everywhere.\n"
    #             "Fields grow quiet, skies turn gray,\n"
    #             "Summer softly fades away."
    #         ),
    #     ),
    #     ("human", "Following the style of your previous response, write a short poem about winter."),
    # ]

    messages = [
        {"role": "system", "content": "You are an English poet."},
        {"role": "human", "content": "Write a short poem about autumn."},
        {
            "role": "ai",
            "content": (
                "Golden leaves drift through the air,\n"
                "Cool winds wander everywhere.\n"
                "Fields grow quiet, skies turn gray,\n"
                "Summer softly fades away."
            ),
        },
        {"role": "human", "content": "Following the style of your previous response, write a short poem about winter."},
    ]
    response = chat_llm.stream(input=messages)
    for chunk in response:
        print(chunk.content, end="", flush=True)


def embed() -> None:
    # embedding = DashScopeEmbeddings(
    #     model="text-embedding-v4",
    # )
    embedding = OpenAIEmbeddings(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        check_embedding_ctx_length=False,
    )
    text = "Hello wen2ms!"
    query_result = embedding.embed_query(text)
    print(f"The length of text vector: {len(query_result)}")
    texts = ["Hi there!", "Oh, hello!", "What's your name?", "My friends call me World", "Hello World!"]
    document_result = embedding.embed_documents(texts)
    print(f"The number of texts vectors: {len(document_result)}, the length of text vector: {len(document_result[0])}")


def prompt_demo() -> None:
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="glm-5"
    )
    # prompt = PromptTemplate.from_template("Explain {topic} to a {level} learner.")
    # prompt_value = prompt.invoke({"topic": "recursion", "level": "beginner"})
    # response = chat_llm.invoke(input=prompt_value)
    # print(response.content)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a professional {role}."),
            MessagesPlaceholder("history"),
            ("human", "Explain {topic} to a {level} learner."),
        ]
    )
    history = [
        ("human", "I already know Python functions."),
        ("ai", "Great. Then I can explain concepts based on your knowledge of functions."),
        ("human", "I also know that functions can be passed as arguments."),
        ("ai", "Excellent. That will make decorators much easier to understand."),
    ]
    prompt_value = prompt.invoke(
        input={"role": "Python teacher", "topic": "decorators", "level": "beginner", "history": history}
    )
    response = chat_llm.invoke(input=prompt_value)
    print(response.content)


def fewshot_prompt_demo() -> None:
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
    )

    examples = [{"input": "big", "antonym": "small"}, {"input": "up", "antonym": "down"}]
    # example_prompt = PromptTemplate.from_template("Question: {input}, Answer: {antonym}")
    # prompt = FewShotPromptTemplate(
    #     examples=examples,
    #     example_prompt=example_prompt,
    #     prefix="Answer the following question.",
    #     suffix="Question: {input}, Answer:",
    #     input_variables=["input"],
    # )
    # prompt_value = prompt.invoke(input={"input": "left"})
    # response = chat_llm.invoke(input=prompt_value)
    # print(response.content)

    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{antonym}")])
    few_shot_prompt = FewShotChatMessagePromptTemplate(examples=examples, example_prompt=example_prompt)
    prompt = ChatPromptTemplate.from_messages([("system", "You are a teacher."), few_shot_prompt, ("human", "{input}")])
    prompt_value = prompt.invoke(input={"input": "left"})
    response = chat_llm.invoke(input=prompt_value)
    print(response.content)


def chain_demo() -> None:
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a professional {role}."),
            ("placeholder", "{conversation}"),
            ("human", "Explain {topic} to a {level} learner."),
        ]
    )
    chain = prompt | chat_llm
    conversation = [
        ("human", "I already understand Python functions."),
        ("ai", "Great. That will help you understand decorators."),
        ("human", "I also know functions can be passed as arguments."),
        ("ai", "Excellent. Then we can build on that idea."),
    ]
    response = chain.stream(
        input={"role": "Python teacher", "topic": "decorators", "level": "beginner", "history": conversation}
    )
    for chunk in response:
        print(chunk.content, end="", flush=True)


def parser_demo() -> None:
    str_parser = StrOutputParser()
    json_parser = JsonOutputParser()
    chat_llm = ChatOpenAI(
        base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", model="qwen3.7-flash"
    )
    prompt1 = PromptTemplate.from_template(
        """Analyze the Python special method {method}.

Return your answer as JSON in exactly this format:
{{
    "method": "the complete method name",
    "abbreviation": "the part between the underscores"
}}"""
    )

    def make_question_func(data: dict[str, str]) -> dict[str, str]:
        return {"question": (f'What does "{data["abbreviation"]}" in "{data["method"]}" stand for in Python?')}

    # make_question = RunnableLambda(
    #     lambda data: {"question": (f'What does "{data["abbreviation"]}" in "{data["method"]}" stand for in Python?')}
    # )
    make_question = RunnableLambda(make_question_func)

    prompt2 = PromptTemplate.from_template("{question}\nExplain it in one short sentence.")
    chain = prompt1 | chat_llm | json_parser | make_question | prompt2 | chat_llm | str_parser
    response = chain.invoke(input={"method": "__repr__"})
    print(response)


if __name__ == "__main__":
    # invoke()
    # stream()
    # chat()

    # embed()

    # prompt_demo()
    # fewshot_prompt_demo()

    # chain_demo()
    parser_demo()
