from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

examples_data = {
    "Match": [
        (
            (
                "Amazon shares surged after the company reported its strongest "
                "quarterly revenue growth in more than four years."
            ),
            (
                "Amazon stock jumped following strong quarterly results that "
                "increased investor confidence in its AI investments."
            ),
        ),
        (
            (
                "Microsoft shares recorded their strongest one-day gain since 2008 "
                "after the company reported better-than-expected quarterly profit."
            ),
            (
                "Microsoft stock soared as strong Azure growth suggested that the "
                "company's AI investments were generating returns."
            ),
        ),
    ],
    "Not Match": [
        (
            "Apple shares fell after the company warned that component supply constraints could limit its growth.",
            (
                "Saudi Arabia's Public Investment Fund received EU approval for "
                "its acquisition of video game developer Electronic Arts."
            ),
        ),
        (
            "Oil prices increased as tensions in the Middle East raised concerns about global crude supplies.",
            "Microsoft shares rose following strong growth in its Azure cloud business.",
        ),
    ],
}

questions = [
    (
        "Wall Street closed higher as Amazon's quarterly results improved investor confidence in AI-related companies.",
        (
            "Strong earnings from Amazon helped lift the U.S. stock market and "
            "reduced concerns about excessive AI spending."
        ),
    ),
    (
        "Apple stock declined after the company warned that supply shortages could affect future revenue growth.",
        "European regulators approved a 55-billion-dollar deal involving Electronic Arts and a Saudi investor group.",
    ),
    (
        (
            "European stocks recorded a monthly gain because strong company earnings "
            "helped offset geopolitical and AI-related concerns."
        ),
        "The STOXX 600 ended July with gains as corporate earnings supported the market despite Middle East tensions.",
    ),
]
client = OpenAI(base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": (
            "You are a semantic text matching assistant. "
            "You will receive two sentences enclosed in square brackets. "
            "Determine whether the two sentences describe the same core event "
            "or express substantially the same meaning. "
            "Reply with exactly 'Match' or 'Not Match'. "
        ),
    }
]
for label, sentence_pairs in examples_data.items():
    for sentence_1, sentence_2 in sentence_pairs:
        messages.append({"role": "user", "content": f"Sentence 1: [{sentence_1}]\nSentence 2: [{sentence_2}]"})
        messages.append({"role": "assistant", "content": label})
for sentence_1, sentence_2 in questions:
    completion = client.chat.completions.create(
        model="glm-5",
        messages=[*messages, {"role": "user", "content": (f"Sentence 1: [{sentence_1}]\nSentence 2: [{sentence_2}]")}],
    )
    print(completion.choices[0].message.content, flush=True)
