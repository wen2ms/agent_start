from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

examples_data = {
    "Algebra": (
        "A quadratic equation has the general form ax^2 + bx + c = 0, "
        "where a is not equal to zero. Its solutions can be obtained using "
        "the quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / (2a)."
    ),
    "Calculus": (
        "The derivative of a function describes its instantaneous rate of change. "
        "For example, if f(x) = x^2, then its derivative is f'(x) = 2x."
    ),
    "Geometry": (
        "The Pythagorean theorem states that, in a right triangle, the square "
        "of the hypotenuse is equal to the sum of the squares of the other "
        "two sides: a^2 + b^2 = c^2."
    ),
    "Probability and Statistics": (
        "The probability of an event is a numerical measure between 0 and 1. "
        "For a fair coin, the probability of obtaining heads in a single toss "
        "is equal to 1/2."
    ),
}

examples_types = [
    "Algebra",
    "Calculus",
    "Geometry",
    "Probability and Statistics",
]
questions = [
    ("Solve the equation 2x + 6 = 14. Subtracting 6 from both sides gives 2x = 8, so the solution is x = 4."),
    ("Find the derivative of f(x) = 3x^3 - 2x + 5. By applying the power rule, we obtain f'(x) = 9x^2 - 2."),
    (
        "A right triangle has two perpendicular sides of lengths 3 and 4. "
        "According to the Pythagorean theorem, the hypotenuse has length 5."
    ),
    (
        "A standard six-sided die is rolled once. The probability of rolling "
        "an even number is 3/6, which is equal to 1/2."
    ),
    "Tom enjoys playing basketball with his friends.",
]
client = OpenAI(base_url="https://ws-yi9oakgdflk8zstn.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": (
            "You are a mathematics expert. "
            f"Classify each input text into exactly one of the following categories: {examples_types}, "
            "Return 'Unknown' if the text does not belong to any of the categories above."
        ),
    }
]
for key, value in examples_data.items():
    messages.append({"role": "user", "content": value})
    messages.append({"role": "assistant", "content": key})

for question in questions:
    completion = client.chat.completions.create(
        model="qwen3.7-flash", messages=[*messages, {"role": "user", "content": f"{question}"}]
    )
    print(completion.choices[0].message.content, flush=True)
