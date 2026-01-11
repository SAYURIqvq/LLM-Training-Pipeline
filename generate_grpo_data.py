"""
Generate sample GRPO training data with verifiable tasks (math, logic).
These tasks have clear correct answers, making rule-based rewards effective.

Usage:
    python generate_grpo_data.py --output data/grpo_prompts.jsonl --num 2000
"""

import argparse
import json
import os
import random


def generate_arithmetic(difficulty: str = "easy"):
    """Generate arithmetic problems with answers."""
    if difficulty == "easy":
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
    elif difficulty == "medium":
        a, b = random.randint(10, 1000), random.randint(10, 1000)
        op = random.choice(["+", "-", "*", "/"])
        if op == "/":
            a = a * b  # Ensure clean division
    else:
        a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(1, 50)
        expr_type = random.choice(["chain", "mixed"])
        if expr_type == "chain":
            prompt = f"Calculate: {a} × {b} + {c}"
            answer = str(a * b + c)
            return prompt, answer
        else:
            prompt = f"Calculate: ({a} + {b}) × {c}"
            answer = str((a + b) * c)
            return prompt, answer

    result = eval(f"{a} {op} {b}")
    if op == "/":
        result = int(result)

    op_word = {"+" : "+", "-": "-", "*": "×", "/": "÷"}[op]
    prompt = f"What is {a} {op_word} {b}?"
    return prompt, str(result)


def generate_algebra():
    """Generate simple algebra problems."""
    templates = [
        lambda: (
            f"Solve for x: {(a := random.randint(2, 10))}x + {(b := random.randint(1, 20))} = {(c := a * random.randint(1, 10) + b)}",
            str((c - b) // a)
        ),
        lambda: (
            f"Solve for x: {(a := random.randint(2, 8))}x - {(b := random.randint(1, 15))} = {(c := a * random.randint(1, 10) - b)}",
            str((c + b) // a)
        ),
        lambda: (
            f"If x + {(a := random.randint(1, 20))} = {(b := random.randint(a + 1, 50))}, what is x?",
            str(b - a)
        ),
        lambda: (
            f"If {(a := random.randint(2, 5))}x = {(b := a * random.randint(2, 20))}, what is x?",
            str(b // a)
        ),
    ]
    return random.choice(templates)()


def generate_word_problem():
    """Generate word problems."""
    templates = [
        lambda: (
            f"A store sells apples for ${(p := random.randint(1, 5))} each. "
            f"If you buy {(n := random.randint(3, 20))} apples, how much do you pay in total?",
            str(p * n)
        ),
        lambda: (
            f"A train travels at {(s := random.randint(40, 120))} km/h. "
            f"How far does it travel in {(t := random.randint(2, 8))} hours?",
            str(s * t)
        ),
        lambda: (
            f"You have {(total := random.randint(50, 200))} books and want to put them equally "
            f"on {(shelves := random.choice([5, 10, 20, 25]))} shelves. How many books per shelf?",
            str(total // shelves)
        ),
        lambda: (
            f"A rectangle has a length of {(l := random.randint(5, 30))} cm and a width of "
            f"{(w := random.randint(3, 20))} cm. What is its area in square centimeters?",
            str(l * w)
        ),
        lambda: (
            f"There are {(g := random.randint(5, 30))} girls and {(b := random.randint(5, 30))} boys "
            f"in a class. How many students are there in total?",
            str(g + b)
        ),
    ]
    return random.choice(templates)()


def generate_sequence():
    """Generate number sequence problems."""
    # Arithmetic sequence
    start = random.randint(1, 20)
    diff = random.randint(1, 10)
    seq = [start + i * diff for i in range(5)]
    answer = str(seq[-1] + diff)
    prompt = f"What is the next number in the sequence: {', '.join(map(str, seq))}?"
    return prompt, answer


def generate_comparison():
    """Generate comparison/logic problems."""
    templates = [
        lambda: (
            f"Which is larger: {(a := random.randint(100, 10000))} or {(b := random.randint(100, 10000))}?",
            str(max(a, b))
        ),
        lambda: (
            f"What is the remainder when {(a := random.randint(50, 500))} is divided by {(b := random.randint(3, 20))}?",
            str(a % b)
        ),
    ]
    return random.choice(templates)()


def generate_reasoning():
    """Generate reasoning prompts (no single correct answer, use format reward)."""
    topics = [
        "Explain step by step how photosynthesis works.",
        "Describe the water cycle in detail.",
        "Explain how a computer processor executes instructions.",
        "Describe the process of natural selection.",
        "Explain how vaccines help the immune system.",
        "Describe how a search engine ranks web pages.",
        "Explain the greenhouse effect step by step.",
        "Describe how sound travels from a speaker to your ear.",
    ]
    prompt = random.choice(topics)
    return prompt, None  # No reference answer; relies on format reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/grpo_prompts.jsonl")
    parser.add_argument("--num", type=int, default=2000)
    parser.add_argument("--include_reasoning", action="store_true", default=False,
                        help="Include open-ended reasoning prompts (no reference answer)")
    args = parser.parse_args()

    generators = [
        (generate_arithmetic, 0.25, {"difficulty": "easy"}),
        (generate_arithmetic, 0.10, {"difficulty": "medium"}),
        (generate_arithmetic, 0.10, {"difficulty": "hard"}),
        (generate_algebra, 0.20, {}),
        (generate_word_problem, 0.20, {}),
        (generate_sequence, 0.10, {}),
        (generate_comparison, 0.05, {}),
    ]

    if args.include_reasoning:
        generators.append((generate_reasoning, 0.10, {}))
        # Re-normalize weights
        total_w = sum(w for _, w, _ in generators)
        generators = [(fn, w / total_w, kw) for fn, w, kw in generators]

    data = []
    for _ in range(args.num):
        r = random.random()
        cumulative = 0
        for fn, weight, kwargs in generators:
            cumulative += weight
            if r < cumulative:
                prompt, answer = fn(**kwargs)
                item = {"prompt": prompt}
                if answer is not None:
                    item["answer"] = answer
                data.append(item)
                break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    n_with_answer = sum(1 for d in data if "answer" in d)
    print(f"Generated {len(data)} GRPO prompts ({n_with_answer} with answers)")
    print(f"Saved to {args.output}")

    # Show samples
    print("\nSamples:")
    for item in random.sample(data, min(5, len(data))):
        print(f"  Q: {item['prompt']}")
        if "answer" in item:
            print(f"  A: {item['answer']}")
        print()


if __name__ == "__main__":
    main()
