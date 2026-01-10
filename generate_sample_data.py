"""
Generate sample data files for testing the full post-training pipeline.
Creates synthetic SFT, preference (DPO/RM), and prompt (PPO) data.

Usage:
    python generate_sample_data.py --output_dir data/
"""

import argparse
import json
import os
import random


SAMPLE_TOPICS = [
    "quantum computing", "climate change", "machine learning",
    "space exploration", "renewable energy", "blockchain",
    "genetics", "artificial intelligence", "cybersecurity",
    "robotics", "neuroscience", "sustainable farming",
]

SAMPLE_QUESTIONS = [
    "Explain {} in simple terms.",
    "What are the main benefits of {}?",
    "How does {} work?",
    "What are the latest developments in {}?",
    "Can you summarize the key concepts of {}?",
    "What are the challenges facing {}?",
    "How will {} impact the future?",
    "Compare and contrast different approaches to {}.",
]

SAMPLE_GOOD_ANSWERS = [
    "{topic} is a fascinating field. Here are the key points:\n\n"
    "1. It involves the study and application of advanced principles.\n"
    "2. Recent breakthroughs have made significant progress.\n"
    "3. The potential applications are wide-ranging.\n\n"
    "The field continues to evolve rapidly with new discoveries.",

    "Great question about {topic}! Let me break it down:\n\n"
    "At its core, {topic} deals with fundamental challenges in technology and science. "
    "Researchers have been working on this for decades, and recent advances in computing "
    "power and methodology have accelerated progress significantly.\n\n"
    "The most exciting aspect is how {topic} could transform everyday life.",
]

SAMPLE_BAD_ANSWERS = [
    "I don't really know much about that. Maybe try Google?",
    "{topic}? That's complicated. I can't explain it.",
    "Sure, {topic} is... a thing. It does stuff. Next question.",
]


def generate_sft_data(num_examples: int = 1000) -> list:
    """Generate SFT chat examples."""
    examples = []
    for _ in range(num_examples):
        topic = random.choice(SAMPLE_TOPICS)
        question = random.choice(SAMPLE_QUESTIONS).format(topic)
        answer = random.choice(SAMPLE_GOOD_ANSWERS).format(topic=topic)

        messages = [
            {"role": "system", "content": "You are a helpful, knowledgeable assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        # Sometimes add multi-turn
        if random.random() < 0.3:
            followup = f"Can you elaborate on the practical applications of {topic}?"
            followup_answer = (
                f"Certainly! {topic} has several practical applications:\n\n"
                f"In industry, it's used for optimization and efficiency improvements. "
                f"In research, it enables new discoveries and methodologies. "
                f"For consumers, it leads to better products and services.\n\n"
                f"The key is understanding how to bridge theory and practice."
            )
            messages.extend([
                {"role": "user", "content": followup},
                {"role": "assistant", "content": followup_answer},
            ])

        examples.append({"messages": messages})
    return examples


def generate_preference_data(num_examples: int = 500) -> list:
    """Generate preference pairs (chosen vs rejected) for DPO/RM."""
    examples = []
    for _ in range(num_examples):
        topic = random.choice(SAMPLE_TOPICS)
        question = random.choice(SAMPLE_QUESTIONS).format(topic)
        good_answer = random.choice(SAMPLE_GOOD_ANSWERS).format(topic=topic)
        bad_answer = random.choice(SAMPLE_BAD_ANSWERS).format(topic=topic)

        chosen = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": good_answer},
        ]
        rejected = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": bad_answer},
        ]

        examples.append({"chosen": chosen, "rejected": rejected})
    return examples


def generate_prompt_data(num_examples: int = 500) -> list:
    """Generate prompts for PPO training."""
    examples = []
    for _ in range(num_examples):
        topic = random.choice(SAMPLE_TOPICS)
        question = random.choice(SAMPLE_QUESTIONS).format(topic)
        prompt = [{"role": "user", "content": question}]
        examples.append({"prompt": prompt})
    return examples


def save_jsonl(data: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} examples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--num_sft", type=int, default=1000)
    parser.add_argument("--num_preference", type=int, default=500)
    parser.add_argument("--num_prompts", type=int, default=500)
    args = parser.parse_args()

    print("Generating sample training data...\n")

    sft_data = generate_sft_data(args.num_sft)
    save_jsonl(sft_data, os.path.join(args.output_dir, "sft_train.jsonl"))

    pref_data = generate_preference_data(args.num_preference)
    save_jsonl(pref_data, os.path.join(args.output_dir, "preference_train.jsonl"))

    prompt_data = generate_prompt_data(args.num_prompts)
    save_jsonl(prompt_data, os.path.join(args.output_dir, "prompts.jsonl"))

    print(f"\nAll data saved to {args.output_dir}/")
    print("Files:")
    print(f"  sft_train.jsonl         - {args.num_sft} SFT examples")
    print(f"  preference_train.jsonl  - {args.num_preference} preference pairs")
    print(f"  prompts.jsonl           - {args.num_prompts} prompts (for PPO)")


if __name__ == "__main__":
    main()
