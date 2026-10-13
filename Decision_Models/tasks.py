"""Frozen task definitions. Both APIs are built from this file so the words cannot drift.

Do not edit a template after a run has started. A changed instruction or option
string is a different experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


JEV_MODEL = "jev-1.13.0"
OPENAI_MODEL = "gpt-6-luna"

# USD per input token. Output tokens are free on both APIs.
PRICE_PER_TOKEN = {
    JEV_MODEL: 0.042 / 1_000_000,
    OPENAI_MODEL: 0.10 / 1_000_000,
}

MODEL_ORDER = (JEV_MODEL, OPENAI_MODEL)


@dataclass(frozen=True)
class Task:
    key: str
    hf_path: str
    hf_config: str | None
    split: str
    expected_n: int
    text_column: str
    state_field: str
    instruction: str
    # Names on the dataset's own ClassLabel, index 0 first.
    feature_labels: tuple[str, ...]
    # Option values sent to both APIs, same index order as feature_labels.
    prompt_labels: tuple[str, ...]
    descriptions: dict[str, str]

    def state(self, text: str) -> dict[str, str]:
        return {self.state_field: text}

    def criteria(self) -> dict[str, str]:
        return {label: self.descriptions[label] for label in self.prompt_labels}


SENTIMENT_DESCRIPTIONS = {
    "negative": "The text expresses a negative opinion.",
    "positive": "The text expresses a positive opinion.",
}

# TODO: Before the full run, check these six descriptions against the frozen
# templates in arXiv:2609.37647 (Deußer, Sparrenberg & Sifa, "Evaluating and
# Benchmarking the System One Model Jev"). The paper's released code
# (jev_benchmarking/tasks/classification.py, class Emotion) passes no
# description for these labels (criteria value None). The instruction below
# matches that code. Do not change this text between runs once it is confirmed.
EMOTION_DESCRIPTIONS = {
    "sadness": "The author most strongly expresses sadness.",
    "joy": "The author most strongly expresses joy.",
    "love": "The author most strongly expresses love.",
    "anger": "The author most strongly expresses anger.",
    "fear": "The author most strongly expresses fear.",
    "surprise": "The author most strongly expresses surprise.",
}

# TODO: Before the full run, check these descriptions and the instruction
# against arXiv:2609.37647. The paper's released code uses these stock-price
# sentences, and its instruction is "From an investor's point of view, what is
# the sentiment of the financial news `sentence`?" with backticks around
# sentence. This project uses the instruction string below, without backticks.
# Do not change either string between runs.
FINANCIAL_DESCRIPTIONS = {
    "negative": "The news is likely to have a negative effect on the company's stock price.",
    "neutral": "The news is unlikely to affect the company's stock price.",
    "positive": "The news is likely to have a positive effect on the company's stock price.",
}


TASKS: dict[str, Task] = {
    "sst2": Task(
        key="sst2",
        hf_path="stanfordnlp/sst2",
        hf_config=None,
        split="validation",
        expected_n=872,
        text_column="sentence",
        state_field="review",
        instruction="What is the overall sentiment of `review` toward the movie?",
        feature_labels=("negative", "positive"),
        prompt_labels=("negative", "positive"),
        descriptions=SENTIMENT_DESCRIPTIONS,
    ),
    "rotten_tomatoes": Task(
        key="rotten_tomatoes",
        hf_path="cornell-movie-review-data/rotten_tomatoes",
        hf_config=None,
        split="test",
        expected_n=1066,
        text_column="text",
        state_field="review",
        instruction="What is the overall sentiment of `review` toward the movie?",
        # The Hub ClassLabel is neg/pos. Index 0 is neg, index 1 is pos.
        # The prompt still says negative/positive. Load time checks the Hub names.
        feature_labels=("neg", "pos"),
        prompt_labels=("negative", "positive"),
        descriptions=SENTIMENT_DESCRIPTIONS,
    ),
    "emotion": Task(
        key="emotion",
        hf_path="dair-ai/emotion",
        hf_config="split",
        split="test",
        expected_n=2000,
        text_column="text",
        state_field="text",
        instruction="Which emotion does the author of `text` express most strongly?",
        feature_labels=("sadness", "joy", "love", "anger", "fear", "surprise"),
        prompt_labels=("sadness", "joy", "love", "anger", "fear", "surprise"),
        descriptions=EMOTION_DESCRIPTIONS,
    ),
    "financial_phrasebank": Task(
        key="financial_phrasebank",
        hf_path="atrost/financial_phrasebank",
        hf_config=None,
        split="test",
        expected_n=970,
        text_column="sentence",
        state_field="sentence",
        instruction="From an investor's point of view, what is the sentiment of the financial news sentence?",
        feature_labels=("negative", "neutral", "positive"),
        prompt_labels=("negative", "neutral", "positive"),
        descriptions=FINANCIAL_DESCRIPTIONS,
    ),
}


def get_task(key: str) -> Task:
    if key not in TASKS:
        known = ", ".join(TASKS)
        raise SystemExit(f"Unknown dataset {key!r}. Known: {known}.")
    return TASKS[key]
