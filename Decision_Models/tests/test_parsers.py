"""Parser tests use saved example responses. They do not call either API."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clients.common import ResponseSchemaError
from clients.jev import JEV_URL, build_body as build_jev
from clients.jev import parse_response as parse_jev
from clients.openai_decisions import OPENAI_URL, build_body as build_openai
from clients.openai_decisions import parse_response as parse_openai
from tasks import JEV_MODEL, OPENAI_MODEL, get_task

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ParserTests(unittest.TestCase):
    def test_endpoints_and_pinned_models(self):
        self.assertEqual(JEV_URL, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(OPENAI_URL, "https://api.openai.com/v1/decisions")
        self.assertEqual(build_jev(get_task("sst2"), "hi")["model"], JEV_MODEL)
        self.assertEqual(build_openai(get_task("sst2"), "hi")["model"], OPENAI_MODEL)

    def test_both_apis_get_the_same_words(self):
        text = "The café scene was dull."
        for key in ("sst2", "rotten_tomatoes", "emotion", "financial_phrasebank"):
            task = get_task(key)
            jev = build_jev(task, text)
            openai = build_openai(task, text)
            question = jev["questions"]["answer"]
            other = openai["questions"][0]
            self.assertEqual(question["instructions"], other["instructions"])
            self.assertEqual(question["instructions"], task.instruction)
            self.assertEqual(list(question["criteria"]), [choice["value"] for choice in other["choices"]])
            for choice in other["choices"]:
                self.assertEqual(choice["description"], question["criteria"][choice["value"]])
            self.assertEqual(json.loads(openai["input"]), jev["state"])
            self.assertEqual(jev["state"], {task.state_field: text})

    def test_frozen_template_text(self):
        sentiment = "What is the overall sentiment of `review` toward the movie?"
        self.assertEqual(get_task("sst2").instruction, sentiment)
        self.assertEqual(get_task("rotten_tomatoes").instruction, sentiment)
        self.assertEqual(
            get_task("sst2").descriptions["negative"],
            "The text expresses a negative opinion.",
        )
        self.assertEqual(
            get_task("sst2").descriptions["positive"],
            "The text expresses a positive opinion.",
        )
        self.assertEqual(
            get_task("emotion").instruction,
            "Which emotion does the author of `text` express most strongly?",
        )
        self.assertEqual(
            get_task("emotion").prompt_labels,
            ("sadness", "joy", "love", "anger", "fear", "surprise"),
        )
        financial = get_task("financial_phrasebank")
        self.assertEqual(
            financial.instruction,
            "From an investor's point of view, what is the sentiment of the financial news sentence?",
        )
        self.assertNotIn("`", financial.instruction)
        self.assertEqual(
            financial.descriptions["neutral"],
            "The news is unlikely to affect the company's stock price.",
        )

    def test_jev_choice_fixture(self):
        parsed = parse_jev(load_fixture("jev_choice.json"), Path("unused"))
        self.assertEqual(parsed.pred, "negative")
        self.assertEqual(parsed.probabilities["negative"], 0.91)
        self.assertEqual(parsed.probabilities["positive"], 0.09)
        self.assertEqual(parsed.confidence, 0.84)
        self.assertEqual(parsed.input_tokens, 120)
        self.assertEqual(parsed.error, "")

    def test_openai_choice_fixture(self):
        parsed = parse_openai(load_fixture("openai_choice.json"), Path("unused"))
        self.assertEqual(parsed.pred, "negative")
        self.assertEqual(parsed.probabilities["negative"], 0.93)
        self.assertEqual(parsed.probabilities["positive"], 0.07)
        self.assertEqual(parsed.confidence, 0.9)
        self.assertEqual(parsed.input_tokens, 80)
        self.assertEqual(parsed.error, "")

    def test_openai_refusal_is_a_row_error(self):
        parsed = parse_openai(load_fixture("openai_refusal.json"), Path("unused"))
        self.assertEqual(parsed.pred, "")
        self.assertEqual(parsed.error, "refusal")
        self.assertIsNone(parsed.probabilities)
        self.assertEqual(parsed.input_tokens, 40)

    def test_jev_missing_choice_keeps_probabilities(self):
        parsed = parse_jev(load_fixture("jev_missing_choice.json"), Path("unused"))
        self.assertEqual(parsed.pred, "")
        self.assertEqual(parsed.error, "missing choice")
        self.assertEqual(parsed.probabilities["negative"], 0.7)
        self.assertEqual(parsed.confidence, 0.4)

    def test_jev_missing_probabilities_saves_redacted_body(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            with self.assertRaises(ResponseSchemaError) as caught:
                parse_jev(load_fixture("jev_missing_probabilities.json"), logs)
            saved = list(logs.glob("schema_jev_*.json"))
            self.assertEqual(len(saved), 1)
            text = saved[0].read_text(encoding="utf-8")
            self.assertIn("Bearer [redacted]", text)
            self.assertNotIn("SHOULD-NOT-LEAK", text)
            self.assertIn(str(saved[0]), str(caught.exception))
            self.assertEqual(caught.exception.input_tokens, 10)

    def test_openai_missing_answers_saves_body(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            with self.assertRaises(ResponseSchemaError):
                parse_openai(load_fixture("openai_missing_answers.json"), logs)
            self.assertEqual(len(list(logs.glob("schema_openai_*.json"))), 1)

    def test_openai_probability_map_is_a_schema_error(self):
        payload = {
            "answers": [
                {
                    "type": "choice",
                    "name": "answer",
                    "choice": "negative",
                    "probabilities": {"negative": 0.9, "positive": 0.1},
                    "confidence": 0.8,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            with self.assertRaises(ResponseSchemaError):
                parse_openai(payload, logs)
            self.assertTrue(list(logs.glob("schema_openai_*.json")))


if __name__ == "__main__":
    unittest.main()
