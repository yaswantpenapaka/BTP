"""End-to-end with fake HTTP sessions. No network."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval.analyze import analyze
from eval.datasets import load_items
from eval.models import JevPlugin, LunaPlugin, StopModel
from eval.run import out_path, run_one

JB = [
    {"case_id": "c1", "domain": "d", "question_key": "team", "type": "choice", "instructions": "Which team?",
     "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}, {"id": "c", "description": "C"}],
     "levels": None, "gold": "b", "state": "text one", "item_id": "jevbench_c1_team"},
    {"case_id": "c1", "domain": "d", "question_key": "flag", "type": "noul", "instructions": "Is it?",
     "options": None, "levels": None, "gold": "unknown", "state": "text one", "item_id": "jevbench_c1_flag"},
    {"case_id": "c1", "domain": "d", "question_key": "lvl", "type": "score", "instructions": "How much?",
     "options": None, "levels": ["low", "mid", "high"], "gold": "2", "state": "text one", "item_id": "jevbench_c1_lvl"},
]
FD = [{"input": "blurb", "domain": "screen_tags", "item_id": "fd_1", "output": {"classifications": [
    {"task": "genres", "true_label": ["drama", "horror"], "labels": ["comedy", "drama", "horror"], "multi_label": True},
    {"task": "format", "true_label": ["series"], "labels": ["film", "series"], "multi_label": False}]}}]


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.headers = status, payload, {}
        self.text = json.dumps(payload)

    def json(self):
        return self._p


def pick(options):
    # deterministic fake: favour the 2nd option for multi-way, 'yes' for yes/no
    ids = list(options)
    best = "yes" if "yes" in ids else ids[min(1, len(ids) - 1)]
    rest = (1 - 0.7) / (len(ids) - 1)
    return best, {o: (0.7 if o == best else rest) for o in ids}


class JevSession:
    calls = 0

    def post(self, url, headers, json, timeout):
        JevSession.calls += 1
        ans = {}
        for name, q in json["questions"].items():
            best, probs = pick(q["criteria"])
            ans[name] = {"choice": best, "probabilities": dict(reversed(list(probs.items()))), "confidence": 0.7}
        return FakeResp(200, {"model": "jev-1.13.0", "answers": ans, "usage": {"input_tokens": 100}})


class LunaSession:
    def post(self, url, headers, json, timeout):
        ans = []
        for q in json["questions"]:
            ids = [c["value"] for c in q["choices"]]
            best, probs = pick(ids[::-1])  # disagrees with Jev on multi-way questions
            ans.append({"type": "choice", "name": q["name"], "choice": best, "confidence": 0.7,
                        "probabilities": [{"value": k, "probability": v} for k, v in probs.items()]})
        return FakeResp(200, {"model": "gpt-6-luna", "answers": ans, "usage": {"input_tokens": 100}})


class DenySession:
    def post(self, *a, **k):
        return FakeResp(401, {"error": "bad key"})


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write(self.tmp / "jb.jsonl", JB)
        write(self.tmp / "fd.jsonl", FD)
        self.jb = load_items("jevbench", self.tmp / "jb.jsonl")
        self.fd = load_items("fastdecisions", self.tmp / "fd.jsonl")
        self.quiet = lambda *_: None

    def plugins(self):
        logs = self.tmp / "logs"
        return (JevPlugin("k", logs, log=self.quiet, session=JevSession(), sleep=lambda s: None),
                LunaPlugin("k", logs, log=self.quiet, session=LunaSession(), sleep=lambda s: None))

    def test_adapters(self):
        self.assertIsNone(self.jb[1].questions[0].gold)  # 'unknown' -> no usable gold
        self.assertEqual(self.jb[2].questions[0].option_ids, ["0", "1", "2"])
        self.assertEqual(len(self.fd[0].questions), 4)  # 3 per-label yes/no + 1 single-label

    def test_run_resume_and_rows(self):
        jev, luna = self.plugins()
        out = self.tmp / "out"
        run_one(jev, "jevbench", self.jb, out, log=self.quiet)
        before = JevSession.calls
        s = run_one(jev, "jevbench", self.jb, out, log=self.quiet)
        self.assertEqual(s["sent"], 0)
        self.assertEqual(JevSession.calls, before)  # resume sent nothing
        rows = [json.loads(l) for l in out_path(out, "jev", "jevbench").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 3)
        r0 = rows[0]
        self.assertEqual(list(r0["probs"]), ["a", "b", "c"])  # dataset order, not API order
        self.assertAlmostEqual(sum(r0["probs"].values()), 1.0)
        self.assertEqual((r0["chosen"], r0["correct"]), ("b", 1))
        self.assertIsNone(rows[1]["correct"])  # unknown gold
        for k in ("item_id", "model_version", "confidence", "latency_ms", "prompt_hash", "timestamp", "error", "refusal"):
            self.assertIn(k, r0)
        self.assertNotIn("Bearer", json.dumps(rows))

    def test_fast_multilabel_exact_match(self):
        jev, _ = self.plugins()
        out = self.tmp / "out"
        run_one(jev, "fastdecisions", self.fd, out, log=self.quiet)
        r = json.loads(out_path(out, "jev", "fastdecisions").read_text(encoding="utf-8"))
        # fake says yes to every label -> predicted {comedy, drama, horror} != {drama, horror}
        self.assertEqual(r["fd_pred"]["genres"], ["comedy", "drama", "horror"])
        self.assertEqual(r["correct"], 0)

    def test_stop_on_401(self):
        jev = JevPlugin("k", self.tmp / "logs", log=self.quiet, session=DenySession())
        with self.assertRaises(StopModel):
            jev.decide(self.jb[0])
        s = run_one(jev, "jevbench", self.jb, self.tmp / "o2", log=self.quiet)
        self.assertTrue(s["stopped"])
        self.assertFalse(out_path(self.tmp / "o2", "jev", "jevbench").exists() and
                         out_path(self.tmp / "o2", "jev", "jevbench").read_text().strip())

    def test_analyze_matrices(self):
        jev, luna = self.plugins()
        out = self.tmp / "out"
        for p in (jev, luna):
            run_one(p, "jevbench", self.jb, out, log=self.quiet)
            run_one(p, "fastdecisions", self.fd, out, log=self.quiet)
        res = analyze(out, ["jev", "luna"])
        jl = res["jevbench"][("jev", "luna")]
        # choice item: jev picks b, luna picks b too? luna reverses -> picks 'b' (2nd of [c,b,a]) -> agree
        # score item: options 0,1,2 -> both pick '1' -> agree; noul true/false -> jev 'false', luna 'true'
        self.assertEqual(jl["n_units"], 3)
        self.assertAlmostEqual(jl["agreement"], 2 / 3)
        self.assertEqual(jl["n_gold_items"], 2)
        self.assertAlmostEqual(res["jevbench"][("luna", "jev")]["tvd"], jl["tvd"])
        with open(out / "analysis" / "jevbench" / "matrix_kl.csv", encoding="utf-8-sig") as fh:
            m = list(csv.reader(fh))
        self.assertEqual(m[0], ["Mi \\ Mj", "jev", "luna"])
        self.assertEqual(m[1][1], "—")
        self.assertTrue((out / "analysis" / "jevbench" / "merged.csv").exists())
        self.assertTrue((out / "analysis" / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
