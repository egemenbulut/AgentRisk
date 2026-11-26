"""Regression checks for the fixed split, artifact replay and even-size ensemble."""
import unittest

from compare import family
from protocol import REFERENCE, ROOT, dataset, read_jsonl, scores, vote


class FiveFoldTest(unittest.TestCase):
    def test_replay_qwen_and_double_the_voters(self):
        train, dev, folds = dataset()
        oof, voters, _ = family(REFERENCE, True, train, dev, folds)
        self.assertEqual(oof["severity_mae"], 0.25)
        rows = read_jsonl(ROOT / "data/test/qwen25_7b_5fold_ensemble_dev.jsonl")
        self.assertEqual(len(rows), len(dev))
        for row in rows:
            items = [v[row["id"]] for v in voters]
            expected = {k: v for k, v in row.items() if k != "id"}
            self.assertEqual(vote(items), expected)
            self.assertEqual(vote(items * 2), expected)

    def test_probability_median_and_distinct_ten_vote_thresholds(self):
        records = []
        for i in range(10):
            pred = {"risk_severity": 4,
                    "risk_domains": ["No Material Harm"] if i < 5 else ["Financial / Assets"],
                    "risk_drivers": ["Risky Requested Outcome"] if i < 4 else
                                    ["Execution / Action Hazard"] if i < 8 else ["No Material Risk"]}
            records.append({"example_id": "demo", "parsed": pred, "severity_probs": [0.5, 0, 0, 0.5, 0]})
        result = vote(records)
        self.assertEqual(result["risk_severity"], 1)
        self.assertEqual(result["risk_domains"], ["Financial / Assets"])
        self.assertEqual(result["risk_drivers"], ["Execution / Action Hazard", "Risky Requested Outcome"])
        records[-1]["parsed"]["risk_domains"] = ["No Material Harm"]
        self.assertEqual(vote(records)["risk_domains"], ["No Material Harm"])

    def test_malformed_answers_remain_in_denominator(self):
        pred = {"example_id": "demo", "parsed": None, "severity_probs": None}
        gold = {"demo": {"risk_severity": 3, "risk_drivers": [], "risk_domains": []}}
        metrics = scores({"demo": vote([pred] * 5)}, gold)
        self.assertEqual((metrics["n"], metrics["malformed"], metrics["severity_mae"]), (1, 1, 4))
        self.assertEqual(metrics["drivers_jaccard"], 0)


if __name__ == "__main__":
    unittest.main()
