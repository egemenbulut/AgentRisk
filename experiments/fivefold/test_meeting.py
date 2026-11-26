"""Check interim metrics against matching Qwen rows, including older runs."""
import tempfile
import unittest
from pathlib import Path

from meeting_report import make_report
from protocol import REFERENCE, dataset, read_jsonl, save_rows


class MeetingReportTest(unittest.TestCase):
    def test_partial_preview_matches_same_qwen_ids_and_full_file_wins(self):
        data = dataset()
        samples = read_jsonl(REFERENCE / "predictions/qwen25_7b_train_only_fold_0_val.jsonl")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "overnight.log"
            log.write_text("Fold 0 update 1/1 loss=1.1\nFold 0 update 1: validation loss 1.2\n")
            self.assertNotIn("Latest training log", make_report(root, log, data))
            log.write_text("Fold 0 update 10/171 loss=1.4\nFold 0 update 20: validation loss 0.7\n")
            save_rows(root / "fold_0/preview_8.partial.jsonl", samples[:4])
            report = make_report(root, log, data)
            self.assertIn("validation loss 0.7", report)
            self.assertIn("4/24 scored", report)
            lines = report.splitlines()
            self.assertEqual(next(s[s.index("severity"):] for s in lines if s.startswith("  Llama")),
                             next(s[s.index("severity"):] for s in lines if s.startswith("  Qwen")))
            save_rows(root / "fold_0/preview_8.jsonl", samples[:24])
            save_rows(root / "fold_0/val.jsonl", samples)
            report = make_report(root, log, data)
            self.assertIn("24/24 scored", report)
            self.assertIn("117/117 scored", report)
            self.assertNotIn("(4/24 scored)", report)
