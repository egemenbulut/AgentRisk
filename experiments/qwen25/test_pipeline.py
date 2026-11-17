"""Run with python -m unittest discover -s experiments/qwen25 -v."""
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from build_dataset import generate
from data_utils import (DATA, ROOT, TARGET_KEYS, encode_training, fingerprint,
                        pad_features, read_jsonl, sha256, to_chat, validate_target)
from evaluate import score
from predict import parse_prediction


class ToyChatTokenizer:
    """Tests masking logic, not Qwen's actual tokenizer (checked by preflight)."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = ''.join('<'+m['role']+'>'+m['content']+'</end>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(text.encode('utf-8'))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = read_jsonl(ROOT/'data/raw/train.jsonl')
        cls.dev = read_jsonl(ROOT/'data/raw/dev.jsonl')
        cls.synthetic = read_jsonl(DATA/'synthetic_train.jsonl')
        cls.extended = read_jsonl(DATA/'train_extended.jsonl')

    def test_originals_unchanged_and_counts(self):
        self.assertEqual(self.extended, self.original + self.synthetic)
        self.assertEqual(len(self.extended), 1336)
        self.assertEqual(len(self.dev), 143)

    def test_reproducible_generation(self):
        rows, provenance = generate()
        self.assertEqual(rows, self.synthetic)
        self.assertEqual(provenance, read_jsonl(DATA/'provenance.jsonl'))

    def test_hashes(self):
        manifest = json.loads((DATA/'manifest.json').read_text())
        for name, digest in manifest['source_sha256'].items():
            self.assertEqual(sha256(ROOT/name), digest)
        for name, digest in manifest['output_sha256'].items():
            self.assertEqual(sha256(DATA/name), digest)

    def test_no_exact_overlap_or_duplicate(self):
        self.assertEqual(len({fingerprint(r) for r in self.extended}), len(self.extended))
        self.assertFalse({fingerprint(r) for r in self.extended} & {fingerprint(r) for r in self.dev})
        self.assertFalse({r['example_id'] for r in self.extended} & {r['example_id'] for r in self.dev})

    def test_synthetic_label_schema_and_final_action(self):
        for r in self.synthetic:
            validate_target({k:r[k] for k in TARGET_KEYS})
            self.assertTrue(r['formatted_interaction'].endswith('[ACTION]: '+r['proposed_action']))
            self.assertEqual(r['risk_domains'] == ['No Material Harm'], r['risk_drivers'] == ['No Material Risk'])
            self.assertEqual(r['risk_severity'] == 1, r['risk_domains'] == ['No Material Harm'])

    def test_family_counts_and_no_metadata_in_prompt(self):
        provenance = read_jsonl(DATA/'provenance.jsonl')
        self.assertEqual(set(Counter(p['family_id'] for p in provenance).values()), {12})
        chats = read_jsonl(DATA/'train_extended_sft.jsonl')
        raw = {r['example_id']:r for r in self.extended}
        self.assertEqual(len(chats), len(raw))
        for chat in chats:
            self.assertEqual(chat, to_chat(raw[chat['id']]))
            self.assertEqual(set(json.loads(chat['messages'][1]['content'])),
                             {'formatted_interaction', 'proposed_action'})
        for chat in read_jsonl(DATA/'dev_inference.jsonl'):
            self.assertEqual([m['role'] for m in chat['messages']], ['system', 'user'])

    def test_loss_mask_and_padding(self):
        tokenizer = ToyChatTokenizer()
        row = to_chat(self.synthetic[0])
        encoded = encode_training(row, tokenizer, 20000)
        prefix = tokenizer.apply_chat_template(row['messages'][:-1], add_generation_prompt=True)
        self.assertTrue(all(v == -100 for v in encoded['labels'][:len(prefix)]))
        self.assertEqual(encoded['labels'][len(prefix):], encoded['input_ids'][len(prefix):])
        padded = pad_features([encoded, {k:v[:-3] for k,v in encoded.items()}], pad_token_id=42)
        self.assertEqual(padded['labels'][1][-3:], [-100]*3)
        self.assertEqual(padded['attention_mask'][1][-3:], [0]*3)
        self.assertEqual(padded['input_ids'][1][-3:], [42]*3)

    def test_overlong_evidence_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            encode_training(to_chat(self.synthetic[0]), ToyChatTokenizer(), 10)

    def test_invalid_target_is_not_silently_coerced(self):
        target = {k:self.synthetic[0][k] for k in TARGET_KEYS}
        for field, value in [('risk_severity', True), ('risk_severity', '1'),
                             ('risk_domains', ['Safety / Health']),
                             ('risk_drivers', ['No Material Risk', 'Execution / Action Hazard'])]:
            with self.assertRaises(ValueError):
                validate_target({**target, field: value})
        with self.assertRaises(ValueError):
            parse_prediction('```json\n'+json.dumps(target)+'\n```')

    def test_missing_invalid_and_valid_metrics(self):
        gold = self.synthetic[:3]
        perfect = [{'id':r['example_id'], **{k:r[k] for k in TARGET_KEYS}} for r in gold]
        result = score(gold, perfect)
        self.assertEqual(result['severity_accuracy_all'], 1.)
        self.assertEqual(result['severity_mae_penalized_all'], 0.)
        self.assertEqual(result['domains_jaccard_all'], 1.)
        result = score(gold, [perfect[0], {'id':gold[1]['example_id'], 'valid_output':False}])
        self.assertEqual((result['valid_outputs'], result['invalid_outputs'], result['missing_outputs']), (1,1,1))
        self.assertAlmostEqual(result['domains_jaccard_all'], 1/3)
        self.assertAlmostEqual(result['severity_mae_penalized_all'], 8/3)

    def test_metric_duplicate_and_unknown_ids_fail(self):
        row = self.synthetic[0]
        prediction = {'id':row['example_id'], **{k:row[k] for k in TARGET_KEYS}}
        with self.assertRaises(ValueError):
            score([row], [prediction, prediction])
        with self.assertRaises(ValueError):
            score([row], [{**prediction, 'id':'unknown'}])


if __name__ == '__main__':
    unittest.main()
