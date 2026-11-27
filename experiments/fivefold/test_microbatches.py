"""CPU- or GPU-runnable checks for the optional MLX microbatch trainer."""
import importlib.util
from pathlib import Path
import json
import tempfile
import unittest

try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipUnless(mx is not None and importlib.util.find_spec("mlx_lm"), "MLX and mlx-lm required")
class MicrobatchTest(unittest.TestCase):
    def test_same_loss_and_gradients_with_right_padding(self):
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.llama import Model, ModelArgs
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from llama import LORA, group_gradients, microbatches, pair_response_loss, response_loss

        # Partial update, dissimilar prompt boundaries, nonzero LoRA weights, one long singleton.
        examples = [([9, 8, 7, 6, 5, 4, 3, 2, 1], 3), ([9, 12, 5, 1, 4, 3, 2, 18], 6),
                    ([2, 3, 4] * 8, 18)]
        self.assertEqual([len(batch) for batch in microbatches(examples, 2, 24)], [2, 1])
        for tied in (True, False):
            with self.subTest(tied=tied):
                mx.random.seed(42)
                model = Model(ModelArgs(model_type="llama", hidden_size=64, intermediate_size=128,
                              num_hidden_layers=2, num_attention_heads=2, rms_norm_eps=1e-5,
                              vocab_size=128, tie_word_embeddings=tied))
                nn.quantize(model, group_size=64, bits=4)
                model.freeze()
                linear_to_lora_layers(model, len(model.layers), LORA)
                for key, layer in model.named_modules():
                    if hasattr(layer, "lora_b"):
                        layer.lora_b = mx.random.normal(layer.lora_b.shape) * .01
                single = nn.value_and_grad(model, response_loss)
                pair = nn.value_and_grad(model, pair_response_loss)
                a, ga = group_gradients(examples, 1, 24, single, pair, model)
                b, gb = group_gradients(examples, 2, 24, single, pair, model)
                self.assertAlmostEqual(a, b, places=4)
                leaves = dict(tree_flatten(gb))
                for key, orig in tree_flatten(ga):
                    self.assertTrue(mx.allclose(orig, leaves[key], rtol=2e-4, atol=2e-5).item(), key)

    def test_benchmark_and_production_fold_without_weights_download(self):
        import mlx.nn as nn
        from mlx_lm.models.llama import Model, ModelArgs
        from llama import train_adapter

        mx.random.seed(100)
        model = Model(ModelArgs(model_type="llama", hidden_size=64, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=2, rms_norm_eps=1e-5, vocab_size=128))
        nn.quantize(model, group_size=64, bits=4)
        encoded = {str(i): ([1, 2, 3] * (4 if i < 23 else 12), 7) for i in range(25)}
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "fold_2"
            folder.mkdir()
            (folder.parent / "benchmark-batches").touch()
            train_adapter(model, encoded, list(encoded)[:24], ["24"], 2, folder, smoke=False)
            metadata = json.loads((folder / "microbatch.json").read_text())
            self.assertEqual(metadata["timed_examples_per_mode"], 16)
            self.assertTrue(metadata["serial_seconds"] > 0 and metadata["paired_seconds"] > 0)
            self.assertTrue((folder / "adapters.safetensors").is_file())
            self.assertEqual(json.loads((folder / "training.json").read_text())["best"]["update"], 9)
