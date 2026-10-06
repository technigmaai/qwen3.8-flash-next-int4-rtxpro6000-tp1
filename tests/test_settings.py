"""Host-only tests; no GPU, network, model loading or external service changes."""
import tempfile
import unittest
from pathlib import Path
from settings import ROOT, load

class SettingsTest(unittest.TestCase):
    def fixture(self, changes=None):
        content = (ROOT/'.env.example').read_text().replace('/home/your-user/', '/home/test-user/')
        for before,after in (changes or {}).items():
            content = content.replace(before,after)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)/'.env'
        path.write_text(content)
        return path

    def test_profile(self):
        c = load(self.fixture())
        self.assertEqual(c['MAX_MODEL_LEN'],'524288')
        self.assertEqual(c['MAX_NUM_SEQS'],'8')
        self.assertEqual(c['MTP_TOKENS'],'3')
        self.assertEqual(c['INDEXER_KV_DTYPE'], 'bf16')

    def test_fp8_indexer_setting(self):
        c = load(self.fixture({'INDEXER_KV_DTYPE=bf16': 'INDEXER_KV_DTYPE=fp8'}))
        self.assertEqual(c['INDEXER_KV_DTYPE'], 'fp8')

    def test_unsupported_indexer_setting(self):
        with self.assertRaises(ValueError):
            load(self.fixture({'INDEXER_KV_DTYPE=bf16': 'INDEXER_KV_DTYPE=mxfp4'}))

    def test_shell_expansion_rejected(self):
        with self.assertRaises(ValueError):
            load(self.fixture({'/home/test-user/':'$HOME/'}))

    def test_model_runtime_overlap_rejected(self):
        with self.assertRaises(ValueError):
            load(self.fixture({'/home/test-user/.cache/qwen38-rtxpro6000':
                '/home/test-user/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/runtime'}))

    def test_context_limit(self):
        with self.assertRaises(ValueError):
            load(self.fixture({'MAX_MODEL_LEN=524288':'MAX_MODEL_LEN=1048576'}))

    def test_missing_alias(self):
        with self.assertRaises(ValueError):
            load(self.fixture({' rtx\'':'\''}))

if __name__ == '__main__':
    unittest.main()
