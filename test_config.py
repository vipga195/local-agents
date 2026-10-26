import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import ConfigError, load_config, get_slot, get_warn_tokens, make_brain


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text):
        p = self.dir / 'config.toml'
        p.write_text(text)
        return p

    def test_missing_file_defaults(self):
        cfg = load_config(self.dir / 'nope.toml')
        self.assertEqual(cfg['slots']['main'], {'provider': 'ollama', 'model': None, 'host': None})

    def test_env_path(self):
        p = self.write('[slots.main]\nmodel = "m1"\n')
        with mock.patch.dict(os.environ, {'AGENT_CONFIG': str(p)}):
            self.assertEqual(load_config()['slots']['main']['model'], 'm1')

    def test_per_key_override(self):
        p = self.write('[slots.main]\nmodel = "llama3"\n')
        main = load_config(p)['slots']['main']
        self.assertEqual(main['model'], 'llama3')
        self.assertEqual(main['provider'], 'ollama')
        self.assertIsNone(main['host'])

    def test_new_slot_keeps_default_main(self):
        p = self.write('[slots.aux]\nprovider = "ollama"\nmodel = "small"\n')
        cfg = load_config(p)
        self.assertIn('main', cfg['slots'])
        self.assertEqual(cfg['slots']['aux']['model'], 'small')

    def test_defaults_not_mutated(self):
        p = self.write('[slots.main]\nmodel = "x"\n')
        load_config(p)
        self.assertIsNone(load_config(self.dir / 'nope.toml')['slots']['main']['model'])

    def test_bad_toml(self):
        p = self.write('[slots.main\nmodel = ')
        with self.assertRaises(ConfigError):
            load_config(p)

    def test_unreadable_file(self):
        with self.assertRaises(ConfigError):
            load_config(self.dir)  # a directory exists but cannot be read as a file

    def test_unknown_slot(self):
        with self.assertRaises(ConfigError):
            get_slot(load_config(self.dir / 'nope.toml'), 'zzz')

    def test_unsupported_provider(self):
        with self.assertRaises(ConfigError) as cm:
            make_brain({'provider': 'foo'})
        self.assertIn("provider 'foo' not supported yet", str(cm.exception))

    def test_make_brain_passes_host_model(self):
        brain = make_brain({'provider': 'ollama', 'host': 'box:1234', 'model': 'm2'})
        self.assertEqual(brain.host, 'http://box:1234')
        self.assertEqual(brain.model, 'm2')

    def test_openai_and_anthropic_happy_path(self):
        env = {'OPENAI_API_KEY': 'k1', 'ANTHROPIC_API_KEY': 'k2'}
        with mock.patch.dict(os.environ, env):
            b = make_brain({'provider': 'openai', 'model': 'gpt-x', 'host': 'http://h/v1/'})
            self.assertEqual((type(b).__name__, b.api_key, b.model, b.host), ('OpenAIBrain', 'k1', 'gpt-x', 'http://h/v1'))
            for prov in ('anthropic', 'claude'):
                b = make_brain({'provider': prov, 'model': 'c-x'})
                self.assertEqual((type(b).__name__, b.api_key, b.host), ('AnthropicBrain', 'k2', 'https://api.anthropic.com'))

    def test_missing_env_var_names_variable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as cm:
                make_brain({'provider': 'openai', 'model': 'm'})
            self.assertIn('OPENAI_API_KEY', str(cm.exception))
            with self.assertRaises(ConfigError) as cm:
                make_brain({'provider': 'claude', 'model': 'm'})
            self.assertIn('ANTHROPIC_API_KEY', str(cm.exception))

    def test_limits_default_and_override(self):
        self.assertEqual(get_warn_tokens(load_config(self.dir / 'nope.toml')), 100000)
        p = self.write('[limits]\nwarn_tokens = 5000\n')
        self.assertEqual(get_warn_tokens(load_config(p)), 5000)

    def test_limits_empty_table_keeps_default(self):
        p = self.write('[limits]\n')
        self.assertEqual(get_warn_tokens(load_config(p)), 100000)

    def test_limits_invalid(self):
        for val in ('0', '-1', '"x"', 'true', '1.5'):
            p = self.write(f'[limits]\nwarn_tokens = {val}\n')
            with self.assertRaises(ConfigError, msg=val):
                load_config(p)

    def test_missing_model_raises(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'k'}):
            with self.assertRaises(ConfigError):
                make_brain({'provider': 'openai', 'model': None})

    def test_literal_api_key_rejected(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'k'}):
            with self.assertRaises(ConfigError) as cm:
                make_brain({'provider': 'openai', 'model': 'm', 'api_key': 'sk-secret'})
            self.assertIn('api_key_env', str(cm.exception))
            self.assertNotIn('sk-secret', str(cm.exception))

    def test_custom_api_key_env(self):
        with mock.patch.dict(os.environ, {'MY_KEY': 'abc'}, clear=True):
            self.assertEqual(make_brain({'provider': 'anthropic', 'model': 'm', 'api_key_env': 'MY_KEY'}).api_key, 'abc')
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as cm:
                make_brain({'provider': 'anthropic', 'model': 'm', 'api_key_env': 'MY_KEY'})
            self.assertIn('MY_KEY', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
