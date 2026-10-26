"""Load agent configuration (provider slots) from a TOML file."""

import copy
import os
import tomllib

from brain import AnthropicBrain, OllamaBrain, OpenAIBrain

DEFAULT_PATH = '~/.local-agent/config.toml'
DEFAULTS = {'slots': {'main': {'provider': 'ollama', 'model': None, 'host': None}},
            'limits': {'warn_tokens': 100000}}


class ConfigError(Exception):
    """Raised for unreadable or invalid configuration."""
    pass


def load_config(path=None) -> dict:
    """Load config from path (or AGENT_CONFIG, or the default); merge over defaults."""
    config = copy.deepcopy(DEFAULTS)
    path = os.path.expanduser(path or os.environ.get('AGENT_CONFIG') or DEFAULT_PATH)
    if not os.path.exists(path):
        return config
    try:
        with open(path, 'rb') as f:
            user = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f'invalid TOML in {path}: {e}') from e
    except OSError as e:
        raise ConfigError(f'cannot read {path}: {e}') from e
    slots = user.get('slots')
    if isinstance(slots, dict):
        for name, keys in slots.items():
            if isinstance(keys, dict):
                config['slots'].setdefault(name, {}).update(keys)
    limits = user.get('limits')
    if isinstance(limits, dict):
        config['limits'].update(limits)
    warn = config['limits'].get('warn_tokens')
    if not isinstance(warn, int) or isinstance(warn, bool) or warn <= 0:
        raise ConfigError('limits.warn_tokens must be a positive integer')
    return config


def get_warn_tokens(config) -> int:
    """Return the cumulative-token warning threshold."""
    return config['limits']['warn_tokens']


def get_slot(config, name='main') -> dict:
    """Return the named slot or raise ConfigError."""
    slot = config.get('slots', {}).get(name)
    if slot is None:
        raise ConfigError(f'unknown slot {name!r}')
    return slot


def get_optional_slot(config, name) -> dict:
    """Return the named slot or None (never raises)."""
    return config.get('slots', {}).get(name)


def get_fallback_slot_name(config):
    """Return main.fallback (a slot name) or None; raise ConfigError if invalid."""
    value = get_slot(config, 'main').get('fallback')
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError('main.fallback must be a slot name string')
    if value == 'main':
        raise ConfigError('main.fallback must not be main')
    if get_optional_slot(config, value) is None:
        raise ConfigError(f'unknown fallback slot {value!r}')
    return value


def make_brain(slot):
    """Build a brain for a slot; None host/model fall back inside the brain."""
    provider = slot.get('provider')
    if provider == 'ollama':
        return OllamaBrain(host=slot.get('host'), model=slot.get('model'))
    if provider in ('openai', 'anthropic', 'claude'):
        if 'api_key' in slot:
            raise ConfigError('api_key is not allowed in config; use api_key_env (name of an environment variable)')
        if not slot.get('model'):
            raise ConfigError(f'model is required for provider {provider!r}')
        openai = provider == 'openai'
        name = slot.get('api_key_env') or ('OPENAI_API_KEY' if openai else 'ANTHROPIC_API_KEY')
        key = os.environ.get(name)
        if not key:
            raise ConfigError(f'environment variable {name} is not set')
        cls = OpenAIBrain if openai else AnthropicBrain
        return cls(host=slot.get('host'), model=slot['model'], api_key=key)
    raise ConfigError(f'provider {provider!r} not supported yet')
