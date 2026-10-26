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
    max_tokens = config['limits'].get('max_tokens')
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0):
        raise ConfigError('limits.max_tokens must be a positive integer')
    if max_tokens is not None and max_tokens < warn:
        raise ConfigError('limits.max_tokens must be >= limits.warn_tokens')
    min_context = config['limits'].get('min_context')
    if min_context is not None and (not isinstance(min_context, int) or isinstance(min_context, bool) or min_context <= 0):
        raise ConfigError('limits.min_context must be a positive integer')
    for name, slot in config['slots'].items():
        if 'context' in slot:
            context = slot['context']
            if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
                raise ConfigError(f'slot {name} context must be a positive integer')
    return config


def get_warn_tokens(config) -> int:
    """Return the cumulative-token warning threshold."""
    return config['limits']['warn_tokens']


def get_max_tokens(config):
    """Return the hard context limit (limits.max_tokens) or None."""
    return config['limits'].get('max_tokens')


def check_min_context(config, slot_names):
    """Raise ConfigError if a named slot's declared context violates limits.min_context or max_tokens."""
    min_context = config['limits'].get('min_context')
    max_tokens = config['limits'].get('max_tokens')
    for name in slot_names:
        slot = get_optional_slot(config, name)
        if slot is None:
            continue
        context = slot.get('context')
        if min_context is not None:
            if context is None:
                raise ConfigError(f'slot {name} must declare context when limits.min_context is set')
            if context < min_context:
                raise ConfigError(f'slot {name} context {context} is below limits.min_context {min_context}')
        if max_tokens is not None and context is not None and max_tokens > context:
            raise ConfigError(f'limits.max_tokens {max_tokens} exceeds slot {name} context {context}')


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
