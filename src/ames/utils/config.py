"""
Tiny helper for reading YAML into a Python dict.
"""

import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
