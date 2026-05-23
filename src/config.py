"""Load the YAML config into a plain dict.

Keeping this tiny on purpose. Scripts call load_config() and read keys.
"""

from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    """Read the config file and return it as a nested dict."""
    path = Path(path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
