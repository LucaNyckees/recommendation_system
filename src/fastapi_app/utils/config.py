import toml
import os

from src.paths import ROOT

with open(os.path.join(ROOT, "config.toml"), "r") as f:
    config = toml.load(f)
