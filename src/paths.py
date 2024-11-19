from pathlib import Path
import os

ROOT = Path(__file__).parents[1]
DATA_PATH = ROOT / "data"
RESULTS_PATH = ROOT / "results"
RESOURCES_PATH = ROOT / "resources"
IMAGES_PATH = ROOT / "images"
FIGURES_PATH = ROOT / "figures"

for p in {DATA_PATH, RESULTS_PATH, RESOURCES_PATH, IMAGES_PATH, FIGURES_PATH}:
    os.makedirs(p, exist_ok=True)
