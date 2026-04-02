"""
download_datasets.py
────────────────────
One-shot downloader for all project datasets.
Run once from the project root: python download_datasets.py
"""
import os
from datasets import load_dataset
from loguru import logger

os.makedirs("data/raw", exist_ok=True)

DATASETS = [
    ("sst2",           "data/raw/sst2"),
    ("imdb",           "data/raw/imdb"),
    ("amazon_polarity","data/raw/amazon"),
    ("yelp_polarity",  "data/raw/yelp"),
]

for name, path in DATASETS:
    if os.path.exists(path):
        logger.info(f"Already exists: {path}")
        continue
    logger.info(f"Downloading {name}...")
    try:
        ds = load_dataset(name)
        ds.save_to_disk(path)
        logger.success(f"✔ {name} saved to {path}")
    except Exception as e:
        logger.warning(f"Failed to download {name}: {e}")

logger.success("Done — datasets in data/raw/")
