from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

# ── SST-2 (primary training data)
print("Downloading SST-2...")
sst2 = load_dataset("sst2")
sst2.save_to_disk("data/raw/sst2")

# ── IMDB (augmentation)
print("Downloading IMDB...")
imdb = load_dataset("imdb")
imdb.save_to_disk("data/raw/imdb")

# ── Amazon Polarity (3-class)
print("Downloading Amazon Polarity...")
amz = load_dataset("amazon_polarity")
amz.save_to_disk("data/raw/amazon")

# ── Yelp (optional)
print("Downloading Yelp Polarity...")
yelp = load_dataset("yelp_polarity")
yelp.save_to_disk("data/raw/yelp")

print("All datasets downloaded to data/raw/")
