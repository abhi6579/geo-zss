"""
Download LoveDA dataset directly from official source.
Run this on Kaggle as a setup cell.
"""
import os, subprocess, zipfile

DATA_DIR = "/kaggle/working/data/LoveDA"
os.makedirs(DATA_DIR, exist_ok=True)

# Official LoveDA download from Google Drive
FILES = {
    "Train.zip": "https://drive.google.com/uc?id=1XdZCBVFPuLkBnBJcVKBMekzFxETBjJNP",
    "Val.zip":   "https://drive.google.com/uc?id=1YBh0YXnQDVBtDhITN_iFaOGIYpT4ASFK",
}

# Install gdown for Google Drive downloads
os.system("pip install -q gdown")
import gdown

for fname, url in FILES.items():
    out = os.path.join(DATA_DIR, fname)
    if not os.path.exists(out):
        print(f"Downloading {fname}...")
        gdown.download(url, out, quiet=False)
    else:
        print(f"{fname} already exists")

    # Unzip
    print(f"Extracting {fname}...")
    with zipfile.ZipFile(out, 'r') as z:
        z.extractall(DATA_DIR)
    os.remove(out)

print(f"\nDone. Data at {DATA_DIR}")
os.system(f"find {DATA_DIR} -type d | head -20")
