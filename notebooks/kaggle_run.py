import os, sys, subprocess

REPO = "https://github.com/abhi6579/geo-zss.git"
DIR  = "/kaggle/working/geo-zss"

if os.path.exists(DIR):
    subprocess.run(["git", "-C", DIR, "pull"], check=True)
else:
    subprocess.run(["git", "clone", REPO, DIR], check=True)

sys.path.insert(0, DIR)
os.system("pip install -q git+https://github.com/openai/CLIP.git")
os.system("pip install -q git+https://github.com/facebookresearch/segment-anything.git")
print("Ready.")
