import os
import tarfile
import urllib.request
import ssl

# Bypass SSL verification (safe for known URLs like MVTec)
ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747ecd442f9cc06/download/420938113-1629952094/leather.tar.xz"
SAVE_DIR = "data/mvtec"
FILE_NAME = "leather.tar.xz"

os.makedirs(SAVE_DIR, exist_ok=True)
filepath = os.path.join(SAVE_DIR, FILE_NAME)

print("Downloading leather category (~200MB)...")
urllib.request.urlretrieve(URL, filepath)
print("Download complete. Extracting...")

with tarfile.open(filepath, "r:xz") as tar:
    tar.extractall(SAVE_DIR)

print("Done! Dataset extracted to:", SAVE_DIR)