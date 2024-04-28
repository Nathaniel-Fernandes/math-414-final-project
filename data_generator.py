import os
import subprocess

os.makedirs("data", exist_ok=True)
os.chdir("data")

# Cloning the Dike Dataset
subprocess.run(["git", "clone", "https://github.com/iosifache/DikeDataset.git"])
