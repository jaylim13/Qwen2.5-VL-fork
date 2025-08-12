import os
from tqdm import tqdm

TMPDIR = "/mnt/arc/yygx/tmp/"

all_paths = []
for root, dirs, files in os.walk(TMPDIR, topdown=False):
    for name in files:
        all_paths.append(os.path.join(root, name))
    for name in dirs:
        all_paths.append(os.path.join(root, name))

for path in tqdm(all_paths, desc="Deleting"):
    try:
        if os.path.isfile(path):
            os.remove(path)
        else:
            os.rmdir(path)
    except Exception:
        pass 