
import os
import shutil

file_dir = "/data0/llj/workspace/PASD_moco/runs/pasd-DRSRDeg-moco"
interval = 50000

for file in os.listdir(file_dir):
    if file[:4] == "chec":
        _, iter = file.split("-")
        if int(iter) % interval == 0: continue
        else:
            folder_path = os.path.join(file_dir, file)
            shutil.rmtree(folder_path)
            print(f"===> remove {folder_path}")
