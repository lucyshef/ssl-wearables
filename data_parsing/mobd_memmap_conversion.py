# got boilerplate script from gemini

import os
import numpy as np

filestore = os.path.join("/mnt", "parscratch", "users", "acp25lmc", "ssl-data", "mobd_clean")
filename = "X.npy"
datafile = os.path.join(filestore, filename)

def convert_to_memmap(root_dir, filepath):
    # 1. First pass: Count how many total windows are trapped in the file
    total_windows = 0
    with open(filepath, 'rb') as f:
        while True:
            try:
                block = np.load(f)
                total_windows += block.shape[0]
            except (ValueError, IndexError):
                break

    print(f"Found a total of {total_windows} windows.")

    # 2. Allocate the empty memmap files on disk
    out_dir = os.path.join(root_dir, "memmap")
    os.makedirs(out_dir, exist_ok=True)

    X_mmap = np.memmap(os.path.join(out_dir, "X.npy"), dtype='float32', mode='w+', shape=(total_windows, 300, 3))

    # 3. Second pass: Unpack and stream straight into the memmap
    current_idx = 0
    with open(filepath, 'rb') as f:
        while True:
            try:
                block = np.load(f)
                num_samples = block.shape[0]
                X_mmap[current_idx : current_idx + num_samples] = block
                current_idx += num_samples
            except (ValueError, IndexError):
                break

    X_mmap.flush()
    print("Successfully recovered and converted to memmap!")

if __name__ == "__main__":
    convert_to_memmap(filestore, datafile)