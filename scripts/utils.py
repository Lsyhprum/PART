import numpy as np

# ──────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────


def read_ivecs(filename: str) -> np.ndarray:
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} read, dim={d}")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename: str) -> np.ndarray:
    return read_ivecs(filename).view("float32")



