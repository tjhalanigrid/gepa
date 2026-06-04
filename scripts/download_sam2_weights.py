"""
Downloads SAM2 weights to weights/ directory.
Run this if SAM2 mask generation is not working.

Usage:
    python3 scripts/download_sam2_weights.py
"""

import urllib.request
from pathlib import Path


WEIGHTS = {
    "sam2.1_hiera_base_plus.pt": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
        "sam2.1_hiera_base_plus.pt"
    ),
}


def download():
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    for filename, url in WEIGHTS.items():
        out_path = weights_dir / filename
        if out_path.exists():
            print(f"Already exists: {out_path} ({out_path.stat().st_size/1e6:.0f}MB)")
            continue

        print(f"Downloading {filename}...")
        print(f"  URL: {url}")
        print(f"  Destination: {out_path}")

        def _progress(count, block_size, total_size):
            pct = min(count * block_size * 100 / total_size, 100)
            print(f"\r  {pct:.0f}%", end="", flush=True)

        try:
            urllib.request.urlretrieve(url, out_path, _progress)
            print(f"\n  Done: {out_path.stat().st_size/1e6:.0f}MB")
        except Exception as e:
            print(f"\n  Failed: {e}")
            if out_path.exists():
                out_path.unlink()

    print("\nWeights status:")
    for filename in WEIGHTS:
        p = weights_dir / filename
        status = f"OK ({p.stat().st_size/1e6:.0f}MB)" if p.exists() else "MISSING"
        print(f"  {filename}: {status}")


if __name__ == "__main__":
    download()
