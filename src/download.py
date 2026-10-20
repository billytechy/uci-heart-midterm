"""Step 1: use the four UCI files in data/raw/, downloading any that are missing."""
import hashlib
import urllib.request

from .config import BASE_URL, RAW_DIR, SITE_FILES


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    info = {}
    for site, fname in SITE_FILES.items():
        dest = RAW_DIR / fname
        if force or not dest.exists():
            print(f"  downloading {fname}")
            urllib.request.urlretrieve(BASE_URL + fname, dest)
        else:
            print(f"  using data/raw/{fname}")
        lines = [l for l in dest.read_text().splitlines() if l.strip()]
        info[site] = {"file": fname, "rows": len(lines), "sha256": sha256(dest)}
    return info
