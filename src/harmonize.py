"""Step 2: load the four files into one table and make missing values explicit."""
import numpy as np
import pandas as pd

from .config import COLUMNS, INTERIM_DIR, RAW_DIR, SITE_FILES, ZERO_MEANS_MISSING


def load_raw():
    frames = []
    for site, fname in SITE_FILES.items():
        df = pd.read_csv(RAW_DIR / fname, header=None, names=COLUMNS,
                         na_values="?", skipinitialspace=True)
        df.insert(0, "site", site)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def harmonize(df):
    df = df.copy()
    df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    zeros = {}
    for col in ZERO_MEANS_MISSING:
        mask = df[col] == 0
        zeros[col] = df.loc[mask].groupby("site").size().to_dict()
        df.loc[mask, col] = np.nan
    df["target"] = (df["num"] > 0).astype(int)
    return df, zeros


def run():
    raw = load_raw()
    df, zeros = harmonize(raw)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_DIR / "harmonized.csv", index=False)
    return raw, df, zeros
