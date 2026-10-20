"""Step 4: split, decide which features to keep, impute, encode, export.

Every learned quantity (missing shares used for dropping, medians, modes,
means and scales) comes from the training rows only.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (BINARY, CATEGORICAL, FEATURES, MAX_MISSING_SHARE, NUMERIC,
                     PROCESSED_DIR, SEED, TEST_SIZE)


def split(df):
    strata = df["site"] + "_" + df["target"].astype(str)
    return train_test_split(df, test_size=TEST_SIZE, stratify=strata, random_state=SEED)


def choose_features(train):
    share = train[FEATURES].isna().mean()
    dropped = sorted(share[share > MAX_MISSING_SHARE].index)
    kept = [c for c in FEATURES if c not in dropped]
    return kept, dropped, share.round(3).to_dict()


def build_transformer(kept):
    num = [c for c in NUMERIC if c in kept]
    binary = [c for c in BINARY if c in kept]
    cat = [c for c in CATEGORICAL if c in kept]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), cat),
    ], verbose_feature_names_out=False)


def run(df):
    train, test = split(df)
    kept, dropped, share = choose_features(train)
    transformer = build_transformer(kept)
    X_train = transformer.fit_transform(train[kept])
    X_test = transformer.transform(test[kept])
    names = list(transformer.get_feature_names_out())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, X, part in [("train", X_train, train), ("test", X_test, test)]:
        out = pd.DataFrame(X, columns=names, index=part.index)
        out.insert(0, "site", part["site"])
        out["target"] = part["target"]
        out.to_csv(PROCESSED_DIR / f"{name}.csv", index_label="row_id")

    return {
        "train": train, "test": test, "kept": kept, "dropped": dropped,
        "missing_share_train": share, "encoded_features": names,
    }
