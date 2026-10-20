"""Step 5: two small logistic-regression checks.

M1  baseline on the kept clinical features (the pipeline's final number).
M2  missingness-only model: uses nothing but "was this value missing?" flags for
    all 13 features. If it scores well above 0.5 AUC, missingness alone leaks
    the label (through the site), which is why M1 uses no missing-indicator flags.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import FEATURES, RESULTS_DIR, SEED
from .preprocess import build_transformer


def _model():
    return LogisticRegression(max_iter=1000)


def run(prep):
    train, test, kept = prep["train"], prep["test"], prep["kept"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    m1 = Pipeline([("prep", build_transformer(kept)), ("clf", _model())])
    cv_auc = cross_val_score(m1, train[kept], train["target"], cv=cv, scoring="roc_auc")
    m1.fit(train[kept], train["target"])
    p = m1.predict_proba(test[kept])[:, 1]
    yhat = (p >= 0.5).astype(int)

    per_site = []
    for site, idx in test.groupby("site", sort=False).groups.items():
        y = test.loc[idx, "target"]
        ps = pd.Series(p, index=test.index).loc[idx]
        per_site.append({
            "site": site, "n_test": len(y), "n_disease": int(y.sum()),
            "accuracy": round(accuracy_score(y, (ps >= 0.5).astype(int)), 3),
            "auc": round(roc_auc_score(y, ps), 3) if y.nunique() == 2 else None,
        })

    flags_train = train[FEATURES].isna().astype(int)
    flags_test = test[FEATURES].isna().astype(int)
    m2 = _model().fit(flags_train, train["target"])
    m2_auc = roc_auc_score(test["target"], m2.predict_proba(flags_test)[:, 1])

    coef = pd.Series(m1.named_steps["clf"].coef_[0],
                     index=m1.named_steps["prep"].get_feature_names_out())
    out = RESULTS_DIR / "model"
    out.mkdir(parents=True, exist_ok=True)
    coef.sort_values().round(3).to_csv(out / "M1_coefficients.csv", header=["coef"])
    pd.DataFrame(per_site).to_csv(out / "M1_test_by_site.csv", index=False)

    return {
        "m1_cv_auc_mean": round(float(np.mean(cv_auc)), 3),
        "m1_cv_auc_sd": round(float(np.std(cv_auc)), 3),
        "m1_test_auc": round(float(roc_auc_score(test["target"], p)), 3),
        "m1_test_accuracy": round(float(accuracy_score(test["target"], yhat)), 3),
        "m1_test_by_site": per_site,
        "m2_missingness_only_test_auc": round(float(m2_auc), 3),
    }
