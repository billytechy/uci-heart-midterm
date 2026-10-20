"""Run the whole pipeline and check every checkpoint against expected/checkpoints.json.

    python run_all.py                  # run and compare with the expected values
    python run_all.py --force-download # re-download the raw files first
    python run_all.py --save-expected  # (author only) store this run as the reference
"""
import argparse
import json
import platform
import re
import sys

import matplotlib
import numpy
import pandas
import sklearn

from src import baseline, download, eda, harmonize, preprocess
from src.config import EXPECTED_DIR, RESULTS_DIR, ROOT

TOL = 0.002  # numeric tolerance when comparing with the expected run


def header(text):
    print(f"\n== {text}")


def collect(force_download):
    cp = {}

    header("Step 1  Download raw files")
    cp["C1_raw_files"] = download.run(force=force_download)
    for site, v in cp["C1_raw_files"].items():
        print(f"  {site:<14} {v['rows']:>4} rows  sha256 {v['sha256'][:12]}...")

    header("Step 2  Load and harmonize")
    raw, df, zeros = harmonize.run()
    cp["C2_rows_total"] = len(df)
    cp["C2_zeros_set_to_missing"] = zeros
    cp["C2_missing_counts_after_harmonize"] = {
        c: int(n) for c, n in df.drop(columns=["site", "num", "target"]).isna().sum().items()}
    print(f"  rows: {len(df)}   zero values turned into missing: {zeros}")

    header("Step 3  Exploratory analysis")
    summary, miss, mvl = eda.run(df)
    cp["C3_prevalence_pct"] = summary["prevalence_pct"].to_dict()
    cp["C3_missing_pct_by_site"] = {s: miss[s].to_dict() for s in miss.columns}
    print(summary.to_string())
    print("  tables and figures written to results/eda/")

    header("Step 4  Split, select, impute, encode")
    prep = preprocess.run(df)
    cp["C4_train_rows"] = len(prep["train"])
    cp["C4_test_rows"] = len(prep["test"])
    cp["C4_dropped_features"] = prep["dropped"]
    cp["C4_encoded_feature_count"] = len(prep["encoded_features"])
    print(f"  train {len(prep['train'])} / test {len(prep['test'])}")
    print(f"  dropped (> 30% missing in train): {prep['dropped']}")
    print(f"  encoded features: {len(prep['encoded_features'])}")

    header("Step 5  Baseline models")
    res = baseline.run(prep)
    cp["C5_models"] = res
    print(f"  M1 5-fold CV AUC (train): {res['m1_cv_auc_mean']} +/- {res['m1_cv_auc_sd']}")
    print(f"  M1 test AUC: {res['m1_test_auc']}   test accuracy: {res['m1_test_accuracy']}")
    print(f"  M2 missingness-only test AUC: {res['m2_missingness_only_test_auc']}")

    cp["environment"] = {
        "python": platform.python_version(), "pandas": pandas.__version__,
        "numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    return cp


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                out.update(flatten(item, f"{key}.{item.get('site', '')}."))
        else:
            out[key] = v
    return out


def compare(current, expected):
    cur, exp = flatten(current), flatten(expected)
    problems = []
    for key, want in exp.items():
        if key.startswith("environment"):
            continue
        got = cur.get(key, "<missing>")
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            ok = abs(got - want) <= TOL
        else:
            ok = got == want
        if not ok:
            problems.append((key, want, got))
    return problems


def update_readme(cp):
    m = cp["C5_models"]
    rows = [
        ("C1", "Raw rows per file", ", ".join(f"{s} {v['rows']}" for s, v in cp["C1_raw_files"].items())),
        ("C2", "Rows after loading", str(cp["C2_rows_total"])),
        ("C2", "Zeros turned into missing", "; ".join(
            f"{c}: " + ", ".join(f"{s} {n}" for s, n in v.items()) for c, v in cp["C2_zeros_set_to_missing"].items() if v)),
        ("C3", "Prevalence % (disease = num >= 1)", ", ".join(f"{s} {v}" for s, v in cp["C3_prevalence_pct"].items())),
        ("C4", "Train / test rows", f"{cp['C4_train_rows']} / {cp['C4_test_rows']}"),
        ("C4", "Dropped features", ", ".join(cp["C4_dropped_features"]) or "none"),
        ("C4", "Encoded feature count", str(cp["C4_encoded_feature_count"])),
        ("C5", "M1 CV AUC (train, 5-fold)", f"{m['m1_cv_auc_mean']} ± {m['m1_cv_auc_sd']}"),
        ("C5", "M1 test AUC / accuracy", f"{m['m1_test_auc']} / {m['m1_test_accuracy']}"),
        ("C5", "M2 missingness-only test AUC", str(m["m2_missingness_only_test_auc"])),
    ]
    env = cp["environment"]
    table = ["| # | Checkpoint | Expected value |", "|---|---|---|"]
    table += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    table.append("")
    table.append(f"Reference run: Python {env['python']}, pandas {env['pandas']}, numpy {env['numpy']}, "
                 f"scikit-learn {env['scikit-learn']}, matplotlib {env['matplotlib']}. "
                 f"Full values: `expected/checkpoints.json`.")
    readme = ROOT / "README.md"
    text = readme.read_text()
    new = re.sub(r"<!-- CHECKPOINTS:START -->.*<!-- CHECKPOINTS:END -->",
                 "<!-- CHECKPOINTS:START -->\n" + "\n".join(table) + "\n<!-- CHECKPOINTS:END -->",
                 text, flags=re.S)
    readme.write_text(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--save-expected", action="store_true")
    args = ap.parse_args()

    cp = collect(args.force_download)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "checkpoints.json").write_text(json.dumps(cp, indent=2))

    if args.save_expected:
        EXPECTED_DIR.mkdir(exist_ok=True)
        (EXPECTED_DIR / "checkpoints.json").write_text(json.dumps(cp, indent=2))
        update_readme(cp)
        print("\nSaved as the reference run (expected/checkpoints.json, README table updated).")
        return

    ref = EXPECTED_DIR / "checkpoints.json"
    header("Comparison with the reference run")
    if not ref.exists():
        print("  expected/checkpoints.json not found, nothing to compare.")
        return
    problems = compare(cp, json.loads(ref.read_text()))
    if not problems:
        print("  ALL CHECKPOINTS MATCH")
    else:
        print(f"  {len(problems)} checkpoint(s) differ:")
        for key, want, got in problems:
            print(f"    {key}: expected {want}, got {got}")
        sys.exit(1)


if __name__ == "__main__":
    main()
