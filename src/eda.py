"""Step 3: describe the data per site and pooled. Writes tables and figures to results/eda/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BINARY, CATEGORICAL, FEATURES, NUMERIC, RESULTS_DIR, SITE_FILES

SITES = list(SITE_FILES)
SHORT = {"Cleveland": "CLE", "Hungary": "HUN", "Switzerland": "SWI", "VA Long Beach": "VA"}
INK, MUTED, GRID = "#16202E", "#6B7684", "#E1E5EA"
DISEASE, NO_DISEASE = "#2563A8", "#CDD3DA"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def site_summary(df):
    g = df.groupby("site", sort=False)
    out = pd.DataFrame({
        "patients": g.size(),
        "disease": g["target"].sum(),
        "prevalence_pct": (g["target"].mean() * 100).round(1),
        "age_mean": g["age"].mean().round(1),
        "male_pct": (g["sex"].mean() * 100).round(1),
    }).reindex(SITES)
    pooled = pd.DataFrame({
        "patients": [len(df)], "disease": [df["target"].sum()],
        "prevalence_pct": [round(df["target"].mean() * 100, 1)],
        "age_mean": [round(df["age"].mean(), 1)],
        "male_pct": [round(df["sex"].mean() * 100, 1)],
    }, index=["Pooled"])
    return pd.concat([out, pooled])


def missing_by_site(df):
    pct = df.groupby("site", sort=False)[FEATURES].apply(lambda x: x.isna().mean() * 100)
    pct = pct.reindex(SITES).T.round(1)
    pct["Pooled"] = (df[FEATURES].isna().mean() * 100).round(1)
    return pct


def numeric_by_site(df):
    rows = []
    for col in NUMERIC:
        row = {"variable": col}
        for site in SITES + ["Pooled"]:
            s = df[col] if site == "Pooled" else df.loc[df["site"] == site, col]
            s = s.dropna()
            row[site] = ("n/a" if s.empty else
                         f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]")
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


def categorical_by_site(df):
    rows = []
    for col in BINARY + CATEGORICAL:
        levels = sorted(df[col].dropna().unique())
        for lvl in levels + ["missing"]:
            row = {"variable": col, "level": lvl}
            for site in SITES:
                s = df.loc[df["site"] == site, col]
                n = s.isna().sum() if lvl == "missing" else (s == lvl).sum()
                row[site] = f"{n} ({n / len(s) * 100:.0f}%)"
            rows.append(row)
    return pd.DataFrame(rows)


def missing_vs_label(df):
    """Disease rate among patients with and without a value. A big gap means the
    missingness itself carries information about the label."""
    rows = []
    for col in FEATURES:
        miss = df[col].isna()
        if miss.sum() == 0:
            continue
        rows.append({
            "variable": col,
            "n_missing": int(miss.sum()),
            "disease_pct_if_observed": round(df.loc[~miss, "target"].mean() * 100, 1),
            "disease_pct_if_missing": round(df.loc[miss, "target"].mean() * 100, 1),
        })
    return pd.DataFrame(rows).set_index("variable")


def fig_prevalence(summary, path):
    s = summary.loc[SITES]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = np.arange(len(SITES))[::-1]
    ax.barh(y, s["disease"], color=DISEASE, height=0.55, label="Heart disease")
    ax.barh(y, s["patients"] - s["disease"], left=s["disease"], color=NO_DISEASE,
            height=0.55, label="No disease", edgecolor="white", linewidth=1.5)
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.text(r["patients"] + 5, yi, f"{r['prevalence_pct']:.1f}%", va="center", color=INK)
    ax.set_yticks(y, SITES)
    ax.set_xlabel("Patients")
    ax.set_xlim(0, s["patients"].max() * 1.18)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Patients and heart-disease prevalence per site", loc="left", color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_missing_heatmap(miss, path):
    m = miss[SITES]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.imshow(m.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(SITES)), SITES, rotation=20, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, "-" if v == 0 else f"{v:.0f}%", ha="center", va="center",
                    color="white" if v > 45 else INK, fontsize=9)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Missing values per site, after harmonization (%)", loc="left", color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_numeric(df, path):
    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(12, 3.2))
    for ax, col in zip(axes, NUMERIC):
        data = [df.loc[df["site"] == s, col].dropna() for s in SITES]
        keep = [i for i, d in enumerate(data) if len(d)]
        ax.boxplot([data[i] for i in keep], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#DCE6F3", color=DISEASE),
                   medianprops=dict(color=INK), whiskerprops=dict(color=MUTED),
                   capprops=dict(color=MUTED),
                   flierprops=dict(marker="o", markersize=2, markeredgecolor=MUTED))
        ax.set_xticks(range(1, len(keep) + 1), [SHORT[SITES[i]] for i in keep],
                      rotation=30)
        ax.set_title(col, color=INK)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    fig.suptitle("Numeric variables by site (missing values excluded)", x=0.01, ha="left",
                 color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_correlation(df, path):
    cols = NUMERIC + BINARY + ["cp", "restecg", "slope", "target"]
    c = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(c.values[i, j]) > 0.6 else INK)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Spearman correlation, pooled (pairwise complete)", loc="left", color=INK)
    fig.savefig(path)
    plt.close(fig)


def run(df):
    out = RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    summary = site_summary(df)
    miss = missing_by_site(df)
    summary.to_csv(out / "T1_site_summary.csv")
    miss.to_csv(out / "T2_missing_by_site_pct.csv")
    numeric_by_site(df).to_csv(out / "T3_numeric_by_site_median_iqr.csv")
    categorical_by_site(df).to_csv(out / "T4_categorical_by_site.csv", index=False)
    mvl = missing_vs_label(df)
    mvl.to_csv(out / "T5_missingness_vs_label.csv")
    fig_prevalence(summary, out / "F1_prevalence_by_site.png")
    fig_missing_heatmap(miss, out / "F2_missing_heatmap.png")
    fig_numeric(df, out / "F3_numeric_by_site.png")
    fig_correlation(df, out / "F4_correlation.png")
    return summary, miss, mvl
