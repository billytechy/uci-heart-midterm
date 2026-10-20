# UCI Heart Disease: Site-Aware EDA and Preprocessing

Midterm project, Data Processing for Big Science (Fall 2026)
Muh. Isra Nabil Iksan (Bill) · isra.official19@gmail.com

This repository is the full description of my planned analysis. It is written so that someone else can rerun it from scratch and check, step by step, that they get the same numbers I did. If you only want to run it, go to [Quick start](#quick-start). If something does not match, see [Troubleshooting](#troubleshooting) or email me.

---

## 1. Question and scope

The UCI Heart Disease data comes from four hospitals: Cleveland (US), Budapest (Hungary), Zurich and Basel (Switzerland), and the VA in Long Beach (US). The hospitals differ a lot in how many patients have heart disease and in which variables they actually recorded.

The project asks one practical question:

> **How should this data be cleaned when missing values and disease rates both depend on the hospital?**

The focus is EDA and preprocessing. A simple logistic regression is included at the end only as a sanity check, so there is one final number you can compare against. It is not meant to be a good model.

## 2. Data

| | |
|---|---|
| Source | UCI Machine Learning Repository, Heart Disease (dataset id 45), license CC BY 4.0 |
| Files used | `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data` |
| Size | 920 patients, 14 standard variables, plus a `site` column added by the pipeline |
| Target | `num` (0–4) turned into `target`: 0 = no disease, 1 = disease (`num` ≥ 1) |

The variable codes are in [`docs/data_dictionary.md`](docs/data_dictionary.md).

**Please do not use other copies of this dataset.** They look similar but give different results:

| Copy | Problem |
|---|---|
| `ucimlrepo` package, `fetch_ucirepo(id=45)` | returns Cleveland only (303 rows) |
| Kaggle `heart.csv` (1,025 rows) | duplicated rows and different category codes |
| Kaggle "Heart Failure Prediction" (918 rows) | mixes in a fifth dataset, no hospital column, no `ca`/`thal` |
| Kaggle `heart_disease_uci.csv` (920 rows) | same patients, but categories are text labels and the column is called `thalch`; the code here will not read it |

The four files are already in `data/raw/`, copied unchanged from the UCI archive (CC BY 4.0 allows this with attribution). You don't need to download anything. If you want to check that my copies match UCI's, delete `data/raw/` and run `python run_all.py --force-download`; the file hashes in checkpoint C1 will tell you if anything differs.

## 3. Quick start

You need conda (Miniconda, Miniforge or Anaconda) and an internet connection for installing packages. Tested on macOS (Apple Silicon). It should take about 5 minutes.

```bash
git clone <this repository URL>
cd uci-heart-midterm

conda env create -f environment.yml
conda activate dpbs-midterm

python run_all.py
```

The last line of the output should be:

```
ALL CHECKPOINTS MATCH
```

If you see that, you have reproduced the full pipeline. The figures and tables are in `results/`.

**Step by step in a notebook (optional).** `walkthrough.ipynb` runs the same five steps one at a time and shows the tables and figures in between. Open it in VS Code or Jupyter, select the `dpbs-midterm` kernel and run all cells. The last cell compares your headline numbers with mine.

**Docker (optional).** If conda gives you trouble, Docker runs the same thing in Linux:

```bash
docker build -t uci-heart-midterm .
docker run --rm -v "$PWD":/app uci-heart-midterm
```

## 4. What the pipeline does

`run_all.py` runs five steps in order. Each step prints its key numbers, and those numbers are the checkpoints in [Section 6](#6-checkpoints).

### Step 1. Get the raw files (`src/download.py`)
Uses the four files in `data/raw/` and records the row count and SHA-256 hash of each. If a file is missing, it downloads it from UCI.

### Step 2. Load and harmonize (`src/harmonize.py`)
- Reads the four files with the 14 standard column names and adds a `site` column.
- `?` is read as missing.
- `chol = 0` and `trestbps = 0` are set to missing. Nobody has a cholesterol or blood pressure of zero; these are tests that were not done. Every Swiss patient has `chol = 0`, so without this rule cholesterol looks "complete" when it is in fact entirely absent at one site.
- Creates the binary `target`.
- Writes `data/interim/harmonized.csv`.

### Step 3. Exploratory analysis (`src/eda.py`)
Everything is computed per site and pooled, because the sites are too different for pooled numbers alone to mean much.

| Output | Content |
|---|---|
| `T1_site_summary.csv` | patients, cases, prevalence, mean age, % male per site |
| `T2_missing_by_site_pct.csv` | % missing for every variable at every site |
| `T3_numeric_by_site_median_iqr.csv` | median [IQR] of the five numeric variables |
| `T4_categorical_by_site.csv` | counts per category level, including "missing" |
| `T5_missingness_vs_label.csv` | disease rate among patients with vs without each variable |
| `F1`–`F4` (PNG) | prevalence by site, missingness heatmap, numeric boxplots by site, Spearman correlation |

All in `results/eda/`.

### Step 4. Split, select, impute, encode (`src/preprocess.py`)
1. **Split first.** 80% train, 20% test, stratified by site *and* target, `random_state = 42`. Every step after this learns only from the training rows.
2. **Drop features with more than 30% missing in the training set.** Expected to drop `ca`, `thal` and `slope`.
3. **Impute** numeric variables with the training median and binary/categorical variables with the training mode. Imputation is pooled, not per site (see decision D4).
4. **Encode.** Numeric variables are standardized. `cp` and `restecg` are one-hot encoded because their codes are labels, not quantities.
5. **No missing-value indicator columns** (see decision D5).
6. Writes `data/processed/train.csv` and `test.csv` with the `site` column kept for reference but not used as a feature.

### Step 5. Sanity-check models (`src/baseline.py`)
- **M1, baseline.** Logistic regression on the processed features. Reports 5-fold cross-validated AUC on the training set and AUC and accuracy on the test set, overall and per site.
- **M2, missingness only.** Logistic regression that sees only "was this value missing?" for each of the 13 features. If it reaches an AUC well above 0.5, the missing pattern alone predicts the label. This is the evidence behind decision D5.

Results are in `results/model/` and in `results/checkpoints.json`.

## 5. Decisions

Each choice, the reason and the alternative I rejected are written up in [`docs/decision_log.md`](docs/decision_log.md). In short:

| # | Decision | Main reason |
|---|---|---|
| D1 | Use the four original UCI files | other copies differ in rows and codes |
| D2 | Treat `chol = 0` and `trestbps = 0` as missing | physiologically impossible values |
| D3 | Drop features with > 30% missing (in training data) | imputing most of a column invents data |
| D4 | Pooled, not per-site, imputation | per-site is impossible: Switzerland has no cholesterol values at all |
| D5 | No missing-indicator features, `site` not a feature | missingness tracks the hospital, and the hospital tracks the label |
| D6 | Split before any fitting | prevents test information from leaking into imputation and scaling |

## 6. Checkpoints

Compare your run with these values. `run_all.py` does the comparison automatically (numeric tolerance ±0.002) and prints any difference.

<!-- CHECKPOINTS:START -->
| # | Checkpoint | Expected value |
|---|---|---|
| C1 | Raw rows per file | Cleveland 303, Hungary 294, Switzerland 123, VA Long Beach 200 |
| C2 | Rows after loading | 920 |
| C2 | Zeros turned into missing | chol: Switzerland 123, VA Long Beach 49; trestbps: VA Long Beach 1 |
| C3 | Prevalence % (disease = num >= 1) | Cleveland 45.9, Hungary 36.1, Switzerland 93.5, VA Long Beach 74.5, Pooled 55.3 |
| C4 | Train / test rows | 736 / 184 |
| C4 | Dropped features | ca, slope, thal |
| C4 | Encoded feature count | 15 |
| C5 | M1 CV AUC (train, 5-fold) | 0.875 ± 0.028 |
| C5 | M1 test AUC / accuracy | 0.869 / 0.783 |
| C5 | M2 missingness-only test AUC | 0.741 |

Reference run: Python 3.12.15, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, matplotlib 3.11.2. Full values: `expected/checkpoints.json`.
<!-- CHECKPOINTS:END -->

## 7. What the reference run shows

These are the results the pipeline should give you. They are also the main points for discussion.

- **Disease rates differ a lot between hospitals:** 36% in Hungary, 46% in Cleveland, 75% at the VA and 94% in Switzerland. The pooled rate (55%) doesn't describe any of them.
- **So does the patient mix.** Mean age ranges from 48 (Hungary) to 59 (VA), and the share of men from 68% (Cleveland) to 97% (VA).
- **Missing values depend on the hospital.** Cleveland is almost complete. Elsewhere `ca` is missing for 96–99% of patients, and once zeros are treated as missing, no Swiss patient has a cholesterol value.
- **Missingness alone predicts disease.** M2, which only sees which values are missing, reaches a test AUC of 0.741. This is why the pipeline uses no missing-value indicators (D5).
- **The baseline works unevenly across sites.** M1 reaches a test AUC of 0.869 overall, but 0.955 in Cleveland and 0.733 at the VA. The model does best on the most complete site. The Swiss figures rest on only 2 patients without disease in the test set, so don't read much into them.

## 8. Repository layout

```
uci-heart-midterm/
├── README.md               this guide
├── run_all.py              runs every step and checks the checkpoints
├── walkthrough.ipynb       the same steps one by one, with tables and figures
├── environment.yml         conda environment (pinned versions)
├── Dockerfile              optional container
├── src/
│   ├── config.py           paths, column names, seed, thresholds
│   ├── download.py         step 1
│   ├── harmonize.py        step 2
│   ├── eda.py              step 3
│   ├── preprocess.py       step 4
│   └── baseline.py         step 5
├── docs/
│   ├── data_dictionary.md
│   └── decision_log.md
├── expected/
│   └── checkpoints.json    reference values from my run
├── data/
│   ├── raw/                the four UCI files + heart-disease.names (in git)
│   ├── interim/            created when you run (not in git)
│   └── processed/          created when you run (not in git)
└── results/                outputs of my reference run; running the pipeline overwrites them
```

## 9. Troubleshooting

| Problem | What to do |
|---|---|
| `conda env create` cannot find a package version | Make sure the channel is `conda-forge`. If it still fails, create the environment with the same package names but without the version numbers, run the pipeline, and tell me which versions you got. |
| C1 hash differs | Your copy of a raw file is not the same as mine. Re-clone the repository, or tell me which file. |
| Model numbers differ slightly (3rd decimal) | Usually a different scikit-learn or numpy version. Check `results/checkpoints.json` → `environment`. |
| Anything else | Email me with the full terminal output. |

## 10. Limitations

- Pooled median imputation gives every Swiss patient the same cholesterol value. Cholesterol therefore carries some site information, even though `site` is not a feature. This is a known trade-off of D4 and worth discussing.
- Dropping `ca` and `thal` removes two of the strongest predictors in the Cleveland literature. The baseline AUC will be lower than published Cleveland-only results, and that is expected.
- With 25 Swiss patients in the test set and only 2 without disease, per-site test metrics for Switzerland are unstable.

## Reference

Janosi A, Steinbrunn W, Pfisterer M, Detrano R. *Heart Disease* [Dataset]. UCI Machine Learning Repository, 1988. https://doi.org/10.24432/C52P4X
