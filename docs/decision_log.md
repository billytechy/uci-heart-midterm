# Decision log

One entry per decision: what I chose, why, what I rejected, and where it happens in the code.

---

### D1. Data source: the four original UCI files

- **Chosen:** `processed.{cleveland,hungarian,switzerland,va}.data` from the UCI archive, stored unchanged in `data/raw/` together with UCI's own description file `heart-disease.names`. The code downloads them again only if they are missing.
- **Why:** these are the original files and keep the hospital of each patient. Shipping them in the repository means the pipeline still runs if the UCI site is down or changes, and the SHA-256 hashes in checkpoint C1 show that everyone starts from identical bytes.
- **Rejected:** `ucimlrepo` (Cleveland only), Kaggle copies (different rows, codes or columns; Kaggle also needs a login to download); `reprocessed.hungarian.data` from the same UCI folder (a separately cleaned version of the Hungarian file, not used in the standard 920-patient set).
- **Code:** `src/download.py`, `src/config.py`

### D2. Impossible zeros are missing values

- **Chosen:** `chol = 0` and `trestbps = 0` become missing.
- **Why:** both are physiologically impossible. In practice they mark a test that was not done. All 123 Swiss patients have `chol = 0`.
- **Rejected:** keeping them as zeros. A model would learn "cholesterol 0 = Swiss = almost always disease".
- **Code:** `src/harmonize.py` (`ZERO_MEANS_MISSING` in `config.py`)

### D3. Drop features with more than 30% missing

- **Chosen:** compute the missing share of each feature on the training set; drop any above 30%. Expected result: `ca`, `thal`, `slope` are dropped.
- **Why:** above that level, imputation mostly invents values, and for these three the missing values come almost entirely from three of the four hospitals, so the imputed value would mostly encode the hospital.
- **Rejected:** (a) imputing them anyway; (b) using only complete cases, which would leave essentially Cleveland and remove the multi-site character of the data; (c) using only Cleveland.
- **Cost:** `ca` and `thal` are strong predictors in the Cleveland literature, so model performance is lower. Accepted, because the goal is a defensible pipeline, not the highest score.
- **Code:** `src/preprocess.py` → `choose_features`, threshold `MAX_MISSING_SHARE` in `config.py`

### D4. Pooled imputation, not per site

- **Chosen:** one median (numeric) or mode (binary, categorical) per feature, learned from all training rows.
- **Why:** per-site imputation is impossible for cholesterol (Switzerland has no observed value) and unstable for other variables with few observed values at small sites.
- **Rejected:** per-site medians; model-based imputation such as KNN or iterative imputation (more moving parts to reproduce, and with site-dependent missingness they mostly borrow values from Cleveland).
- **Known side effect:** every Swiss patient receives the same cholesterol value. See README, Limitations.
- **Code:** `src/preprocess.py` → `build_transformer`

### D5. No missing-indicator features; `site` is not a feature

- **Chosen:** do not add "value was missing" columns, and do not give the model the hospital.
- **Why:** missingness depends on the hospital, and disease prevalence depends on the hospital (36% to 94%). A model given either could predict the label from where the patient was treated rather than from clinical values. Model M2 in `src/baseline.py` measures this directly: it uses only missing-value flags.
- **Rejected:** missing indicators, which are often recommended in general practice but would act as a proxy for site here.
- **Code:** `src/preprocess.py`, `src/baseline.py` (M2)

### D6. Split before fitting anything

- **Chosen:** stratified 80/20 split on site × target with seed 42, before feature selection, imputation and scaling.
- **Why:** if imputation or scaling is fitted on all rows, test-set information leaks into training.
- **Rejected:** cleaning the full table first and splitting afterwards.
- **Code:** `src/preprocess.py` → `split`

### D7. Fixed versions and one seed

- **Chosen:** pinned package versions in `environment.yml`; `SEED = 42` in `config.py` is the only source of randomness.
- **Why:** default behaviour and numerical results can change between library versions.
- **Code:** `environment.yml`, `src/config.py`
