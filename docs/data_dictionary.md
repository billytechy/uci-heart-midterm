# Data dictionary

The 14 standard variables, with the codes used in the UCI `processed.*.data` files. In the raw files, missing values are written as `?`.

| # | Variable | Meaning | Type | Values / unit | Used as feature? |
|---|---|---|---|---|---|
| 1 | `age` | Age | numeric | years | yes |
| 2 | `sex` | Sex | binary | 1 = male, 0 = female | yes |
| 3 | `cp` | Chest pain type | categorical | 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic | yes, one-hot |
| 4 | `trestbps` | Resting blood pressure on admission | numeric | mm Hg; 0 treated as missing | yes |
| 5 | `chol` | Serum cholesterol | numeric | mg/dl; 0 treated as missing | yes |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | binary | 1 = true, 0 = false | yes |
| 7 | `restecg` | Resting ECG | categorical | 0 normal, 1 ST-T wave abnormality, 2 probable or definite left ventricular hypertrophy (Estes' criteria) | yes, one-hot |
| 8 | `thalach` | Maximum heart rate during exercise test | numeric | beats per minute | yes |
| 9 | `exang` | Exercise-induced angina | binary | 1 = yes, 0 = no | yes |
| 10 | `oldpeak` | ST depression, exercise relative to rest | numeric | mm | yes |
| 11 | `slope` | Slope of the peak exercise ST segment | ordinal | 1 upsloping, 2 flat, 3 downsloping | dropped if > 30% missing (D3) |
| 12 | `ca` | Major vessels coloured by fluoroscopy | discrete | 0–3 | dropped if > 30% missing (D3) |
| 13 | `thal` | Thallium scan | categorical | 3 normal, 6 fixed defect, 7 reversible defect | dropped if > 30% missing (D3) |
| 14 | `num` | Angiographic diagnosis | ordinal | 0 = < 50% diameter narrowing in any major vessel; 1–4 = > 50% | source of the target |

Added by the pipeline:

| Variable | Meaning |
|---|---|
| `site` | Hospital: Cleveland, Hungary, Switzerland, VA Long Beach. Kept for stratification and reporting, not used as a feature. |
| `target` | 0 if `num` = 0, 1 if `num` ≥ 1 |

Note: `cp`, `restecg` and `thal` are stored as numbers, but the numbers are labels. `thal` = 7 is not "more" than `thal` = 3.
