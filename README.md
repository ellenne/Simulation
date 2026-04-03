# Simulation

Coursework for the **Simulation** technical assessment. Remote: [github.com/ellenne/Simulation](https://github.com/ellenne/Simulation).

This repository contains **four end-to-end supervised-learning workflows** (one per dataset): exploratory analysis, **ydata-profiling** reports, rich EDA visualisations, feature engineering, **logistic-regression baselines**, and **optimised models** (Random Forest and XGBoost for tabular tasks; tuned logistic regression and calibrated **LinearSVC** for text). Each pipeline produces **held-out test predictions**, **PNG figures**, and a **Section 7** printable Q&A block for the assessment.

---

## What’s in this repo

| Artifact | Description |
|----------|-------------|
| **`DataExploration.ipynb`** (project root) | Cross-dataset narrative: business context, E/R-style sketches, structural checks, plots, **consolidated insights** (see below), **missing-value audit**, **defensive cleaning policy** (aligned with `scripts/data_cleaning.py`), and optional automation cells for the baseline script. |
| **`notebooks/candidate_success.ipynb`**, **`notebooks/complaint_nlp.ipynb`**, **`notebooks/hotel_demand.ipynb`**, **`notebooks/medical_risk.ipynb`** | Submission-ready notebooks: profiling → EDA → feature engineering → baseline → tuned models → predictions + assessment answers. |
| **`scripts/notebook_sources/*.nb.txt`** | Source templates for the four task notebooks; edit here and regenerate. |
| **`scripts/generate_notebooks.py`** | Regenerates the `.ipynb` files under `notebooks/` from `scripts/notebook_sources/`. |
| **`Data/`** | Train/test CSVs (2,000 train / 500 test rows per problem). |
| **`profiles/profile_<dataset>.html`** | ydata-profiling HTML reports. |
| **`output/<dataset>_predictions.csv`** | Test predictions: `id`, `predicted_label`. |
| **`figures/`** | Saved PNGs from notebooks (e.g. `candidate_success_fig_*.png`) and/or `ml_baseline_pipeline.py` (e.g. `candidate_*`, `hotel_*`, `medical_*`, `complaint_*`). |
| **`requirements.txt`** | Python stack (pandas, scikit-learn, xgboost, ydata-profiling, etc.). |
| **`scripts/data_cleaning.py`** | **Train-only** imputation for tabular predictors (median / mode from train), NLP placeholder for empty `complaint_text`, optional CSV audit (`audit_csvs`). Shared policy used by `ml_baseline_pipeline.py`. |
| **`final_answers.py`** | Reads `answers_*.json` from all 4 notebooks and prints/saves a formatted `final_answers.txt` answer sheet. |
| **`build_submission.py`** | Packages notebooks, predictions, profiles, figures, and answers into a timestamped `submission_*.zip`. |
| **`scripts/ml_baseline_pipeline.py`** | Optional **end-to-end** runner for all four problems: loads CSVs → applies cleaning → EDA figures → task-specific feature engineering → **tabular:** scaled **RandomForest** vs **XGBoost** (stratified hold-out, class imbalance handling), validation **accuracy / weighted F1 / ROC-AUC**, confusion matrices and **feature importances**, then retrains the better model on full train and writes `output/*_predictions.csv` → **complaint:** **TF–IDF + logistic regression**, validation report, multiclass ROC-AUC (OVR), confusion matrix. |

---

## Main insights from `DataExploration.ipynb`

The exploration notebook is the **single place** for cross-cutting narrative and quality checks. Highlights below come from its **Insights (summary)** sections, structural reports, and EDA figures.

### Top three (priority)

1. **Complaint NLP — duplicate messages:** The training set has only **~25 unique complaint texts** across 2,000 rows; **1,975** rows duplicate the same `(complaint_text, category_label)` pair (different `id`). That strongly affects modelling (memorisation, split leakage); **deduplication or careful validation** is important when training NLP models.

2. **Medical risk — age and labs:** **Age** has the strongest Pearson association with `risk_label`; **cholesterol** and **blood pressure** follow. **Smokers** and **family history** show higher mean risk than non-smokers / no history. Class mix is roughly **~33%** positive vs **~67%** negative.

3. **Hotel demand — season and lead time:** **Season** separates high vs low demand clearly. **High demand** rows tend to have **shorter lead times** and **lower mean nightly price** in this sample; **hotel_type** is a weaker discriminator than season.

### Ten-point summary

| # | Insight |
|---|--------|
| 1 | Complaints: **~25 unique texts**, massive duplication — high **leakage / overfitting** risk for NLP. |
| 2 | Medical: **Age** strongest Pearson correlation with `risk_label`; cholesterol and BP next. |
| 3 | Medical: **Class imbalance** (~33% high risk). |
| 4 | Medical: **Smoker** and **family_history** lift mean risk in marginal rates. |
| 5 | Hotel: **Season** strongly stratifies `demand_label`. |
| 6 | Hotel: High demand ↔ **lower mean price** and **much shorter lead time**. |
| 7 | Hotel: **Hotel type** shows similar demand rates between types (weaker than season). |
| 8 | Complaints: Categories **roughly balanced**; **service** texts longest on average, **cleanliness** shortest. |
| 9 | Candidate: **Projects completed** and **ML skill score** correlate most with `success_label`. |
| 10 | Candidate: **Education level** is near-flat vs success rate — weak linear predictor here. |

**Cross-cutting:** In the **shipped** CSVs, tabular datasets have **no missing values** and **no duplicate** feature+target rows (ignoring `id`). **Complaint NLP** is the exception on duplicates. For reproducibility and future data drops, **`scripts/data_cleaning.py`** encodes **train-only** imputation and NLP text cleanup; **`scripts/ml_baseline_pipeline.py`** calls it after every `read_csv`. See `DataExploration.ipynb` for full tables, charts, correlations, and the cleaning section.

---

## Datasets (quick reference)

| Problem | Files | Target | Inputs (summary) |
|--------|--------|--------|-------------------|
| Medical risk | `medical_risk_*.csv` | `risk_label` | Vitals and lifestyle (age, BMI, BP, labs, smoker, activity, etc.) |
| Hotel demand | `hotel_demand_*.csv` | `demand_label` | Booking attributes (lead time, stay length, party size, season, price, history, requests) |
| Complaint NLP | `complaint_nlp_*.csv` | `category_label` | `complaint_text` → booking, billing, cleanliness, service, technical |
| Candidate success | `candidate_success_*.csv` | `success_label` | Skills, experience, education, GitHub activity, certifications |

Use the official brief for submission format, allowed libraries, and evaluation rules.

---

## Workflow

1. **Explore** — Open `DataExploration.ipynb` (repository root) for overview and insights.  
2. **Model** — Open the matching notebook under `notebooks/` for profiling, EDA, models, and `Section 7` answers.  
3. **Regenerate notebooks from sources** (after editing `scripts/notebook_sources/<name>.nb.txt`):

   ```powershell
   python scripts/generate_notebooks.py
   ```

4. **Re-run** the notebook (Restart & Run All) so outputs stay in sync.

Optional baseline script (writes figures under `figures/` and prediction CSVs under `output/`; run from repo root so `scripts/` imports resolve):

```powershell
python scripts/ml_baseline_pipeline.py
```

Tabular figure filenames from this script use short prefixes (`candidate_*`, `hotel_*`, `medical_*`). Notebook-generated plots may use longer names such as `candidate_success_fig_*.png`.

---

## Setup

Python 3.10+ is recommended. Dependencies are in `requirements.txt`.

```powershell
cd path\to\Simulation
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In Cursor/VS Code, use **Python: Select Interpreter** and pick `venv\Scripts\python.exe`.

---

## Notes

- Profiling HTML is written under **`profiles/profile_<dataset>.html`**.  
- Test predictions are written under **`output/<dataset>_predictions.csv`** when you run the task notebooks or **`scripts/ml_baseline_pipeline.py`**.  
- Figures are saved under **`figures/`** from the notebooks and/or the baseline script (see workflow above for naming).  
- **`scripts/data_cleaning.py`** centralises missing-data handling; extend there if you change imputation rules so notebooks and the CLI pipeline stay consistent.  
- Prefer opening the project folder as the workspace root so **`Data/`**, **`scripts/`**, and automation cells resolve correctly; each notebook uses `find_data_dir()` which walks up from `cwd` so `Data/` still resolves if the kernel’s working directory is a subfolder (e.g. `notebooks/`).  
- For decisions, assumptions, and limitations specific to your submission, align with the course brief and your write-up.

---

## Execution Order

1. **Install dependencies:**

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ydata-profiling
```

2. **Run all four notebooks** top-to-bottom (Restart & Run All):
   - `notebooks/candidate_success.ipynb`
   - `notebooks/complaint_nlp.ipynb`
   - `notebooks/hotel_demand.ipynb`
   - `notebooks/medical_risk.ipynb`

   Each notebook saves: `profiles/profile_*.html`, `figures/*.png`, `output/*_predictions.csv`, `answers_*.json`.

   The **Claude API cell** at the end of each notebook calls `claude-sonnet-4-20250514` to generate narrative Section 7 answers. Set the `ANTHROPIC_API_KEY` environment variable before running. If the API is unavailable the cell fails gracefully and the existing f-string answers above it remain valid.

3. **Collect all Section 7 answers:**

```powershell
python final_answers.py
```

   Prints and saves `final_answers.txt` (copy-paste ready).

4. **Build submission ZIP:**

```powershell
python build_submission.py
```

   Creates `submission_YYYYMMDD_HHMM.zip`.

## Deliverables

- 4 × `.ipynb` notebooks (one per dataset)
- 4 × `profile_*.html` (ydata-profiling reports)
- 4 × `*_predictions.csv` (test set predictions)
- `final_answers.txt` (all Section 7 answers, copy-paste ready)
- `figures/` (all EDA and evaluation plots)
- `submission_*.zip` (full package)
