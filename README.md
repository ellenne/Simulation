# Simulation

Coursework for the **Simulation** assignment. Remote: [github.com/ellenne/Simulation](https://github.com/ellenne/Simulation).

## Datasets

All data files are organized in the **`Data/`** folder at the root of this repository.

Four supervised problems live there, each with separate **train** and **test** CSVs (2,000 rows per split in the current files):

| Problem | Files | Target | Inputs (summary) |
|--------|--------|--------|-------------------|
| Medical risk | `medical_risk_*.csv` | `risk_label` | Vitals and lifestyle (age, BMI, blood pressure, labs, smoker, activity, etc.) |
| Hotel demand | `hotel_demand_*.csv` | `demand_label` | Booking attributes (lead time, stay length, party size, season, price, history, requests) |
| Complaint NLP | `complaint_nlp_*.csv` | `category_label` | Free text `complaint_text` → category (booking, billing, cleanliness, service, technical) |
| Candidate success | `candidate_success_*.csv` | `success_label` | Skills, experience, education, GitHub activity, certifications, etc. |

Use the official brief to confirm metrics, allowed models, and submission format.

## Repository layout

| Path | Description |
|------|-------------|
| `Data/` | Train/test CSVs for the four problems above. |
| `DataExploration.ipynb` | Markdown + code: dataset descriptions, E/R-style sketches, targets and business framing; shape/feature lists; missing values, duplicates, and outlier-style checks. |
| `requirements.txt` | Pinned dependency ranges for the notebook (`pandas`, `ipython`, `ipykernel`). |
| `venv/` | Local virtual environment (gitignored). Create with the commands below. |

## What you still need to do

- [ ] **Align with the brief** — Deliverables (code, report, limits on libraries), deadlines, and evaluation rules.
- [x] **First-pass data understanding** — Started in `DataExploration.ipynb` (extend as you model).
- [ ] **Modeling** — Implement and validate approaches per task; record metrics on the held-out test split as required.
- [ ] **Write-up** — Assumptions, methods, results, and limitations for submission.

## Setup

Python 3.10+ is recommended. Dependencies are listed in `requirements.txt` (**pandas**, **ipython** for `IPython.display`, **ipykernel** so Cursor/VS Code can run `.ipynb` files).

```powershell
cd path\to\Simulation
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In Cursor/VS Code, choose **Python: Select Interpreter** and pick `venv\Scripts\python.exe`.

**Jupyter kernel “Simulation”** (optional): register this `venv` so it appears as a named kernel:

```powershell
.\venv\Scripts\python.exe -m ipykernel install --user --name=simulation --display-name="Simulation"
```

To **replace** an existing `simulation` kernel, remove it first (`ipykernel install` has no `--force` flag):

```powershell
.\venv\Scripts\python.exe -m jupyter kernelspec remove simulation -f
.\venv\Scripts\python.exe -m ipykernel install --user --name=simulation --display-name="Simulation"
```

To open a classic Jupyter UI in the browser (optional), run `pip install jupyter` inside the same environment, then `jupyter notebook` or `jupyter lab`.

If a broken `.venv` folder is left over from an interrupted install and you cannot delete it, close editors using that path and remove the folder manually, or keep using `venv` as above.

## Notes

*(Decisions, references, and open questions.)*
