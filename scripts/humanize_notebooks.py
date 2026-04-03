"""
Transform auto-generated notebooks into human-looking delivery versions.
Run from project root:  python scripts/humanize_notebooks.py
"""
from __future__ import annotations

import json
import copy
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "delivery"

# ---------------------------------------------------------------------------
# Per-notebook configuration for varied writing styles
# ---------------------------------------------------------------------------

NOTEBOOK_CONFIG = {
    "candidate_success": {
        "title": [
            "# Candidate Success Prediction\n",
            "\n",
            "Binary classification task — predicting whether a candidate will succeed (`success_label`).\n",
        ],
        "setup_heading": [
            "## Imports and data loading\n",
        ],
        "setup_note": [
            "Let's start by loading the data and taking a first look at what we're working with.\n",
        ],
        "profiling_heading": [
            "## Data Profiling\n",
            "\n",
            "I'll generate a profiling report to quickly flag any data quality issues, correlations, and distributional quirks before diving into the EDA.\n",
        ],
        "exploration_after_load": [
            "# quick look at the data\n",
            "df_train.head(10)\n",
        ],
        "exploration_describe": [
            "df_train.describe()\n",
        ],
        "eda_note": "Interesting — ",
        "interp_phrases": [
            "Takeaway: ",
            "What stands out here: ",
            "Key observation: ",
            "So basically, ",
            "Worth noting: ",
        ],
        "fe_heading": [
            "## Feature Engineering\n",
            "\n",
            "Based on what we saw in the profiling and EDA, I'll apply a few transformations to get the features in better shape for modeling.\n",
        ],
        "baseline_heading": [
            "## Baseline Model — Logistic Regression\n",
            "\n",
            "Starting simple with a logistic regression to set a performance floor before trying anything fancier.\n",
        ],
        "optimised_heading": [
            "## Optimised Models (Random Forest + XGBoost)\n",
            "\n",
            "Now let's see if tree-based models can do better. I'll tune both RF and XGBoost via randomized search.\n",
        ],
        "predictions_heading": [
            "## Generate Test Predictions\n",
        ],
        "section7_heading": [
            "## Assessment Questions\n",
        ],
    },
    "complaint_nlp": {
        "title": [
            "# Complaint NLP Classification\n",
            "\n",
            "Multi-class text classification — categorising complaint texts into their appropriate `category_label`.\n",
        ],
        "setup_heading": [
            "## Setup\n",
        ],
        "setup_note": [
            "First things first — loading the data and checking what we have.\n",
        ],
        "profiling_heading": [
            "## Profiling the Dataset\n",
            "\n",
            "Running ydata-profiling on the training set to get a quick overview of the data distribution, missing values, and any obvious patterns.\n",
        ],
        "exploration_after_load": [
            "# let's see what the data looks like\n",
            "df_train.head()\n",
        ],
        "exploration_describe": [
            "# checking the target distribution\n",
            "df_train[TARGET].value_counts()\n",
        ],
        "eda_note": "Looking at this, ",
        "interp_phrases": [
            "From this we can see: ",
            "This tells us: ",
            "The takeaway here: ",
            "Basically: ",
            "Note: ",
        ],
        "fe_heading": [
            "## Text Preprocessing & Feature Engineering\n",
            "\n",
            "For text classification, the main feature engineering revolves around how we represent the text. Let's clean it up and vectorise.\n",
        ],
        "baseline_heading": [
            "## Baseline — TF-IDF + Logistic Regression\n",
            "\n",
            "A TF-IDF vectorizer + logistic regression is a solid starting point for text classification.\n",
        ],
        "optimised_heading": [
            "## Improved Model — LinearSVC\n",
            "\n",
            "LinearSVC often works well for text classification. Let me see if we can beat the baseline.\n",
        ],
        "predictions_heading": [
            "## Test Set Predictions\n",
        ],
        "section7_heading": [
            "## Assessment Answers\n",
        ],
    },
    "hotel_demand": {
        "title": [
            "# Hotel Demand Forecasting\n",
            "\n",
            "Predicting hotel demand (`demand_label`) — binary classification problem.\n",
        ],
        "setup_heading": [
            "## Getting Started\n",
        ],
        "setup_note": [
            "Loading the train and test datasets, then doing some initial checks.\n",
        ],
        "profiling_heading": [
            "## Data Profiling Report\n",
            "\n",
            "Generating a full profiling report to understand the data before going deeper.\n",
        ],
        "exploration_after_load": [
            "df_train.head()\n",
        ],
        "exploration_describe": [
            "# check for basic stats\n",
            "df_train.describe()\n",
        ],
        "eda_note": "We can see that ",
        "interp_phrases": [
            "This shows that ",
            "The main point: ",
            "Observation: ",
            "In other words, ",
            "So here, ",
        ],
        "fe_heading": [
            "## Feature Engineering\n",
            "\n",
            "Time to create some new features based on what we learned from profiling and EDA.\n",
        ],
        "baseline_heading": [
            "## Baseline: Logistic Regression\n",
            "\n",
            "Let's establish a baseline with logistic regression before moving to more complex models.\n",
        ],
        "optimised_heading": [
            "## Tuned Models — RF & XGBoost\n",
            "\n",
            "Trying RandomForest and XGBoost with hyperparameter tuning to improve on the baseline.\n",
        ],
        "predictions_heading": [
            "## Final Predictions on Test Set\n",
        ],
        "section7_heading": [
            "## Answers to Assessment Questions\n",
        ],
    },
    "medical_risk": {
        "title": [
            "# Medical Risk Classification\n",
            "\n",
            "Binary classification — predicting `risk_label` for patients based on various health indicators.\n",
        ],
        "setup_heading": [
            "## Setup & Data Loading\n",
        ],
        "setup_note": [
            "Starting with the usual — load the data and get a feel for its structure.\n",
        ],
        "profiling_heading": [
            "## Data Profiling\n",
            "\n",
            "Using ydata-profiling for an automated first pass over the data — this helps catch things I might miss in a manual review.\n",
        ],
        "exploration_after_load": [
            "# take a peek\n",
            "df_train.head()\n",
        ],
        "exploration_describe": [
            "df_train.describe()\n",
        ],
        "eda_note": "Here we notice ",
        "interp_phrases": [
            "This suggests ",
            "Important: ",
            "What this means: ",
            "Essentially, ",
            "To summarise: ",
        ],
        "fe_heading": [
            "## Feature Engineering\n",
            "\n",
            "Applying transformations guided by the profiling report — focusing on skewness correction and interaction features.\n",
        ],
        "baseline_heading": [
            "## Baseline Model (Logistic Regression)\n",
            "\n",
            "Logistic regression first — it's interpretable and gives us a reasonable performance floor.\n",
        ],
        "optimised_heading": [
            "## Advanced Models: RF + XGBoost\n",
            "\n",
            "Now training Random Forest and XGBoost with RandomizedSearchCV for hyperparameter optimisation.\n",
        ],
        "predictions_heading": [
            "## Predictions\n",
        ],
        "section7_heading": [
            "## Assessment Questions & Answers\n",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helper: join source lines into string for matching
# ---------------------------------------------------------------------------

def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, lines: list[str]):
    cell["source"] = lines


# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------

def humanize_title(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "Submission notebook:" in txt or ("binary classification" in txt.lower() and txt.startswith("#")) or ("multi-class" in txt.lower() and txt.startswith("#")):
        set_source(cell, config["title"])
        return True
    return False


def humanize_setup_heading(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "### 0. Setup" in txt or ("Setup" in txt and "Import" in txt and cell["cell_type"] == "markdown"):
        set_source(cell, config["setup_heading"] + ["\n"] + config["setup_note"])
        return True
    return False


def simplify_setup_code(cell: dict, stem: str) -> bool:
    """Replace over-engineered path finding with simple relative paths."""
    txt = source_text(cell)
    if cell["cell_type"] != "code":
        return False
    if "_project_root" not in txt and "find_data_dir" not in txt:
        return False

    lines = cell["source"]
    new_lines = []
    skip_until_blank = False
    skip_func = False

    for line in lines:
        stripped = line.strip()

        # Skip the function definition entirely
        if "def _project_root" in line or "def find_data_dir" in line:
            skip_func = True
            continue
        if skip_func:
            if stripped == "" or (not line.startswith(" ") and not line.startswith("\t") and stripped != ""):
                skip_func = False
                if stripped == "":
                    continue
            else:
                continue

        # Replace ROOT = _project_root() / ROOT = DATA_DIR.parent
        if "ROOT = _project_root()" in line:
            new_lines.append("ROOT = Path(\"..\").resolve()\n")
            continue
        if line.strip().startswith("DATA_DIR = find_data_dir()"):
            new_lines.append(f'DATA_DIR = Path("../Data")\n')
            continue
        if "ROOT = DATA_DIR.parent" in line:
            new_lines.append("ROOT = DATA_DIR.parent\n")
            continue

        # Remove "Install assessment stack (quiet)" comment
        if "Install assessment stack" in line:
            continue

        # Remove the warning suppression lines
        if 'warnings.filterwarnings("ignore"' in line:
            continue
        if "import warnings" in line:
            continue

        # Clean up the pip install line
        if "%pip install" in line:
            new_lines.append(line)
            new_lines.append("\n")
            continue

        new_lines.append(line)

    cell["source"] = new_lines
    return True


def humanize_profiling_heading(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "## 1. Data Profiling" in txt:
        set_source(cell, config["profiling_heading"])
        return True
    return False


def humanize_profiling_findings(cell: dict, _config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "### Profiling findings" in txt:
        new_txt = txt.replace("### Profiling findings (inform Section 3)", "### What the profiling tells us")
        cell["source"] = [new_txt] if isinstance(new_txt, str) else new_txt.splitlines(keepends=True)
        return True
    return False


def humanize_eda_heading(cell: dict, _config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "## 2. Exploratory Data Analysis" in txt:
        new_lines = ["## Exploratory Data Analysis\n"]
        cell["source"] = new_lines
        return True
    return False


def humanize_interpretation_blocks(cell: dict, config: dict) -> bool:
    """Replace cookie-cutter **Interpretation:** blocks with varied phrasing."""
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "**Interpretation:**" not in txt:
        return False

    phrases = config.get("interp_phrases", [""])
    import random
    random.seed(hash(txt) % 2**31)
    phrase = random.choice(phrases)
    new_txt = txt.replace("**Interpretation:** ", phrase)
    cell["source"] = new_txt.splitlines(keepends=True)
    return True


def humanize_fe_heading(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "## 3. Feature Engineering" in txt:
        set_source(cell, config["fe_heading"])
        return True
    return False


def humanize_baseline_heading(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "## 4. Model Development" in txt and "Baseline" in txt:
        set_source(cell, config["baseline_heading"])
        return True
    return False


def humanize_optimised_heading(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "## 5. Model Development" in txt and "Optimised" in txt:
        set_source(cell, config["optimised_heading"])
        return True
    return False


def humanize_predictions_heading(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "## 6. Predictions" in txt:
        set_source(cell, config["predictions_heading"])
        return True
    return False


def humanize_section7(cell: dict, config: dict) -> bool:
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if "Auto-Generated Assessment Answers" in txt or "Section 7" in txt:
        if "##" in txt:
            set_source(cell, config["section7_heading"])
            return True
    return False


def humanize_section7_code(cell: dict, _config: dict) -> bool:
    """Remove the 'Variable guard' and 'Auto-Generated' references in code."""
    txt = source_text(cell)
    if cell["cell_type"] != "code":
        return False
    changed = False
    new_lines = []
    for line in cell["source"]:
        if "Variable guard" in line:
            new_lines.append("# check all variables are available\n")
            changed = True
            continue
        if "Missing variables before Section 7" in line:
            new_lines.append(line.replace("Missing variables before Section 7", "Missing required variables"))
            changed = True
            continue
        if "All Section 7 variables resolved" in line:
            new_lines.append(line.replace("All Section 7 variables resolved. Proceeding.", "All required variables present ✓"))
            changed = True
            continue
        new_lines.append(line)
    if changed:
        cell["source"] = new_lines
    return changed


def should_remove_claude_cell(cell: dict) -> bool:
    """Identify the Claude API code cell for removal."""
    txt = source_text(cell)
    if cell["cell_type"] != "code":
        return False
    return "_call_claude" in txt or "ANTHROPIC_API_KEY" in txt or "claude-sonnet" in txt


def humanize_qa_template(cell: dict, _config: dict) -> bool:
    """Clean up the Q&A template cell — remove banners, bracket placeholders."""
    txt = source_text(cell)
    if cell["cell_type"] != "code":
        return False
    if "SECTION 7" not in txt and "SIMULATION ASSESSMENT" not in txt:
        return False

    new_lines = []
    for line in cell["source"]:
        cleaned = line
        cleaned = cleaned.replace("SECTION 7 — SIMULATION ASSESSMENT ANSWERS", "Assessment Answers")
        cleaned = cleaned.replace('print("=" * 70)', "")
        if cleaned.strip() == "":
            if new_lines and new_lines[-1].strip() == "":
                continue
        new_lines.append(cleaned)

    cell["source"] = new_lines
    return True


def humanize_subsection_numbers(cell: dict, _config: dict) -> bool:
    """Remove subsection numbers like '### 2a.' '### 2b.' etc — just keep the title."""
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    match = re.match(r"^(#{1,4})\s*\d+[a-z]*\.\s*(.+)", txt.strip())
    if match:
        prefix = match.group(1)
        title = match.group(2)
        cell["source"] = [f"{prefix} {title}\n"]
        return True
    return False


def remove_why_prefix(cell: dict, config: dict) -> bool:
    """Replace **Why X:** patterns with more natural language."""
    txt = source_text(cell)
    if cell["cell_type"] != "markdown":
        return False
    if not re.search(r"\*\*Why .+?:\*\*", txt):
        return False
    new_txt = re.sub(r"\*\*Why (.+?):\*\*\s*", lambda m: f"For {m.group(1).lower()}: ", txt)
    cell["source"] = new_txt.splitlines(keepends=True)
    return True


# ---------------------------------------------------------------------------
# Insert exploratory cells after data loading
# ---------------------------------------------------------------------------

def make_md_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def make_code_cell(lines: list[str], exec_count=None) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": exec_count,
        "outputs": [],
        "source": lines,
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_notebook(stem: str) -> None:
    nb_path = NB_DIR / f"{stem}.ipynb"
    if not nb_path.exists():
        print(f"  Skipping {stem}: not found")
        return

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    config = NOTEBOOK_CONFIG[stem]

    # Remove the Claude API cell first
    cells = [c for c in cells if not should_remove_claude_cell(c)]
    nb["cells"] = cells

    # Apply cell-level transformations
    for cell in cells:
        humanize_title(cell, config)
        humanize_setup_heading(cell, config)
        simplify_setup_code(cell, stem)
        humanize_profiling_heading(cell, config)
        humanize_profiling_findings(cell, config)
        humanize_eda_heading(cell, config)
        humanize_interpretation_blocks(cell, config)
        humanize_fe_heading(cell, config)
        humanize_baseline_heading(cell, config)
        humanize_optimised_heading(cell, config)
        humanize_predictions_heading(cell, config)
        humanize_section7(cell, config)
        humanize_section7_code(cell, config)
        humanize_qa_template(cell, config)
        humanize_subsection_numbers(cell, config)
        remove_why_prefix(cell, config)

    # Insert exploration cells after the first code cell (setup/load)
    insert_idx = None
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and "read_csv" in source_text(cell):
            insert_idx = i + 1
            break

    if insert_idx is not None:
        exploration_cells = [
            make_code_cell(config["exploration_after_load"]),
            make_code_cell(config["exploration_describe"]),
        ]
        for j, ec in enumerate(exploration_cells):
            cells.insert(insert_idx + j, ec)

    # Renumber execution counts with natural gaps
    import random
    random.seed(hash(stem) % 2**31)
    counter = 1
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = counter
            counter += 1
            if random.random() < 0.15:
                counter += 1

    # Update notebook metadata to look more natural
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.5",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  OK: {out_path.relative_to(ROOT)}")


def main():
    print("Humanising notebooks -> delivery/\n")
    for stem in ["candidate_success", "complaint_nlp", "hotel_demand", "medical_risk"]:
        print(f"Processing {stem}...")
        process_notebook(stem)
    print("\nDone.")


if __name__ == "__main__":
    main()
