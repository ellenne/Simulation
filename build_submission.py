"""
Run after all notebooks are executed:
  python build_submission.py
Packages notebooks, predictions, profiles, figures, and source into submission.zip
"""
import datetime
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
zip_path = ROOT / f"submission_{stamp}.zip"

include_patterns = [
    "notebooks/*.ipynb",
    "DataExploration.ipynb",
    "scripts/ml_baseline_pipeline.py",
    "scripts/data_cleaning.py",
    "scripts/generate_notebooks.py",
    "final_answers.py",
    "final_answers.txt",
    "output/*_predictions.csv",
    "figures/*.png",
    "profiles/profile_*.html",
    "answers_*.json",
    "requirements.txt",
    "README.md",
]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for pattern in include_patterns:
        for f in sorted(ROOT.glob(pattern)):
            if f.is_file():
                zf.write(f, f.relative_to(ROOT))
                print(f"  Added: {f.relative_to(ROOT)}")

print(f"\nSubmission ZIP created: {zip_path}")
print(f"Size: {zip_path.stat().st_size / 1024:.1f} KB")
