"""
Run after executing all 4 notebooks:
  python final_answers.py
Reads answers_*.json and prints a formatted submission answer sheet.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
json_files = sorted(ROOT.glob("answers_*.json"))

if not json_files:
    print("No answer files found. Run all 4 notebooks first.")
    exit(1)

SECTION7_QUESTIONS = [
    ("FEATURE_ENGINEERING",  "Describe the feature engineering techniques applied."),
    ("TOP3_FEATURES",        "Which three features contributed most to model performance?"),
    ("BASELINE_MODEL",       "Which baseline model did you implement first?"),
    ("BASELINE_RATIONALE",   "Explain why you selected this baseline model."),
    ("TRAINING_PROCESS",     "Describe the training process (split, CV, tuning)."),
    ("EVAL_METRIC",          "What evaluation metric did you use?"),
    ("BASELINE_SCORE",       "What was the baseline model performance score?"),
    ("BEST_SCORE",           "What was the best model performance achieved?"),
    ("EXPERIMENTS",          "Describe the experiments conducted to improve the model."),
    ("IMPROVEMENT_REASON",   "Explain why the final model performed better."),
    ("DEPLOYMENT",           "How would you deploy this model into production?"),
    ("SUMMARY",              "Short technical summary of your overall approach."),
]

lines = []
for jf in json_files:
    ans = json.loads(jf.read_text(encoding="utf-8"))
    lines.append("=" * 70)
    lines.append(f"DATASET: {ans.get('dataset', jf.stem)}")
    lines.append("=" * 70)
    if "baseline_scores" in ans:
        bs = ans["baseline_scores"]
        lines.append(f"  Baseline scores: {bs}")
    if "best_scores" in ans:
        bs = ans["best_scores"]
        lines.append(f"  Best scores:     {bs}")
    for key, question in SECTION7_QUESTIONS:
        lines.append(f"\nQ: {question}")
        lines.append(f"A: {ans.get(key, '[missing]')}")
    lines.append("")

output = "\n".join(lines)
print(output)
(ROOT / "final_answers.txt").write_text(output, encoding="utf-8")
print(f"\nSaved: final_answers.txt")
