"""
Baseline ML pipeline: EDA, feature engineering, models, test predictions.
Run from project root: python scripts/ml_baseline_pipeline.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import data_cleaning as dc

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "output"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {path}")


def print_df_summary(name: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print("Train shape:", train.shape, "| Test shape:", test.shape)
    print("\nTrain dtypes:\n", train.dtypes)
    print("\nTrain null counts:\n", train.isna().sum().sort_values(ascending=False))
    print("\nTest null counts:\n", test.isna().sum().sort_values(ascending=False))


def plot_target_distribution(y: pd.Series, title: str, fname: str) -> None:
    vc = y.value_counts().sort_index()
    pct = vc / vc.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(i) for i in vc.index], vc.values, color=["#4C78A8", "#E45756"][: len(vc)])
    ax.set_title(title)
    ax.set_ylabel("Count")
    for i, (c, p) in enumerate(zip(vc.values, pct.values)):
        ax.text(i, c + 15, f"{p:.1f}%", ha="center", fontsize=11)
    plt.tight_layout()
    save_fig(fig, fname)


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], fname: str) -> None:
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    save_fig(fig, fname)
    # multicollinearity
    high = []
    for i, a in enumerate(corr.columns):
        for c in corr.columns[i + 1 :]:
            v = abs(corr.loc[a, c])
            if v > 0.8:
                high.append((a, c, v))
    if high:
        print("  Pairs with |r| > 0.8 (multicollinearity risk):")
        for a, c, v in sorted(high, key=lambda x: -x[2]):
            print(f"    {a} vs {c}: {v:.3f}")
    else:
        print("  No feature pairs with |r| > 0.8")


def plot_numeric_hist_box(df: pd.DataFrame, cols: list[str], title: str, fname_prefix: str) -> None:
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for idx, col in enumerate(cols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        df[col].hist(ax=ax, bins=30, color="#72B7B2", edgecolor="white")
        ax.set_title(col)
    for idx in range(len(cols), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)
    fig.suptitle(title + " — histograms", fontsize=12)
    plt.tight_layout()
    save_fig(fig, f"{fname_prefix}_histograms.png")

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes2 = np.atleast_2d(axes2)
    for idx, col in enumerate(cols):
        r, c = divmod(idx, ncols)
        ax = axes2[r, c]
        df.boxplot(column=col, ax=ax)
        ax.set_title(col)
    for idx in range(len(cols), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes2[r, c].set_visible(False)
    fig2.suptitle(title + " — boxplots", fontsize=12)
    plt.tight_layout()
    save_fig(fig2, f"{fname_prefix}_boxplots.png")


def group_mean_vs_target(df: pd.DataFrame, numeric_cols: list[str], target: str) -> None:
    print("  Group means of numeric features by target:")
    for col in numeric_cols:
        if col == target:
            continue
        gm = df.groupby(target)[col].mean()
        print(f"    {col}: {gm.to_dict()}")


def evaluate_binary_binary(
    clf, X_val, y_val, name: str, model_tag: str
) -> tuple[float, float, float, np.ndarray]:
    y_pred = clf.predict(X_val)
    proba = clf.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_val, proba)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_val, y_pred)
    print(f"  [{model_tag}] acc={acc:.4f} f1_weighted={f1:.4f} roc_auc={auc:.4f}")
    return acc, f1, auc, cm


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    save_fig(fig, fname)


def plot_top_importances(importances: np.ndarray, names: np.ndarray, title: str, fname: str, top: int = 10) -> None:
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[order][::-1], importances[order][::-1], color="#4C78A8")
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, fname)


def train_binary_tabular(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    scale: bool = True,
) -> tuple[RandomForestClassifier, xgb.XGBClassifier, dict]:
    """80/20 stratified split; train RF + XGBoost with class imbalance handling."""
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    spw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    if scale:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_full = scaler.fit_transform(X)
    else:
        X_full = X.values

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    acc1, f11, auc1, cm1 = evaluate_binary_binary(rf, X_va, y_va, dataset_name, "RandomForest")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=spw,
        n_jobs=-1,
    )
    xgb_clf.fit(X_tr, y_tr)
    acc2, f2, auc2, cm2 = evaluate_binary_binary(xgb_clf, X_va, y_va, dataset_name, "XGBoost")

    feature_names = X.columns.values
    plot_confusion(cm1, ["0", "1"], f"{dataset_name} RF val", f"{dataset_name.lower()}_cm_rf.png")
    plot_confusion(cm2, ["0", "1"], f"{dataset_name} XGB val", f"{dataset_name.lower()}_cm_xgb.png")

    plot_top_importances(
        rf.feature_importances_,
        feature_names,
        f"{dataset_name} RF top features",
        f"{dataset_name.lower()}_importance_rf.png",
    )
    plot_top_importances(
        xgb_clf.feature_importances_,
        feature_names,
        f"{dataset_name} XGB top features",
        f"{dataset_name.lower()}_importance_xgb.png",
    )

    metrics = {
        "rf": {"acc": acc1, "f1": f11, "auc": auc1},
        "xgb": {"acc": acc2, "f1": f2, "auc": auc2},
    }
    return rf, xgb_clf, metrics


def fit_full_predict_binary(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    use_rf: bool,
    scale: bool = True,
) -> np.ndarray:
    """Retrain best model on full train; return test predictions."""
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
        X_te = scaler.transform(X_test)
    else:
        X_fit = X.values
        X_te = X_test.values
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    if use_rf:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=spw,
            n_jobs=-1,
        )
    model.fit(X_fit, y)
    return model.predict(X_te)


# ---------------------------------------------------------------------------
# 1. Candidate Success
# ---------------------------------------------------------------------------
def run_candidate() -> None:
    train = pd.read_csv(DATA_DIR / "candidate_success_train.csv")
    test = pd.read_csv(DATA_DIR / "candidate_success_test.csv")
    print_df_summary("Candidate Success", train, test)
    train, test, tlog = dc.impute_tabular_train_test(train, test, "success_label")
    print(" ", dc.format_tabular_cleaning_log(tlog))

    y = train["success_label"]
    plot_target_distribution(y, "Candidate train — success_label", "candidate_target_dist.png")

    num_cols = [c for c in train.columns if c not in ("id", "success_label")]
    plot_numeric_hist_box(train, num_cols, "Candidate", "candidate")
    correlation_heatmap(train, num_cols + ["success_label"], "candidate_corr.png")
    group_mean_vs_target(train, num_cols, "success_label")

    pos_rate = y.mean()
    print("\n  Class imbalance: positive rate = {:.1%}".format(pos_rate))
    print("  Moderate imbalance; using class_weight='balanced' / scale_pos_weight in baselines.")

    # Feature engineering
    tr = train.copy()
    te = test.copy()
    for d in (tr, te):
        d["composite_skill_score"] = (d["python_skill_score"] + d["ml_skill_score"]) / 2
        d["engagement_score"] = d["github_activity"] * np.log1p(d["projects_completed"])

    feat = [
        "experience_years",
        "python_skill_score",
        "ml_skill_score",
        "projects_completed",
        "education_level",
        "github_activity",
        "communication_score",
        "certifications",
        "composite_skill_score",
        "engagement_score",
    ]
    X = tr[feat]
    y = tr["success_label"]
    X_test = te[feat]

    rf, xgb_clf, metrics = train_binary_tabular(X, y, "Candidate", scale=True)
    use_rf = metrics["rf"]["f1"] >= metrics["xgb"]["f1"]
    print(f"\n  Using {'RandomForest' if use_rf else 'XGBoost'} for test predictions (higher val F1).")
    preds = fit_full_predict_binary(X, y, X_test, use_rf=use_rf, scale=True)
    out = pd.DataFrame({"id": test["id"], "predicted_label": preds.astype(int)})
    out.to_csv(OUTPUT_DIR / "candidate_success_predictions.csv", index=False)
    print(f"  Wrote {OUTPUT_DIR / 'candidate_success_predictions.csv'}")


# ---------------------------------------------------------------------------
# 2. Complaint NLP
# ---------------------------------------------------------------------------
def run_complaint() -> None:
    train = pd.read_csv(DATA_DIR / "complaint_nlp_train.csv")
    test = pd.read_csv(DATA_DIR / "complaint_nlp_test.csv")
    print_df_summary("Complaint NLP", train, test)
    train, test, clog = dc.clean_complaint_frames(train, test)
    print(" ", dc.format_complaint_cleaning_log(clog))

    y = train["category_label"]
    fig, ax = plt.subplots(figsize=(8, 4))
    vc = y.value_counts().sort_index()
    pct = vc / vc.sum() * 100
    ax.bar(vc.index.astype(str), vc.values, color=sns.color_palette("Set2", len(vc)))
    ax.set_title("Complaint train — category distribution")
    ax.set_ylabel("Count")
    for i, (c, p) in enumerate(zip(vc.values, pct.values)):
        ax.text(i, c + 15, f"{p:.1f}%", ha="center", fontsize=9)
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_fig(fig, "complaint_target_dist.png")

    train = train.copy()
    train["text_len"] = train["complaint_text"].astype(str).str.len()
    print("  Average text length (chars):", train["text_len"].mean().round(2))

    fig, ax = plt.subplots(figsize=(7, 4))
    train.groupby("category_label")["text_len"].mean().sort_index().plot(kind="bar", ax=ax, color="#72B7B2")
    ax.set_title("Mean text length by category")
    plt.tight_layout()
    save_fig(fig, "complaint_len_by_cat.png")

    # Word frequencies (train)
    vec = CountVectorizer(max_features=25)
    vec.fit(train["complaint_text"])
    freq = np.asarray(vec.transform(train["complaint_text"]).sum(axis=0)).ravel()
    words = vec.get_feature_names_out()
    order = np.argsort(freq)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(words[order][::-1], freq[order][::-1], color="#4C78A8")
    ax.set_title("Top word frequencies (CountVectorizer)")
    plt.tight_layout()
    save_fig(fig, "complaint_word_freq.png")

    print("\n  Sample texts per category:")
    for cat in sorted(train["category_label"].unique()):
        samples = train.loc[train["category_label"] == cat, "complaint_text"].head(2).tolist()
        print(f"    [{cat}]: {samples}")

    X_tr, X_va, y_tr, y_va = train_test_split(
        train["complaint_text"],
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred, average="weighted")
    print(f"\n  LogisticRegression + TF-IDF val: accuracy={acc:.4f} f1_weighted={f1:.4f}")
    print("\nClassification report (val):\n", classification_report(y_va, y_pred))

    # Multiclass ROC-AUC (ovr)
    try:
        proba = pipe.predict_proba(X_va)
        auc = roc_auc_score(
            pd.get_dummies(y_va).reindex(columns=pipe.classes_, fill_value=0).values,
            proba,
            average="weighted",
            multi_class="ovr",
        )
        print(f"  ROC-AUC (weighted OVR): {auc:.4f}")
    except Exception as e:
        print("  ROC-AUC skipped:", e)

    cm = confusion_matrix(y_va, y_pred, labels=pipe.classes_)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=pipe.classes_, yticklabels=pipe.classes_, ax=ax)
    ax.set_title("Complaint — confusion matrix (val)")
    plt.tight_layout()
    save_fig(fig, "complaint_cm.png")

    # Full train for test
    pipe.fit(train["complaint_text"], train["category_label"])
    test_preds = pipe.predict(test["complaint_text"])
    out = pd.DataFrame({"id": test["id"], "predicted_label": test_preds})
    out.to_csv(OUTPUT_DIR / "complaint_nlp_predictions.csv", index=False)
    print(f"  Wrote {OUTPUT_DIR / 'complaint_nlp_predictions.csv'}")


# ---------------------------------------------------------------------------
# 3. Hotel Demand
# ---------------------------------------------------------------------------
def run_hotel() -> None:
    train = pd.read_csv(DATA_DIR / "hotel_demand_train.csv")
    test = pd.read_csv(DATA_DIR / "hotel_demand_test.csv")
    print_df_summary("Hotel Demand", train, test)
    train, test, tlog = dc.impute_tabular_train_test(train, test, "demand_label")
    print(" ", dc.format_tabular_cleaning_log(tlog))

    y = train["demand_label"]
    plot_target_distribution(y, "Hotel train — demand_label", "hotel_target_dist.png")

    num_cols = [c for c in train.columns if c not in ("id", "demand_label")]
    plot_numeric_hist_box(train, num_cols, "Hotel", "hotel")
    correlation_heatmap(train, num_cols + ["demand_label"], "hotel_corr.png")
    group_mean_vs_target(train, num_cols, "demand_label")

    tr = train.copy()
    te = test.copy()
    for d in (tr, te):
        d["value_score"] = d["previous_bookings"] / (d["price_per_night"] + 1)
        d["family_flag"] = (d["num_children"] > 0).astype(int)

    feat = num_cols + ["value_score", "family_flag"]
    # num_cols already all numeric; family_flag is new
    feat = list(dict.fromkeys(feat))  # unique preserve order

    X = tr[feat]
    y = tr["demand_label"]
    X_test = te[feat]

    rf, xgb_clf, metrics = train_binary_tabular(X, y, "Hotel", scale=True)
    use_rf = metrics["rf"]["f1"] >= metrics["xgb"]["f1"]
    print(f"\n  Using {'RandomForest' if use_rf else 'XGBoost'} for test predictions.")
    preds = fit_full_predict_binary(X, y, X_test, use_rf=use_rf, scale=True)
    out = pd.DataFrame({"id": test["id"], "predicted_label": preds.astype(int)})
    out.to_csv(OUTPUT_DIR / "hotel_demand_predictions.csv", index=False)
    print(f"  Wrote {OUTPUT_DIR / 'hotel_demand_predictions.csv'}")


# ---------------------------------------------------------------------------
# 4. Medical Risk
# ---------------------------------------------------------------------------
def run_medical() -> None:
    train = pd.read_csv(DATA_DIR / "medical_risk_train.csv")
    test = pd.read_csv(DATA_DIR / "medical_risk_test.csv")
    print_df_summary("Medical Risk", train, test)
    train, test, tlog = dc.impute_tabular_train_test(train, test, "risk_label")
    print(" ", dc.format_tabular_cleaning_log(tlog))

    y = train["risk_label"]
    plot_target_distribution(y, "Medical train — risk_label", "medical_target_dist.png")

    num_cols = [c for c in train.columns if c not in ("id", "risk_label")]
    plot_numeric_hist_box(train, num_cols, "Medical", "medical")
    correlation_heatmap(train, num_cols + ["risk_label"], "medical_corr.png")
    group_mean_vs_target(train, num_cols, "risk_label")

    # Risk rate by smoker / family_history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["smoker", "family_history"]):
        r = train.groupby(col)["risk_label"].mean()
        ax.bar([str(i) for i in r.index], r.values, color=["#72B7B2", "#F58518"][: len(r)])
        ax.set_title(f"Mean risk_label by {col}")
        ax.set_ylabel("Rate")
    plt.tight_layout()
    save_fig(fig, "medical_risk_by_smoker_family.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        train["age"],
        train["bmi"],
        c=train["risk_label"],
        cmap="coolwarm",
        alpha=0.5,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="risk_label")
    ax.set_xlabel("age")
    ax.set_ylabel("bmi")
    ax.set_title("BMI vs age (colored by risk_label)")
    plt.tight_layout()
    save_fig(fig, "medical_bmi_age_scatter.png")

    tr = train.copy()
    te = test.copy()
    for d in (tr, te):
        d["metabolic_risk"] = d["bmi"] * d["glucose_level"] / 1000.0
        d["lifestyle_score"] = d["smoker"] * 2 + (4 - d["physical_activity"]) + d["stress_level"]

    feat = num_cols + ["metabolic_risk", "lifestyle_score"]
    feat = list(dict.fromkeys(feat))

    X = tr[feat]
    y = tr["risk_label"]
    X_test = te[feat]

    rf, xgb_clf, metrics = train_binary_tabular(X, y, "Medical", scale=True)
    use_rf = metrics["rf"]["f1"] >= metrics["xgb"]["f1"]
    print(f"\n  Using {'RandomForest' if use_rf else 'XGBoost'} for test predictions.")
    preds = fit_full_predict_binary(X, y, X_test, use_rf=use_rf, scale=True)
    out = pd.DataFrame({"id": test["id"], "predicted_label": preds.astype(int)})
    out.to_csv(OUTPUT_DIR / "medical_risk_predictions.csv", index=False)
    print(f"  Wrote {OUTPUT_DIR / 'medical_risk_predictions.csv'}")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    print("ML Baseline Pipeline — figures ->", FIGURES_DIR, "| predictions ->", OUTPUT_DIR)
    run_candidate()
    run_complaint()
    run_hotel()
    run_medical()
    print("\nDone. All prediction CSVs saved to output/.")


if __name__ == "__main__":
    main()
