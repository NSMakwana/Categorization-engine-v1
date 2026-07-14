#!/usr/bin/env python3
"""
Side-by-side comparison of baseline ML model vs enhanced context-aware model.

This script loads both models and evaluates them on the cleaned_transactions.csv dataset,
comparing their performance across all categories with detailed per-class metrics.

Models:
- Basic: TF-IDF + LogisticRegression (narration-only)
- Enhanced: TF-IDF + Context Features (narration + is_recurring + salary_probability)

Output:
- Comparison metrics in console
- CSV files with per-class comparison (comparison_per_class.csv)
- Confusion matrices for both models
- Mismatches analysis showing where they differ
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from scipy.sparse import hstack

# Config
DATA_FILE = Path("models/salary_enhanced_dataset.csv")
BASIC_MODEL_FILE = Path("models/tfidf_logreg_pipeline.pkl")
ENHANCED_MODEL_FILE = Path("models/enhanced_model_with_context.pkl")
OUTPUT_DIR = Path("models/comparison_results")

OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    
    # Try multiple possible paths
    possible_files = [
        Path("models/salary_enhanced_dataset.csv"),
        Path("data/cleaned_transactions.csv"),
    ]
    
    data_file = None
    for f in possible_files:
        if f.exists():
            data_file = f
            break
    
    if data_file is None:
        print(f"ERROR: Could not find data file. Tried: {possible_files}")
        return None

    print(f"Loading from {data_file}...")
    df = pd.read_csv(data_file)

    if "Normalized Narration" not in df.columns:
        print("ERROR: 'Normalized Narration' column not found")
        return None

    # Map various possible label column names
    if "Enhanced_Label" in df.columns and "Old Category" not in df.columns:
        df["Old Category"] = df["Enhanced_Label"]
    elif "Category" in df.columns and "Old Category" not in df.columns:
        df["Old Category"] = df["Category"]
    
    if "Old Category" not in df.columns:
        print("WARNING: 'Old Category' (labels) column not found. Evaluation will be limited.")
        df["Old Category"] = None

    return df


def predict_basic_model(df, model_path):
    """Load basic model and make predictions."""
    print(f"\nLoading basic model from {model_path}...")
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return None, None

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    texts = df["Normalized Narration"].astype(str).to_list()
    preds = pipeline.predict(texts)

    try:
        prob_arr = pipeline.predict_proba(texts)
        max_probs = prob_arr.max(axis=1)
    except Exception:
        max_probs = [None] * len(preds)

    print(f"Basic model predictions: {len(preds)} rows")
    return preds, max_probs


def predict_enhanced_model(df, model_path):
    """Load enhanced model with context features and make predictions."""
    print(f"\nLoading enhanced model from {model_path}...")
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return None, None

    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)

    try:
        tfidf = model_artifact.get("tfidf")
        scaler = model_artifact.get("scaler")
        clf = model_artifact.get("clf")

        if not all([tfidf, scaler, clf]):
            print("ERROR: Invalid enhanced model artifact structure")
            return None, None

        texts = df["Normalized Narration"].astype(str).to_list()

        # Transform with TF-IDF
        X_tfidf = tfidf.transform(texts)

        # Add contextual features
        is_recurring_arr = (
            df.get("is_recurring", pd.Series([0] * len(texts)))
            .astype(int)
            .values
        )
        salary_prob_arr = (
            df.get("salary_probability", pd.Series([0.0] * len(texts)))
            .astype(float)
            .values
        )

        if "is_recurring" not in df.columns:
            print(
                "WARNING: is_recurring column not found; using zeros for context features"
            )
        if "salary_probability" not in df.columns:
            print(
                "WARNING: salary_probability column not found; using zeros for context features"
            )

        X_contextual = np.column_stack([is_recurring_arr, salary_prob_arr])
        X_contextual = scaler.transform(X_contextual)

        X_combined = hstack([X_tfidf, X_contextual])

        preds = clf.predict(X_combined)

        try:
            prob_arr = clf.predict_proba(X_combined)
            max_probs = prob_arr.max(axis=1)
        except Exception:
            max_probs = [None] * len(preds)

        print(f"Enhanced model predictions: {len(preds)} rows")
        return preds, max_probs

    except Exception as e:
        print(f"ERROR loading enhanced model: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def compute_metrics(y_true, y_pred, model_name):
    """Compute and return metrics dict."""
    labels = sorted(list(set(y_true)))

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    per_class_metrics = []
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for lab, pp, rr, ff, supp_count in zip(labels, p, r, f, s):
        per_class_metrics.append(
            {
                "label": lab,
                "precision": float(pp),
                "recall": float(rr),
                "f1": float(ff),
                "support": int(supp_count),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "labels": labels,
    }


def print_comparison_summary(metrics_basic, metrics_enhanced):
    """Print side-by-side comparison."""
    print("\n" + "=" * 80)
    print("OVERALL METRICS COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Basic':<20} {'Enhanced':<20} {'Diff':<15}")
    print("-" * 75)

    acc_diff = metrics_enhanced["accuracy"] - metrics_basic["accuracy"]
    prec_diff = metrics_enhanced["precision"] - metrics_basic["precision"]
    rec_diff = metrics_enhanced["recall"] - metrics_basic["recall"]
    f1_diff = metrics_enhanced["f1"] - metrics_basic["f1"]

    print(f"{'Accuracy':<20} {metrics_basic['accuracy']:<20.4f} {metrics_enhanced['accuracy']:<20.4f} {acc_diff:+.4f}")
    print(
        f"{'Precision':<20} {metrics_basic['precision']:<20.4f} {metrics_enhanced['precision']:<20.4f} {prec_diff:+.4f}"
    )
    print(
        f"{'Recall':<20} {metrics_basic['recall']:<20.4f} {metrics_enhanced['recall']:<20.4f} {rec_diff:+.4f}"
    )
    print(
        f"{'F1 (weighted)':<20} {metrics_basic['f1']:<20.4f} {metrics_enhanced['f1']:<20.4f} {f1_diff:+.4f}"
    )

    print("\n" + "=" * 80)
    print("PER-CLASS COMPARISON")
    print("=" * 80)

    # Build comparison table
    comparison_data = []
    for basic_item, enhanced_item in zip(
        metrics_basic["per_class"], metrics_enhanced["per_class"]
    ):
        if basic_item["label"] == enhanced_item["label"]:
            label = basic_item["label"]
            comparison_data.append(
                {
                    "Category": label,
                    "Basic_Precision": basic_item["precision"],
                    "Enhanced_Precision": enhanced_item["precision"],
                    "Prec_Delta": enhanced_item["precision"] - basic_item["precision"],
                    "Basic_Recall": basic_item["recall"],
                    "Enhanced_Recall": enhanced_item["recall"],
                    "Rec_Delta": enhanced_item["recall"] - basic_item["recall"],
                    "Basic_F1": basic_item["f1"],
                    "Enhanced_F1": enhanced_item["f1"],
                    "F1_Delta": enhanced_item["f1"] - basic_item["f1"],
                    "Support": basic_item["support"],
                }
            )

    comp_df = pd.DataFrame(comparison_data)
    comp_df = comp_df.sort_values("F1_Delta", ascending=False)

    print("\n" + comp_df.to_string(index=False))

    print("\n\nTop 5 Category Improvements (by F1):")
    print(comp_df.head(5)[["Category", "Basic_F1", "Enhanced_F1", "F1_Delta", "Support"]].to_string(index=False))

    print("\n\nTop 5 Category Regressions (by F1):")
    regressions = comp_df.tail(5)[["Category", "Basic_F1", "Enhanced_F1", "F1_Delta", "Support"]]
    if len(regressions) > 0:
        print(regressions.to_string(index=False))
    else:
        print("(None - all categories improved or stayed the same)")

    # Save to CSV
    csv_path = OUTPUT_DIR / "comparison_per_class.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Per-class comparison saved to {csv_path}")

    return comp_df


def analyze_mismatches(df, basic_preds, enhanced_preds):
    """Analyze where the two models disagree."""
    print("\n" + "=" * 80)
    print("MISMATCH ANALYSIS")
    print("=" * 80)

    df_comp = df.copy()
    df_comp["Basic_Pred"] = basic_preds
    df_comp["Enhanced_Pred"] = enhanced_preds
    df_comp["Agree"] = df_comp["Basic_Pred"] == df_comp["Enhanced_Pred"]

    mismatch_count = (~df_comp["Agree"]).sum()
    agree_count = df_comp["Agree"].sum()

    print(f"\nTotal predictions: {len(df_comp)}")
    print(f"Models AGREE: {agree_count} ({100*agree_count/len(df_comp):.2f}%)")
    print(f"Models DISAGREE: {mismatch_count} ({100*mismatch_count/len(df_comp):.2f}%)")

    # Show some mismatch examples
    mismatches = df_comp[~df_comp["Agree"]].head(20)
    if not mismatches.empty:
        print("\nSample mismatches (first 20):")
        print("-" * 80)
        for idx, row in mismatches.iterrows():
            print(f"\nNarration: {row['Narration'][:80]}")
            print(f"  True Label: {row.get('Old Category', 'N/A')}")
            print(f"  Basic Model: {row['Basic_Pred']}")
            print(f"  Enhanced Model: {row['Enhanced_Pred']}")

    # Save mismatches
    mismatch_path = OUTPUT_DIR / "mismatches.csv"
    mismatches.to_csv(mismatch_path, index=False)
    print(f"\n[OK] Mismatches saved to {mismatch_path}")


def main():
    """Main comparison workflow."""
    print("\n" + "=" * 80)
    print("ML MODEL COMPARISON: BASIC vs ENHANCED")
    print("=" * 80)

    # Load data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Loaded {len(df)} transactions")

    # Get predictions from both models
    basic_preds, basic_probs = predict_basic_model(df, BASIC_MODEL_FILE)
    if basic_preds is None:
        print("Failed to get basic model predictions. Exiting.")
        return

    enhanced_preds, enhanced_probs = predict_enhanced_model(df, ENHANCED_MODEL_FILE)
    if enhanced_preds is None:
        print("Failed to get enhanced model predictions. Exiting.")
        return

    # Check for labels
    if "Old Category" not in df.columns or df["Old Category"].isna().all():
        print("WARNING: No ground truth labels found. Skipping evaluation.")
        print("Make sure your CSV has an 'Old Category' column for comparison.")
        return

    y_true = df["Old Category"].astype(str).to_list()

    # Compute metrics
    print("\nComputing metrics...")
    metrics_basic = compute_metrics(y_true, basic_preds, "Basic")
    metrics_enhanced = compute_metrics(y_true, enhanced_preds, "Enhanced")

    # Print comparison
    comp_df = print_comparison_summary(metrics_basic, metrics_enhanced)

    # Analyze mismatches
    analyze_mismatches(df, basic_preds, enhanced_preds)

    # Save confusion matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)

    basic_cm_df = pd.DataFrame(
        metrics_basic["confusion_matrix"],
        index=metrics_basic["labels"],
        columns=metrics_basic["labels"],
    )
    basic_cm_path = OUTPUT_DIR / "confusion_matrix_basic.csv"
    basic_cm_df.to_csv(basic_cm_path)
    print(f"[OK] Basic model confusion matrix saved to {basic_cm_path}")

    enhanced_cm_df = pd.DataFrame(
        metrics_enhanced["confusion_matrix"],
        index=metrics_enhanced["labels"],
        columns=metrics_enhanced["labels"],
    )
    enhanced_cm_path = OUTPUT_DIR / "confusion_matrix_enhanced.csv"
    enhanced_cm_df.to_csv(enhanced_cm_path)
    print(f"[OK] Enhanced model confusion matrix saved to {enhanced_cm_path}")

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nComparison results saved to: {OUTPUT_DIR}")
    print(f"  - Per-class comparison: comparison_per_class.csv")
    print(f"  - Confusion matrices: confusion_matrix_basic.csv, confusion_matrix_enhanced.csv")
    print(f"  - Mismatches: mismatches.csv")
    print("\n[OK] Comparison complete!")


if __name__ == "__main__":
    main()
