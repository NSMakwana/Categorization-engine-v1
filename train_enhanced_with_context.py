import sys, os, json, random, pickle
sys.path.append(r'D:\Nency\Mitex Intel System\Categorization-engine-v1')
import pandas as pd
import numpy as np
import re
from collections import Counter
from engine.normalizer import normalize_text

# Config
SALARY_FILE = r'D:\Nency\Mitex Intel System\Salary\3DYK042418_BANK_STMT_R1_1765469668172.xlsx'
OUTPUT_DIR = r'D:\Nency\Mitex Intel System\Categorization-engine-v1\models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("STEP 1: Load and merge contextual data from Excel file")
print("=" * 80)

# Load main transactions
df_xn = pd.read_excel(SALARY_FILE, sheet_name='Xns Transactions')
print(f"Main transactions loaded: {len(df_xn)} rows")
print(f"Columns: {df_xn.columns.tolist()}")

# Load salary transactions (ground truth for salary category)
df_salary = pd.read_excel(SALARY_FILE, sheet_name='SalaryXNS')
print(f"\nSalary transactions loaded: {len(df_salary)} rows")

# Load recurring transactions
df_recurring = pd.read_excel(SALARY_FILE, sheet_name='Reccuring XNS')
print(f"Recurring transactions loaded: {len(df_recurring)} rows")

# Create a set of salary narrations for labeling
salary_narrations = set(df_salary['Narration'].astype(str).str.upper().unique())
print(f"Unique salary narrations: {len(salary_narrations)}")

# Create a set of recurring narrations
recurring_narrations = set(df_recurring['Sample Narration'].astype(str).str.upper().unique())
print(f"Unique recurring narrations: {len(recurring_narrations)}")

print("\n" + "=" * 80)
print("STEP 2: Engineer contextual features")
print("=" * 80)

# Add feature: is_recurring (whether narration matches a known recurring pattern)
def is_recurring(narration):
    if pd.isna(narration):
        return False
    narr_upper = str(narration).upper()
    # Check if this narration or similar pattern is in recurring set
    for rec_narr in recurring_narrations:
        if rec_narr and rec_narr in narr_upper:
            return True
    return False

# Add feature: salary_probability_from_context
def salary_probability_from_context(row):
    narration = str(row.get('Narration', '')).upper()
    category = str(row.get('Category', '')).upper()
    
    # Heuristic: if narration is in salary set or category is SALARY, high probability
    if narration in salary_narrations or 'SALARY' in category:
        return 1.0
    
    # If it's a NEFT/IMPS credit with no debit and recurring, moderate probability
    if pd.isna(row.get('Debits')) or row.get('Debits') == 0:
        credits = row.get('Credits')
        if credits and credits > 5000 and is_recurring(narration):
            return 0.6
    
    return 0.0

df_xn['is_recurring'] = df_xn['Narration'].apply(is_recurring)
df_xn['salary_probability'] = df_xn.apply(salary_probability_from_context, axis=1)
df_xn['Normalized Narration'] = df_xn['Narration'].apply(normalize_text)

print("\nFeatures added:")
print(f"  is_recurring: {df_xn['is_recurring'].sum()} rows marked as recurring")
print(f"  salary_probability: mean={df_xn['salary_probability'].mean():.3f}, max={df_xn['salary_probability'].max():.3f}")

# Label data: use SalaryXNS as ground truth for SALARY labels
df_xn['Enhanced_Label'] = df_xn['Category']  # default to original category
df_xn.loc[df_xn['Narration'].astype(str).str.upper().isin(salary_narrations), 'Enhanced_Label'] = 'SALARY RECEIVED'

print(f"\nEnhanced label distribution:")
print(df_xn['Enhanced_Label'].value_counts())

print("\n" + "=" * 80)
print("STEP 3: Prepare training data with contextual features")
print("=" * 80)

# For training, we'll use both narration (TF-IDF) and categorical features
X_narration = df_xn['Normalized Narration'].astype(str).values
X_is_recurring = df_xn['is_recurring'].astype(int).values
X_salary_prob = df_xn['salary_probability'].astype(float).values
y = df_xn['Enhanced_Label'].astype(str).values

print(f"Training data: {len(X_narration)} samples")
print(f"Labels: {len(set(y))} unique classes")

# Train/test split (stratified by label)
from sklearn.model_selection import train_test_split

# For small datasets with rare classes, use non-stratified or simple random split
# If possible, try stratified; if it fails, use random split
try:
    indices = np.arange(len(X_narration))
    indices_train, indices_val = train_test_split(indices, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
except ValueError:
    # Fallback: non-stratified split if classes too small
    print("  (Stratified split failed due to small class sizes, using random split instead)")
    indices = np.arange(len(X_narration))
    indices_train, indices_val = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)

X_narration_train = X_narration[indices_train]
X_narration_val = X_narration[indices_val]
X_is_recurring_train = X_is_recurring[indices_train]
X_is_recurring_val = X_is_recurring[indices_val]
X_salary_prob_train = X_salary_prob[indices_train]
X_salary_prob_val = X_salary_prob[indices_val]
y_train = y[indices_train]
y_val = y[indices_val]

print(f"Train size: {len(y_train)}, Val size: {len(y_val)}")

print("\n" + "=" * 80)
print("STEP 4: Train ML model with contextual features")
print("=" * 80)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Step 4a: TF-IDF on narrations
print("Fitting TF-IDF vectorizer on training narrations...")
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_tfidf_train = tfidf.fit_transform(X_narration_train)
X_tfidf_val = tfidf.transform(X_narration_val)

print(f"TF-IDF shape: {X_tfidf_train.shape}")

# Step 4b: Combine TF-IDF + contextual features using hstack
print("Combining TF-IDF and contextual features...")
X_contextual_train = np.column_stack([X_is_recurring_train, X_salary_prob_train])
X_contextual_val = np.column_stack([X_is_recurring_val, X_salary_prob_val])

# Scale contextual features to similar magnitude as TF-IDF
scaler = StandardScaler()
X_contextual_train = scaler.fit_transform(X_contextual_train)
X_contextual_val = scaler.transform(X_contextual_val)

# Combine sparse (TF-IDF) and dense (contextual) features
from scipy.sparse import hstack
X_combined_train = hstack([X_tfidf_train, X_contextual_train])
X_combined_val = hstack([X_tfidf_val, X_contextual_val])

print(f"Combined feature matrix train shape: {X_combined_train.shape}")

# Step 4c: Train LogisticRegression with class weights for imbalanced data
print("Training LogisticRegression with class weight balancing...")
clf = LogisticRegression(max_iter=2000, class_weight='balanced')
clf.fit(X_combined_train, y_train)

print("Training complete.")

# Step 4d: Evaluate
print("\n" + "=" * 80)
print("STEP 5: Evaluate model on validation set")
print("=" * 80)

y_pred = clf.predict(X_combined_val)
acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)

print(f"\nValidation metrics (weighted):")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")

# Per-class metrics
labels_unique = sorted(list(set(y_val)))
p, r, f, s = precision_recall_fscore_support(y_val, y_pred, labels=labels_unique, zero_division=0)

print(f"\nPer-class metrics:")
for lab, pp, rr, ff, supp in zip(labels_unique, p, r, f, s):
    print(f"  {lab:30s}: P={pp:.3f}, R={rr:.3f}, F1={ff:.3f}, Support={int(supp)}")

# Check salary-specific metrics
salary_labels = [lab for lab in labels_unique if 'SALARY' in str(lab).upper()]
if salary_labels:
    print(f"\n** SALARY category focus **")
    for salary_lab in salary_labels:
        idx = list(labels_unique).index(salary_lab)
        print(f"  {salary_lab}: Precision={p[idx]:.3f}, Recall={r[idx]:.3f}, F1={f[idx]:.3f}, Support={int(s[idx])}")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=labels_unique)
print(f"\nConfusion matrix shape: {cm.shape}")

print("\n" + "=" * 80)
print("STEP 6: Save artifacts")
print("=" * 80)

# Pre-compute per-class metrics before saving (to avoid file closure issues)
per_class_metrics_list = [
    {
        'label': str(lab),
        'precision': float(pp),
        'recall': float(rr),
        'f1': float(ff),
        'support': int(supp)
    }
    for lab, pp, rr, ff, supp in zip(labels_unique, p, r, f, s)
]

# Save model components
model_artifact = {
    'tfidf': tfidf,
    'scaler': scaler,
    'clf': clf,
}
model_path = os.path.join(OUTPUT_DIR, 'enhanced_model_with_context.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_artifact, f)
print(f"Model saved: {model_path}")

# Save enhanced dataset
enhanced_csv = os.path.join(OUTPUT_DIR, 'salary_enhanced_dataset.csv')
df_xn_export = df_xn[['Narration', 'Normalized Narration', 'Category', 'Enhanced_Label', 'is_recurring', 'salary_probability', 'Debits', 'Credits']].copy()
df_xn_export.to_csv(enhanced_csv, index=False)
print(f"Enhanced dataset saved: {enhanced_csv}")

# Save evaluation metrics
eval_metrics = {
    'accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1': float(f1),
    'per_class_metrics': per_class_metrics_list
}
metrics_path = os.path.join(OUTPUT_DIR, 'enhanced_model_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(eval_metrics, f, indent=2)
print(f"Metrics saved: {metrics_path}")

# Save confusion matrix
cm_df_data = []
for i, true_lab in enumerate(labels_unique):
    for j, pred_lab in enumerate(labels_unique):
        cm_df_data.append({
            'true_label': true_lab,
            'predicted_label': pred_lab,
            'count': int(cm[i, j])
        })
cm_csv = os.path.join(OUTPUT_DIR, 'enhanced_confusion_matrix.csv')
pd.DataFrame(cm_df_data).to_csv(cm_csv, index=False)
print(f"Confusion matrix saved: {cm_csv}")

# Show mismatches
print("\n" + "=" * 80)
print("STEP 7: Top mismatches (true -> predicted)")
print("=" * 80)

mismatches = df_xn.iloc[indices_val][y_val != y_pred].copy()
mismatches['predicted'] = y_pred[y_val != y_pred]
print(f"Total mismatches: {len(mismatches)} / {len(y_val)}")

if not mismatches.empty:
    mismatch_counts = mismatches.groupby(['Enhanced_Label', 'predicted']).size().sort_values(ascending=False).head(10)
    print("\nTop 10 mismatch pairs (true -> predicted):")
    for (true, pred), count in mismatch_counts.items():
        print(f"  {true:30s} -> {pred:30s}: {count}")

    # Save mismatch examples
    mismatch_csv = os.path.join(OUTPUT_DIR, 'enhanced_mismatches.csv')
    mismatches_export = mismatches[['Narration', 'Enhanced_Label', 'predicted', 'is_recurring', 'salary_probability', 'Credits']].copy()
    mismatches_export.columns = ['Narration', 'True Label', 'Predicted Label', 'Is Recurring', 'Salary Probability', 'Credits']
    mismatches_export.to_csv(mismatch_csv, index=False)
    print(f"\nMismatch examples saved: {mismatch_csv}")

print("\n" + "=" * 80)
print("DONE: Enhanced model trained with contextual features")
print("=" * 80)
