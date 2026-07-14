# Enhanced ML Model: Context-Aware Features Guide

## Overview

The **Enhanced ML Model** improves upon the baseline TF-IDF + LogisticRegression model by adding contextual features specifically designed to capture semantic signals beyond raw transaction narration text.

**Performance Impact:**
- **Salary Category**: Precision = 1.0, Recall = 1.0, F1 = 1.0 (100% accurate on salary transactions)
- **Overall Validation**: F1 = 0.8415 on salary-enriched dataset (162 transactions)

---

## Architecture

### Baseline Model (TF-IDF Only)
```
Narration
    ↓
Normalize
    ↓
TF-IDF Vectorization (1-2 grams, min_df=2)
    ↓
LogisticRegression (class_weight='balanced')
    ↓
Category Prediction
```

### Enhanced Model (TF-IDF + Context)
```
Narration + Metadata
    ↓
Normalize & TF-IDF Vectorization
    ↓
Contextual Feature Engineering
    ├── is_recurring (Boolean)
    └── salary_probability (Float 0-1)
    ↓
Combine Features (scipy.sparse.hstack)
    ├── TF-IDF sparse features
    └── Scaled contextual dense features
    ↓
LogisticRegression (class_weight='balanced')
    ↓
Enhanced Category Prediction
```

---

## Contextual Features

### 1. `is_recurring` (Boolean: 0 or 1)

**Semantic Meaning:**
- Indicates whether a transaction is part of a known **recurring pattern** (salary, subscription, auto-debit, etc.)
- Transactions with the same entity appearing multiple times in the dataset are flagged as recurring

**How it's computed:**
```python
RECURRING_NARRATIONS = {
    'SALARY', 'EMI', 'SUBSCRIPTION', 'AIRTEL', 'FLIPKART',
    'AMAZON', 'NETFLIX', 'SPOTIFY', 'INSURANCE', 'ELECTRICITY',
    # ... etc
}

is_recurring = 1 if any(token in normalized_narration for token in RECURRING_NARRATIONS) else 0
```

**Rationale:**
- Recurring transactions are more predictable and have distinct patterns
- Salary receipts, for example, typically occur monthly and from known entities (employer, payroll processor)
- Signal helps distinguish salary deposits from one-time transfers or refunds

**Typical Values:**
- Salary transactions: mostly `1` (recurring)
- E-commerce refunds: mostly `0` (non-recurring)
- Insurance payments: mostly `1` (recurring)

---

### 2. `salary_probability` (Float 0.0 - 1.0)

**Semantic Meaning:**
- Probabilistic heuristic scoring the likelihood that a transaction is a **salary deposit**
- Combines multiple signals: narration patterns, transaction amount, recurrence

**How it's computed:**

```python
def compute_salary_probability(row):
    narration = row['Normalized Narration']
    amount = row.get('Amount', 0)
    is_recurring = row.get('is_recurring', 0)
    
    # Signal 1: Direct salary keyword match
    if 'SALARY' in narration:
        return 1.0
    
    # Signal 2: Large credit + recurring (typical salary pattern)
    if amount >= 5000 and is_recurring:
        return 0.6  # Tentative score
    
    # Signal 3: No salary signal
    return 0.0
```

**Thresholds (Tunable):**
- `1.0` - Direct match with salary narrations ("SALARY DEPOSIT", "SALARY RECEIVED", etc.)
- `0.6` - Large credit (≥ ₹5,000) AND marked as recurring (strong heuristic for likely salary)
- `0.0` - No salary signals detected

**Why These Thresholds?**
- Salary deposits are typically large (₹5,000+) in India's banking context
- Recurring deposits from known recurring patterns (employer entities) are strong indicators
- Narration patterns alone may be insufficient; amount + recurrence combination is semantic

**Rationale:**
- Captures domain knowledge about typical salary characteristics
- Allows model to learn salary-specific decision boundaries
- Reduces false negatives on salary transactions with non-standard narrations

**Typical Values:**
- Salary transactions: mostly `1.0` or `0.6`
- E-commerce: mostly `0.0`
- Refunds: mostly `0.0`
- Large transfers from recurring sources: may be `0.6`

---

## Feature Engineering Process

### Step 1: Load and Augment Data

```python
from train_enhanced_with_context import prepare_enhanced_dataset

df = pd.read_csv('data/your_salary_enriched_data.csv')

# The script adds is_recurring and salary_probability columns
df = prepare_enhanced_dataset(df)
```

### Step 2: Train Enhanced Model

```bash
python train_enhanced_with_context.py
```

This creates:
- `models/enhanced_model_with_context.pkl` - Full model artifact (TF-IDF + scaler + classifier)
- `models/salary_enhanced_dataset.csv` - Dataset with engineered features
- `models/enhanced_model_metrics.json` - Per-class evaluation metrics
- `models/enhanced_confusion_matrix.csv` - Confusion matrix

### Step 3: Use in Production

**Option A: Streamlit UI (Recommended)**

1. Open `app.py` in Streamlit
2. Go to "ML Evaluation" tab
3. Select "Enhanced (TF-IDF + Context: salary, recurring)" from radio button
4. Upload your CSV with `Normalized Narration` column
5. Click "Run ML Evaluation"

**Option B: Direct Python**

```python
import pickle
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline

# Load enhanced model artifact
with open('models/enhanced_model_with_context.pkl', 'rb') as f:
    model_artifact = pickle.load(f)

tfidf = model_artifact['tfidf']
scaler = model_artifact['scaler']
clf = model_artifact['clf']

# Prepare your data
X_text = df['Normalized Narration'].to_list()  # List of strings
X_tfidf = tfidf.transform(X_text)

# Add contextual features
X_context = np.column_stack([
    df['is_recurring'].astype(int).values,
    df['salary_probability'].astype(float).values
])
X_context = scaler.transform(X_context)

# Combine and predict
X_combined = hstack([X_tfidf, X_context])
predictions = clf.predict(X_combined)
probabilities = clf.predict_proba(X_combined)
```

---

## Training Data Requirements

The enhanced model was trained on 162 transactions with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Narration` | str | Raw transaction narration |
| `Normalized Narration` | str | Cleaned/normalized narration |
| `Amount` | float | Transaction amount (for salary_probability computation) |
| `Old Category` | str | Ground truth category label |
| `is_recurring` | int | 1 if recurring, 0 otherwise |
| `salary_probability` | float | Heuristic probability (0.0 - 1.0) |

**Typical Example Row:**
```
Narration: "SALARY DEPOSIT TO ACC 45678 EMPLOYER BANK"
Normalized Narration: "SALARY/DEPOSIT/ACC/45678/EMPLOYER/BANK"
Amount: 50000
Old Category: "SALARY RECEIVED"
is_recurring: 1
salary_probability: 1.0
```

---

## Hyperparameters

### TF-IDF Settings (Inherited from Baseline)

```python
TfidfVectorizer(
    ngram_range=(1, 2),           # Unigrams + bigrams
    min_df=2,                      # Ignore terms appearing in < 2 docs
    max_features=None,             # No limit on vocabulary
    lowercase=False,               # Already normalized
    token_pattern=r'\b\w+\b'      # Word token pattern
)
```

### Contextual Feature Scaling

```python
StandardScaler()  # Mean=0, StdDev=1 scaling applied to context features

# Prevents TF-IDF dominance (sparse high-dim) over context (dense low-dim)
```

### Classifier Settings

```python
LogisticRegression(
    class_weight='balanced',       # Handles imbalanced classes
    max_iter=1000,                 # Convergence iterations
    solver='lbfgs',                # Optimization algorithm
    random_state=42                # Reproducibility
)
```

---

## Performance Analysis

### Overall Metrics (Salary-Enriched Dataset)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7879 |
| Precision (weighted) | 0.9040 |
| Recall (weighted) | 0.7879 |
| F1 (weighted) | 0.8415 |

### Per-Class Performance (Top 5)

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|----|----|
| **SALARY RECEIVED** | **1.00** | **1.00** | **1.00** | 14 |
| TRANSFER IN | 0.93 | 0.93 | 0.93 | 14 |
| TRANSFER OUT | 0.88 | 1.00 | 0.93 | 7 |
| CASH WITHDRAWAL | 1.00 | 0.67 | 0.80 | 3 |
| E-COMMERCE | 0.80 | 1.00 | 0.89 | 4 |

**Key Observation:** The enhanced model achieves **perfect precision and recall on SALARY RECEIVED transactions**, validating that contextual features effectively capture salary semantics.

---

## Limitations & Future Improvements

### Current Limitations

1. **Training Data Size**: Only 162 transactions from one salary-enriched Excel file
   - May not generalize to full production dataset (15,000+ transactions)
   - Recommend retraining on full dataset with engineered contextual features

2. **Heuristic-Based Features**: Not learned end-to-end
   - Salary probability thresholds (₹5,000, 0.6 score) are manually tuned
   - Could benefit from domain expertise validation or regression-based learning

3. **Sparse Context Coverage**: Only 2 contextual features
   - Could expand to include: transaction frequency, day-of-week patterns, account type, merchant category, etc.

4. **Limited Feature Interactions**: Context features not combined with TF-IDF (sparse hstack)
   - Could explore learned feature combinations via neural networks

### Recommended Improvements

#### 1. Retraining on Full Dataset

```bash
# Augment full dataset with context features
python train_enhanced_with_context.py --input data/cleaned_transactions.csv --output models/enhanced_full_dataset.csv

# Retrain
python train_enhanced_with_context.py --use_full_dataset
```

#### 2. Tune Salary Probability Thresholds

Edit `train_enhanced_with_context.py`:

```python
# Adjust based on domain analysis
SALARY_AMOUNT_THRESHOLD = 5000  # Currently ₹5,000
RECURRING_SALARY_SCORE = 0.6    # Currently 0.6
```

Recommend analyzing distribution of salary transactions in your dataset to optimize thresholds.

#### 3. Expand Context Features

Potential new signals:

```python
# Frequency-based (days since last similar transaction)
transaction_frequency = compute_frequency_score(entity, timeframe='30d')

# Temporal patterns (day of month, day of week)
day_of_month = pd.Timestamp(transaction_date).day
is_month_start = day_of_month <= 5

# Account type (from customer info sheet)
account_type = row['Account_Type']  # 'Savings', 'Current', etc.

# Merchant category
merchant_category = row['Merchant_Category']  # 'Bank', 'E-commerce', etc.
```

#### 4. Monitor and Maintain

```python
# Run comparison periodically to detect model drift
python compare_models.py

# If enhanced model performance degrades:
# 1. Collect new labeled data
# 2. Retrain with fresh dataset
# 3. Validate against holdout test set
```

---

## Troubleshooting

### Issue: "is_recurring column not found"

**Cause:** Your CSV doesn't have the pre-engineered context features.

**Solution:**
1. Run `train_enhanced_with_context.py` to generate context features on your dataset
2. Or manually add columns:
   ```python
   df['is_recurring'] = 0  # Default: not recurring
   df['salary_probability'] = 0.0  # Default: no salary signal
   ```

### Issue: Enhanced model performs worse than baseline

**Cause:** Context features may not transfer well to your dataset if it differs significantly from the training data (salary-enriched file).

**Solution:**
1. Check `compare_models.py` output for per-class regressions
2. Verify context features align with your data (check `salary_enhanced_dataset.csv`)
3. Retrain on your full dataset with properly computed context features

### Issue: Model file loading errors

**Cause:** Artifact dict structure mismatch.

**Solution:**
Ensure enhanced model was created with:
```bash
python train_enhanced_with_context.py
```

This generates the correct `{'tfidf': ..., 'scaler': ..., 'clf': ...}` structure.

---

## Model Comparison: Basic vs Enhanced

Run the comparison script to see side-by-side performance:

```bash
python compare_models.py
```

**Output Example:**

```
OVERALL METRICS COMPARISON
================================================================================
Metric               Basic                Enhanced             Difference
Accuracy             0.7273               0.7879              +0.0606
Precision            0.8500               0.9040              +0.0540
Recall               0.7273               0.7879              +0.0606
F1 (weighted)        0.7826               0.8415              +0.0589

PER-CLASS COMPARISON
Salary Enhancement:
  Basic_F1:    0.67
  Enhanced_F1: 1.00
  F1_Δ:        +0.33
```

---

## Model Switching in Production

### Streamlit UI (Recommended)

1. **ML Evaluation Tab** → Radio button: "Enhanced (TF-IDF + Context: salary, recurring)"
2. Choose decision mode: ML-only / Rule-only / Hybrid
3. Click "Run ML Evaluation"

### Backend Code

```python
# Detect which model to use
if use_enhanced_model:
    from models.enhanced_model_with_context import enhanced_predict
    category = enhanced_predict(narration, amount, is_recurring)
else:
    from models.tfidf_logreg_pipeline import basic_predict
    category = basic_predict(narration)
```

---

## Reference

**Related Files:**
- `train_enhanced_with_context.py` - Training script for enhanced model
- `compare_models.py` - Comparison script (Basic vs Enhanced)
- `DOCUMENTATION_DETAILED.md` - Complete technical reference
- `app.py` - Streamlit UI with model selector (ML Evaluation tab)

**Artifacts:**
- `models/enhanced_model_with_context.pkl` - Trained model
- `models/salary_enhanced_dataset.csv` - Training dataset with features
- `models/enhanced_model_metrics.json` - Performance metrics
- `models/comparison_results/` - Comparison output

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Authors:** Financial Intelligence Engine Team
