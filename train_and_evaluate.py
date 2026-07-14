import sys, os, json, random, pickle
sys.path.append(r'D:\Nency\Mitex Intel System\Categorization-engine-v1')
import pandas as pd
import numpy as np
import re
from collections import Counter
from engine.normalizer import normalize_text

# Config
CSV_PATH = r'D:\Nency\Mitex Intel System\Transaction Generator\output\cleaned_transactions.csv'
OUTPUT_DIR = r'D:\Nency\Mitex Intel System\Categorization-engine-v1\models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SYNTHETIC_PER_CLASS_CAP = 2000
RANDOM_SEED = 42
TARGET_COL = None  # auto-detect
NROWS_SAMPLE = None  # set to int for quicker runs, else None

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def detect_label_column(df):
    if 'Old Category' in df.columns:
        return 'Old Category'
    if 'Category' in df.columns and 'Old Category' not in df.columns:
        return 'Category'
    # try common alternatives
    for candidate in ['label', 'Label', 'CATEGORY']:
        if candidate in df.columns:
            return candidate
    raise ValueError('Could not detect label column. Provide a CSV with "Old Category" or "Category" column.')


def augment_text(text):
    # lightweight augmentations: token deletion, swap, punctuation noise, separator changes
    t = normalize_text(text)
    tokens = re.split(r"([\s/_\-:.@,]+)", t)
    # tokens list includes separators; operate on token positions that are words
    word_indices = [i for i in range(0, len(tokens), 2) if tokens[i].strip()]

    # random deletion
    if len(word_indices) > 1 and random.random() < 0.12:
        idx = random.choice(word_indices)
        tokens[idx] = ''

    # swap adjacent
    if len(word_indices) > 2 and random.random() < 0.08:
        i = random.choice(range(len(word_indices)-1))
        a = word_indices[i]
        b = word_indices[i+1]
        tokens[a], tokens[b] = tokens[b], tokens[a]

    # alter separators
    for i in range(1, len(tokens), 2):
        if tokens[i] and random.random() < 0.15:
            tokens[i] = random.choice(['/', '-', ' ', ' / '])

    # append small numeric noise sometimes
    if random.random() < 0.06:
        tokens.append(' / ')
        tokens.append(str(random.randint(100, 99999)))

    augmented = ''.join(tokens)
    augmented = re.sub(r"\s+", ' ', augmented).strip()
    return augmented


def generate_synthetic_for_class(samples, target_count):
    out = list(samples)
    i = 0
    # simple loop: augment randomly selected samples until target_count
    while len(out) < target_count:
        src = random.choice(samples)
        aug = augment_text(src)
        if aug and aug not in out:
            out.append(aug)
        i += 1
        if i > target_count * 10:
            # fallback: allow duplicates if stuck
            out.append(random.choice(samples))
    return out


if __name__ == '__main__':
    print('Loading:', CSV_PATH)
    df = pd.read_csv(CSV_PATH, nrows=NROWS_SAMPLE)
    print('Rows loaded:', len(df))

    label_col = detect_label_column(df)
    print('Label column:', label_col)

    # Ensure narration column normalized
    if 'Normalized Narration' not in df.columns:
        print('Normalizing Narration column...')
        if 'Narration' not in df.columns:
            raise ValueError('CSV must contain a Narration column')
        df['Normalized Narration'] = df['Narration'].apply(normalize_text)

    texts = df['Normalized Narration'].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    counter = Counter(labels)
    print('Class distribution before:', counter)
    max_count = max(counter.values())
    target_per_class = min(max_count, SYNTHETIC_PER_CLASS_CAP)
    print('Target per class:', target_per_class)

    # group texts by label
    by_label = {}
    for t,l in zip(texts, labels):
        by_label.setdefault(l, []).append(t)

    balanced_texts = []
    balanced_labels = []

    for lab, samples in by_label.items():
        if len(samples) >= target_per_class:
            chosen = random.sample(samples, target_per_class)
        else:
            chosen = generate_synthetic_for_class(samples, target_per_class)
        balanced_texts.extend(chosen)
        balanced_labels.extend([lab] * len(chosen))

    print('Balanced dataset size:', len(balanced_texts))

    out_df = pd.DataFrame({'Narration': balanced_texts, 'Label': balanced_labels})
    balanced_csv = os.path.join(OUTPUT_DIR, 'train_balanced.csv')
    out_df.to_csv(balanced_csv, index=False)
    print('Saved balanced CSV to', balanced_csv)

    # train/test split
    from sklearn.model_selection import train_test_split
    # Convert to plain numpy arrays to avoid pyarrow-backed indexing issues
    X = out_df['Narration'].astype(str).to_numpy()
    y = out_df['Label'].astype(str).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    print('Train/val sizes:', len(X_train), len(X_val))

    # TF-IDF + LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    clf = LogisticRegression(max_iter=2000)

    pipeline = Pipeline([('tfidf', vec), ('clf', clf)])

    print('Training classifier...')
    pipeline.fit(X_train, y_train)

    print('Evaluating...')
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)
    print(json.dumps({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}))

    # save model and vectorizer
    model_path = os.path.join(OUTPUT_DIR, 'tfidf_logreg_pipeline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print('Saved model to', model_path)

    # confusion and top errors
    labels_unique = sorted(list(set(y_val) | set(y_pred)))
    cm = confusion_matrix(y_val, y_pred, labels=labels_unique)
    cm_summary = {'labels': labels_unique, 'matrix': cm.tolist()}
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.json')
    with open(cm_path, 'w') as f:
        json.dump(cm_summary, f)
    print('Saved confusion matrix to', cm_path)

    # top confusion pairs
    cm_arr = np.array(cm)
    errors = []
    for i, true_label in enumerate(labels_unique):
        for j, pred_label in enumerate(labels_unique):
            if i != j and cm_arr[i, j] > 0:
                errors.append(((true_label, pred_label), int(cm_arr[i, j])))
    errors.sort(key=lambda x: x[1], reverse=True)
    top_errors = errors[:20]
    print('Top confusion pairs (true->pred):')
    for pair, cnt in top_errors:
        print(pair[0], '->', pair[1], cnt)

    print('Done.')
