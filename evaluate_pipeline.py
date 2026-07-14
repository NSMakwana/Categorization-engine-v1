import sys, json
sys.path.append(r'D:\Nency\Mitex Intel System\Categorization-engine-v1')
import pandas as pd
from engine.pipeline import process_transactions

csv_path = r'D:\Nency\Mitex Intel System\Transaction Generator\output\cleaned_transactions.csv'
print('READING', csv_path)

df = pd.read_csv(csv_path)
print('ROWS', len(df))

df2 = process_transactions(df)

y_true = df2['Old Category'] if 'Old Category' in df2.columns else df2['Category']
y_pred = df2['Category']

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(json.dumps({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}))
except Exception as e:
    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / total if total else 0
    print(json.dumps({'accuracy': acc, 'note': 'sklearn_missing_or_error', 'error': str(e)}))
