from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

for var in ('var1', 'var2'):
    train = pd.read_csv(ROOT / 'data' / f'BT2024064_train_{var}.csv')
    meta = joblib.load(ROOT / 'models' / f'{var}_model.joblib')
    X = train[meta['feature_columns']]
    y = train['y']
    pred = meta['model'].predict(X)

    lo = min(y.min(), pred.min())
    hi = max(y.max(), pred.max())

    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(y, pred, s=16, alpha=0.55)
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2)
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.title(f'{var}: Final Elastic Net — actual vs predicted')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / f'{var}_final_actual_vs_predicted.png', dpi=180)
    plt.close()
