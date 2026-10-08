import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import train_and_save

ROOT = Path(__file__).resolve().parents[2]
meta, _ = train_and_save(
    ROOT / 'data' / 'BT2024064_train_var1.csv',
    ROOT / 'models' / 'var1_model.joblib',
    ROOT / 'models' / 'var1_degree_selection.csv',
    max_degree=10,
)
print(f"var1 selected degree n={meta['degree']}, alpha={meta['best_params']['reg__alpha']:.8g}, "
      f"beta={meta['best_params']['reg__l1_ratio']:.4g}, CV MSE={meta['cv_mse']:.6g}")
