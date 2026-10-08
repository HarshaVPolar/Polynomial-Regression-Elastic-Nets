import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import train_and_save

ROOT = Path(__file__).resolve().parents[2]
meta, _ = train_and_save(
    ROOT / 'data' / 'BT2024064_train_var2.csv',
    ROOT / 'models' / 'var2_model.joblib',
    ROOT / 'models' / 'var2_degree_selection.csv',
    max_degree=20,
)
print(f"var2 selected degree n={meta['degree']}, alpha={meta['best_params']['reg__alpha']:.8g}, "
      f"beta={meta['best_params']['reg__l1_ratio']:.4g}, CV MSE={meta['cv_mse']:.6g}")
