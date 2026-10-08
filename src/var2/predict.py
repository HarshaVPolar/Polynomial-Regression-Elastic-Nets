import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import predict
ROOT=Path(__file__).resolve().parents[2]
predict(ROOT/'models'/'var2_model.joblib',ROOT/'data'/'BT2024064_test_var2.csv',ROOT/'predictions'/'BT2024064_pred_var2.csv')
print('Saved var2 predictions.')
