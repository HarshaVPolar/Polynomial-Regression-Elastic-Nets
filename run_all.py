from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
steps = [
    [sys.executable, str(ROOT / 'src' / 'var1' / 'train.py')],
    [sys.executable, str(ROOT / 'src' / 'var1' / 'predict.py')],
    [sys.executable, str(ROOT / 'src' / 'var2' / 'train.py')],
    [sys.executable, str(ROOT / 'src' / 'var2' / 'predict.py')],
    [sys.executable, str(ROOT / 'src' / 'plot_degree_search.py')],
    [sys.executable, str(ROOT / 'src' / 'plot_final_models.py')],
]

for i, cmd in enumerate(steps, 1):
    print(f'\n=== STEP {i}/{len(steps)} ===', flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

print('\nDone. Final predictions and degree-selection graphs are in the project.', flush=True)
