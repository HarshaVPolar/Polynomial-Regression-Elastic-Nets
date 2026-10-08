from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

for var, max_degree in [('var1', 10), ('var2', 20)]:
    path = ROOT / 'models' / f'{var}_degree_selection.csv'
    df = pd.read_csv(path).sort_values('degree')
    best = df.loc[df['cv_mse'].idxmin()]

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(df['degree'], df['cv_mse'], marker='o')
    plt.axvline(best['degree'], linestyle='--')
    plt.scatter([best['degree']], [best['cv_mse']], s=55)
    plt.xlabel('Polynomial degree n')
    plt.ylabel('Best 5-fold CV MSE')
    plt.title(f'{var}: Elastic Net CV MSE vs polynomial degree')
    plt.xticks(range(1, max_degree + 1))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / f'{var}_elastic_net_cv_mse.png', dpi=180)
    plt.close()
