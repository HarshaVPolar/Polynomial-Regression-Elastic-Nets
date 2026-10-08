# BT2024064 — Polynomial Regression with Elastic Net

This project solves both assignment problems using polynomial regression with Elastic Net regularization.

## Model selection

The degree **n is not hard-coded**. For each candidate degree, the training code searches Elastic Net `alpha` and `l1_ratio` using 5-fold shuffled cross-validation (`random_state=42`) and records the best CV MSE for that degree. The degree with the lowest CV MSE is selected. The selected degree's regularization parameters are then fine-tuned locally and the model is retrained on all 1,000 training rows.

- Var1: degrees 1–10
- Var2: degrees 1–20

`l1_ratio` is the Elastic Net L1/L2 mixing parameter: 1 is Lasso-like, 0 is Ridge-like, and values strictly between 0 and 1 are hybrid Elastic Net settings.

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

The run performs degree/model selection, creates predictions, and generates CV-MSE-vs-degree graphs in `report/figures/`.

## Important files

- `src/utils.py` — Elastic Net degree and hyperparameter search
- `src/var1/train.py` — searches Var1 degree n=1..10
- `src/var2/train.py` — searches Var2 degree n=1..20
- `src/plot_degree_search.py` — generates degree-selection graphs
- `models/*_degree_selection.csv` — CV MSE for every candidate degree
- `models/*_model.joblib` — final fitted models
- `predictions/` — final prediction CSVs
- `report/report.pdf` — 4-page report
- `report/figures/` — degree-search graphs

### Report figures
The project generates two kinds of figures: (1) best Elastic Net CV MSE versus polynomial degree, used to select n, and (2) actual-vs-predicted scatter plots for the final models on the training data. The scatter plots are training-fit diagnostics because the official test targets are hidden.
