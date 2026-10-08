from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CV = KFold(n_splits=5, shuffle=True, random_state=42)
ALPHAS = np.array([3e-3, 1e-2])
L1_RATIOS = np.array([0.2, 0.9])


def load_train(path):
    df = pd.read_csv(path)
    if 'y' not in df.columns:
        raise ValueError("Training CSV must contain column 'y'.")
    return df.drop(columns=['y']), df['y']


def load_test(path, features):
    df = pd.read_csv(path)
    if 'y' in df.columns:
        df = df.drop(columns=['y'])
    return df[list(features)]


def make_pipeline(degree, alpha, l1_ratio):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scale', StandardScaler()),
        ('reg', ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           max_iter=1000, tol=5e-3, selection='random', random_state=42)),
    ])


def search_elastic_net_by_degree(X, y, max_degree):
    """For every degree n, find the best alpha/beta by 5-fold CV.

    beta is represented by sklearn's l1_ratio. The returned CV MSE is
    therefore the best CV MSE achievable on the searched alpha/beta grid
    for that degree.
    """
    rows = []
    for degree in range(1, max_degree + 1):
        model = make_pipeline(degree, ALPHAS[0], L1_RATIOS[0])
        gs = GridSearchCV(
            model,
            {
                'reg__alpha': ALPHAS,
                'reg__l1_ratio': L1_RATIOS,
            },
            cv=CV,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            refit=False,
        )
        gs.fit(X, y)
        rows.append({
            'degree': degree,
            'model': 'ElasticNet',
            'alpha': gs.best_params_['reg__alpha'],
            'l1_ratio': gs.best_params_['reg__l1_ratio'],
            'cv_mse': -gs.best_score_,
        })
        print(f'  degree {degree:2d}: CV MSE={-gs.best_score_:.6g}, '
              f'alpha={gs.best_params_["reg__alpha"]:.6g}, '
              f'beta={gs.best_params_["reg__l1_ratio"]:.2f}', flush=True)

    return pd.DataFrame(rows)


def fine_tune_elastic_net(X, y, degree, coarse_alpha, coarse_beta):
    """Fine search alpha and beta around the best coarse setting."""
    alpha_grid = np.logspace(
        np.log10(max(float(coarse_alpha) / 3, 1e-5)),
        np.log10(float(coarse_alpha) * 3),
        13,
    )
    beta_grid = np.unique(np.clip(
        [coarse_beta + d for d in (-0.2, -0.1, 0.0, 0.1, 0.2)],
        0.05, 0.95
    ))

    model = make_pipeline(degree, alpha_grid[0], beta_grid[0])
    gs = GridSearchCV(
        model,
        {'reg__alpha': alpha_grid, 'reg__l1_ratio': beta_grid},
        cv=CV,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, -gs.best_score_


def train_and_save(train_csv, model_path, results_csv, max_degree):
    X, y = load_train(train_csv)

    print('Searching degree n and Elastic Net alpha/beta:', flush=True)
    degree_results = search_elastic_net_by_degree(X, y, max_degree)
    winner = degree_results.loc[degree_results['cv_mse'].idxmin()]
    degree = int(winner['degree'])

    model, params, cv_mse = fine_tune_elastic_net(
        X, y, degree,
        float(winner['alpha']),
        float(winner['l1_ratio']),
    )

    pred = model.predict(X)
    meta = {
        'model': model,
        'feature_columns': list(X.columns),
        'degree': degree,
        'model_family': 'ElasticNet',
        'best_params': params,
        'cv_mse': float(cv_mse),
        'train_mse': float(mean_squared_error(y, pred)),
        'train_r2': float(r2_score(y, pred)),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta, model_path)
    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    degree_results.to_csv(results_csv, index=False)
    return meta, degree_results


def predict(model_path, test_csv, output_csv):
    meta = joblib.load(model_path)
    X = load_test(test_csv, meta['feature_columns'])
    pd.DataFrame({'y': meta['model'].predict(X)}).to_csv(output_csv, index=False)
