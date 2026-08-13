"""Ridge regression with leave-one-season-out CV (numpy only)."""
import numpy as np
import pandas as pd


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def effective_dof(X: np.ndarray, lam: float) -> float:
    p = X.shape[1]
    H = X @ np.linalg.solve(X.T @ X + lam * np.eye(p), X.T)
    return float(np.trace(H))


def loso_cv(X: np.ndarray, y: np.ndarray, seasons: np.ndarray,
            grid=(0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)) -> tuple[float, pd.DataFrame]:
    """Leave-one-SEASON-out CV. Returns (best lambda, table of MAE per lambda)."""
    rows = []
    uniq = np.unique(seasons)
    for lam in grid:
        errs = []
        for s in uniq:
            tr, te = seasons != s, seasons == s
            b = fit_ridge(X[tr], y[tr], lam)
            pred = X[te] @ b
            pred = pred - pred.mean()  # zero-sum within season
            errs.append(np.abs(pred - (y[te] - y[te].mean())).mean())
        rows.append((lam, float(np.mean(errs))))
    tab = pd.DataFrame(rows, columns=["lam", "mae"])
    best = float(tab.sort_values(["mae", "lam"]).iloc[0].lam)
    return best, tab


def loso_coef_signs(X: np.ndarray, y: np.ndarray, seasons: np.ndarray, lam: float) -> np.ndarray:
    """Fraction of LOSO folds where each coefficient keeps the full-fit sign."""
    full = fit_ridge(X, y, lam)
    uniq = np.unique(seasons)
    agree = np.zeros(X.shape[1])
    for s in uniq:
        b = fit_ridge(X[seasons != s], y[seasons != s], lam)
        agree += (np.sign(b) == np.sign(full)) | (full == 0)
    return agree / len(uniq)
