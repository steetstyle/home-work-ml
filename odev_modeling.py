"""Modeling utilities: CV, linear models, boosting, comparisons."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from odev_helpers import drop_high_corr_columns, regression_metrics

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# String scorer pickles under joblib parallel CV (lambda-based scorers do not).
SCORING_NEG_RMSE = "neg_root_mean_squared_error"


def _neg_rmse_sklearn(est, X, y):
    yp = est.predict(X)
    return -float(root_mean_squared_error(y, yp))


def neg_rmse_scorer():
    """Pickle-safe scorer for contexts that require a callable (use n_jobs=1 if unsure)."""
    return make_scorer(_neg_rmse_sklearn, greater_is_better=True)


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    categorical: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    categorical = categorical or []
    use_cols = [c for c in feature_cols if c in df.columns]
    X = df[use_cols].copy()
    # sklearn SimpleImputer rejects inf — replace so median imputation works
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df[target].values.astype(float)
    mask = np.isfinite(y)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask]
    return X, y, use_cols


def numeric_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", model),
        ]
    )


def cv_eval(
    model, X: pd.DataFrame, y: np.ndarray, cv: TimeSeriesSplit, n_jobs: int = 1
) -> Dict[str, float]:
    """n_jobs=1 by default: nested parallel CV is fragile; inf-cleaned X still safer serial."""
    scores = cross_val_score(
        model, X, y, cv=cv, scoring=SCORING_NEG_RMSE, n_jobs=n_jobs
    )
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=SCORING_NEG_RMSE, n_jobs=1
        )
        scores = np.asarray(scores, dtype=float)
    # scores == -RMSE
    rmse = -scores
    if not np.isfinite(rmse).any():
        return {"cv_rmse_mean": float("nan"), "cv_rmse_std": float("nan")}
    return {
        "cv_rmse_mean": float(np.nanmean(rmse)),
        "cv_rmse_std": float(np.nanstd(rmse)),
    }


def linear_baseline_cv(X, y, cv) -> Dict[str, Any]:
    pipe = numeric_pipeline(LinearRegression())
    return {"model": pipe, **cv_eval(pipe, X, y, cv)}


def correlation_prune_cv(X, y, cv, threshold=0.95) -> Dict[str, Any]:
    Xp = drop_high_corr_columns(X, threshold=threshold)
    pipe = numeric_pipeline(LinearRegression())
    return {"model": pipe, "X": Xp, **cv_eval(pipe, Xp, y, cv)}


def select_k_best_cv(X, y, cv, k=25) -> Dict[str, Any]:
    cols = X.columns.tolist()
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("kb", SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))),
            ("sc", StandardScaler()),
            ("m", LinearRegression()),
        ]
    )
    return {"model": pipe, **cv_eval(pipe, X, y, cv)}


def rfe_cv(X, y, cv, n_features=25) -> Dict[str, Any]:
    n_feat = min(n_features, X.shape[1] - 1, max(1, X.shape[1] // 2))
    base = Ridge(alpha=1.0)
    rfe = RFE(estimator=base, n_features_to_select=n_feat, step=1)
    pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("rfe", rfe), ("sc", StandardScaler()), ("m", LinearRegression())]
    )
    return {"model": pipe, **cv_eval(pipe, X, y, cv)}


def grid_regularized(
    X,
    y,
    cv,
    kind: str = "ridge",
    alphas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    alphas = alphas or [1e-3, 1e-2, 0.1, 1, 10, 100]
    if kind == "ridge":
        est = Ridge()
        param = {"m__alpha": alphas}
    elif kind == "lasso":
        est = Lasso(max_iter=5000)
        param = {"m__alpha": alphas}
    else:
        est = ElasticNet(max_iter=8000, l1_ratio=0.5)
        param = {"m__alpha": alphas}
    pipe = numeric_pipeline(est)
    gs = GridSearchCV(
        pipe,
        param,
        cv=cv,
        scoring=SCORING_NEG_RMSE,
        n_jobs=1,
        refit=True,
    )
    gs.fit(X, y)
    return {"model": gs.best_estimator_, "grid": gs, **cv_eval(gs.best_estimator_, X, y, cv)}


def randomized_xgb(X, y, cv, n_iter=24, seed=42) -> Dict[str, Any]:
    if xgb is None:
        return {}
    reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", reg)])
    space = {
        "m__max_depth": [3, 5, 7, 10],
        "m__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "m__subsample": [0.6, 0.8, 1.0],
        "m__colsample_bytree": [0.6, 0.8, 1.0],
        "m__reg_alpha": [0, 0.1, 1, 10],
        "m__reg_lambda": [0, 0.1, 1, 10],
        "m__min_child_weight": [1, 3, 5],
    }
    rs = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        cv=cv,
        scoring=SCORING_NEG_RMSE,
        random_state=seed,
        n_jobs=1,
        refit=True,
    )
    rs.fit(X, y)
    return {"model": rs.best_estimator_, "search": rs, **cv_eval(rs.best_estimator_, X, y, cv)}


def randomized_lgbm(X, y, cv, n_iter=24, seed=42) -> Dict[str, Any]:
    if lgb is None:
        return {}
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        min_child_samples=1,
        min_data_in_leaf=1,
    )
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", reg)])
    space = {
        "m__max_depth": [3, 5, 7, 10, -1],
        "m__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "m__subsample": [0.6, 0.8, 1.0],
        "m__colsample_bytree": [0.6, 0.8, 1.0],
        "m__reg_alpha": [0, 0.1, 1, 10],
        "m__reg_lambda": [0, 0.1, 1, 10],
        "m__min_child_samples": [1, 5, 10, 20],
    }
    rs = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        cv=cv,
        scoring=SCORING_NEG_RMSE,
        random_state=seed,
        n_jobs=1,
        refit=True,
    )
    rs.fit(X, y)
    return {"model": rs.best_estimator_, "search": rs, **cv_eval(rs.best_estimator_, X, y, cv)}


def fit_predict_metrics(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return regression_metrics(y_test, pred)


def add_ticker_dummies(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    d = pd.get_dummies(df["ticker"], prefix="tk")
    return pd.concat([X.reset_index(drop=True), d.reset_index(drop=True)], axis=1)


def add_sector_peer_returns(
    events: pd.DataFrame, X: pd.DataFrame, sector_ret_col: str = "sector_ret_1d"
) -> pd.DataFrame:
    """Append sector basket return column from events (same as ETF-era helper, new name)."""
    if sector_ret_col in events.columns:
        extra = events[[sector_ret_col]].copy()
        extra.columns = ["sector_feat_ret"]
        return pd.concat([X.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return X
