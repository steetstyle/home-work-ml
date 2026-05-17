"""Modeling utilities: CV, linear models, boosting, comparisons."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
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

LabelSpec = Union[
    Tuple[str, Any],
    Tuple[str, Any, pd.DataFrame],
]


def _neg_rmse_sklearn(est, X, y):
    yp = est.predict(X)
    return -float(root_mean_squared_error(y, yp))


def neg_rmse_scorer():
    """Pickle-safe scorer for contexts that require a callable (use n_jobs=1 if unsure)."""
    return make_scorer(_neg_rmse_sklearn, greater_is_better=True)


def time_series_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    gap: int = 0,
    offset: int = 0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Block expanding-window splits (article-style).
    Train = rows [0 : test_start - gap); test = [test_start : test_end).
  Skips folds with empty train or test_start beyond data.
    """
    if n_samples < 2 or n_splits < 1:
        return
    offset = max(0, int(offset))
    gap = max(0, int(gap))
    denom = max(1, n_splits)
    fold_size = max(1, (n_samples - offset) // denom)
    for i in range(n_splits):
        test_start = offset + i * fold_size + gap
        if test_start >= n_samples:
            continue
        test_end = test_start + fold_size if i < n_splits - 1 else n_samples
        if test_end <= test_start:
            continue
        train_end = max(offset, test_start - gap)
        if train_end < 1:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < 1 or len(test_idx) < 1:
            continue
        yield train_idx, test_idx


class TimeSeriesBlockCV(BaseCrossValidator):
    """Sklearn-compatible block time-series CV with gap and offset."""

    def __init__(self, n_splits: int = 5, gap: int = 0, offset: int = 0):
        self.n_splits = n_splits
        self.gap = gap
        self.offset = offset

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        n = len(X) if X is not None else 0
        return len(list(time_series_cv_splits(n, self.n_splits, self.gap, self.offset)))

    def split(self, X, y=None, groups=None):
        n = len(X)
        for train_idx, test_idx in time_series_cv_splits(
            n, self.n_splits, self.gap, self.offset
        ):
            yield train_idx, test_idx


def time_series_cv(
    n_splits: int = 5, gap: int = 0, offset: int = 0
) -> TimeSeriesBlockCV:
    return TimeSeriesBlockCV(n_splits=n_splits, gap=gap, offset=offset)


def cv_fold_sizes(cv: TimeSeriesBlockCV, n_samples: int) -> pd.DataFrame:
    rows = []
    for i, (tr, te) in enumerate(
        time_series_cv_splits(n_samples, cv.n_splits, cv.gap, cv.offset)
    ):
        rows.append(
            {
                "fold": i + 1,
                "n_train": len(tr),
                "n_test": len(te),
                "train_end": int(tr[-1]) if len(tr) else -1,
                "test_start": int(te[0]) if len(te) else -1,
            }
        )
    return pd.DataFrame(rows)


def prepare_xy_ts(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    date_col: str = "entry_date",
    categorical: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    categorical = categorical or []
    work = df
    if date_col in df.columns:
        work = df.sort_values(date_col)
    use_cols = [c for c in feature_cols if c in work.columns]
    X = work[use_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = work[target].values.astype(float)
    mask = np.isfinite(y)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask]
    return X, y, use_cols


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    categorical: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    return prepare_xy_ts(df, target, feature_cols, date_col="entry_date", categorical=categorical)


def prepare_xy_with_meta(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    meta_cols: Optional[List[str]] = None,
    date_col: str = "entry_date",
) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.DataFrame]:
    """prepare_xy ile aynı satır maskesi; ticker/tarih vb. meta ayrı döner (tahmin tablosu için)."""
    meta_cols = meta_cols or ["ticker", "entry_date", "event_date", "sector"]
    work = df.sort_values(date_col) if date_col in df.columns else df
    use_cols = [c for c in feature_cols if c in work.columns]
    X = work[use_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = work[target].values.astype(float)
    mask = np.isfinite(y)
    keep_meta = [c for c in meta_cols if c in work.columns]
    meta = work.loc[mask, keep_meta].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask]
    return X, y, use_cols, meta


def prepare_xy_class(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    date_col: str = "entry_date",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Chronological rows with binary labels in {0, 1}."""
    work = df.sort_values(date_col) if date_col in df.columns else df
    use_cols = [c for c in feature_cols if c in work.columns and c != label_col]
    X = work[use_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y_raw = work[label_col].values.astype(float)
    mask = np.isfinite(y_raw) & np.isin(y_raw, [0.0, 1.0])
    X = X.loc[mask].reset_index(drop=True)
    y = y_raw[mask].astype(int)
    return X, y, use_cols


def classification_pipeline(est) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", est),
        ]
    )


def discriminant_bundle() -> Dict[str, Pipeline]:
    return {
        "LDA": classification_pipeline(LinearDiscriminantAnalysis()),
        "QDA": classification_pipeline(QuadraticDiscriminantAnalysis(reg_param=0.1)),
        "NaiveBayes": classification_pipeline(GaussianNB()),
    }


def cv_eval_classification(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Union[TimeSeriesBlockCV, Any],
    scoring: str = "roc_auc",
) -> Dict[str, float]:
    """Time-series CV with ROC-AUC (fallback accuracy when AUC undefined)."""
    scores: List[float] = []
    Xv = X
    for tr, te in cv.split(Xv, y):
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_tr)) < 2:
            continue
        est = clone(model)
        Xi_tr = Xv.iloc[tr] if hasattr(Xv, "iloc") else Xv[tr]
        Xi_te = Xv.iloc[te] if hasattr(Xv, "iloc") else Xv[te]
        try:
            est.fit(Xi_tr, y_tr)
        except Exception:
            continue
        try:
            if scoring == "roc_auc" and len(np.unique(y_te)) >= 2:
                proba = est.predict_proba(Xi_te)
                if proba.shape[1] < 2:
                    scores.append(float(accuracy_score(y_te, est.predict(Xi_te))))
                else:
                    scores.append(float(roc_auc_score(y_te, proba[:, 1])))
            else:
                scores.append(float(accuracy_score(y_te, est.predict(Xi_te))))
        except Exception:
            continue
    if not scores:
        return {"cv_score_mean": float("nan"), "cv_score_std": float("nan")}
    arr = np.asarray(scores, dtype=float)
    return {
        "cv_score_mean": float(np.nanmean(arr)),
        "cv_score_std": float(np.nanstd(arr)),
    }


def teach_cv_compare_classify(
    label_specs: Sequence[LabelSpec],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: TimeSeriesBlockCV,
    *,
    scoring: str = "roc_auc",
    plot: bool = False,
    x_series: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows = []
    sizes = cv_fold_sizes(cv, len(y))
    n_train_min = int(sizes["n_train"].min()) if not sizes.empty else 0
    n_test_min = int(sizes["n_test"].min()) if not sizes.empty else 0
    first_est = None
    first_X = None
    for spec in label_specs:
        name = spec[0]
        est = spec[1]
        Xm = spec[2] if len(spec) > 2 else X
        if first_est is None:
            first_est, first_X = est, Xm
        out = cv_eval_classification(est, Xm, y, cv, scoring=scoring)
        rows.append(
            {
                "yöntem": name,
                "cv_score_mean": out["cv_score_mean"],
                "cv_score_std": out["cv_score_std"],
                "n_train_min": n_train_min,
                "n_test_min": n_test_min,
            }
        )
    tab = pd.DataFrame(rows)
    if plot and first_est is not None and first_X is not None:
        plot_all_cv_folds(first_X, y.astype(float), first_est, cv, x_series=x_series)
    return tab


def create_lagged_features(
    df: pd.DataFrame,
    target: str,
    n_lags: int = 1,
    group_col: str = "ticker",
) -> pd.DataFrame:
    return add_target_lags(df, target, n_lags, group_col=group_col)


def add_target_lags(
    df: pd.DataFrame,
    target: str,
    n_lags: int = 1,
    group_col: str = "ticker",
) -> pd.DataFrame:
    out = df.copy()
    if group_col in out.columns:
        g = out.groupby(group_col, group_keys=False)
        for lag in range(1, n_lags + 1):
            out[f"{target}_lag_{lag}"] = g[target].shift(lag)
    else:
        for lag in range(1, n_lags + 1):
            out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out.dropna(subset=[f"{target}_lag_{lag}" for lag in range(1, n_lags + 1)])


def format_sonuc(section: str, bullets: Sequence[str]) -> str:
    lines = [f"### Sonuç — §{section}", ""]
    for b in bullets:
        lines.append(f"- {b}")
    return "\n".join(lines)


def pick_viz_target(
    per_target: Dict[str, Any],
    fallback: Optional[str] = None,
    metric_key: str = "cv_best_lin",
) -> Optional[str]:
    if not per_target:
        return fallback
    scored = []
    for t, d in per_target.items():
        v = d.get(metric_key)
        if v is not None and np.isfinite(v):
            scored.append((t, float(v)))
    if scored:
        return min(scored, key=lambda x: x[1])[0]
    return fallback or next(iter(per_target.keys()), None)


def plot_time_series_cv_fold(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    ax.plot(x, y, color="grey", alpha=0.35, label="Tüm veri")
    ax.scatter(x[train_idx], y[train_idx], color="blue", label="Eğitim", s=18)
    ax.scatter(x[test_idx], y[test_idx], color="green", label="Test", s=22)
    if y_pred is not None:
        ax.plot(x[test_idx], y_pred, color="red", linestyle="--", label="Tahmin (test)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def plot_all_cv_folds(
    X: pd.DataFrame,
    y: np.ndarray,
    model,
    cv: TimeSeriesBlockCV,
    *,
    x_series: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Article-style grid: one subplot per fold."""
    n = len(y)
    splits = list(time_series_cv_splits(n, cv.n_splits, cv.gap, cv.offset))
    if not splits:
        print("CV fold yok (yetersiz örnek).")
        return
    if x_series is None:
        x_series = np.arange(n, dtype=float)
    else:
        x_series = np.asarray(x_series, dtype=float)
    fig, axes = plt.subplots(len(splits), 1, figsize=(figsize[0], max(2, 2 * len(splits))))
    if len(splits) == 1:
        axes = [axes]
    est = clone(model)
    for i, (tr, te) in enumerate(splits):
        ax = axes[i]
        Xi = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        Xe = X.iloc[te] if hasattr(X, "iloc") else X[te]
        est.fit(Xi, y[tr])
        pred = est.predict(Xe)
        plot_time_series_cv_fold(
            ax,
            x_series,
            y,
            tr,
            te,
            y_pred=pred,
            title=f"Fold {i + 1} (gap={cv.gap}, offset={cv.offset})",
        )
    plt.tight_layout()
    plt.show()


def teach_cv_compare(
    label_specs: Sequence[LabelSpec],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: TimeSeriesBlockCV,
    *,
    plot: bool = False,
    x_series: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows = []
    sizes = cv_fold_sizes(cv, len(y))
    n_train_min = int(sizes["n_train"].min()) if not sizes.empty else 0
    n_test_min = int(sizes["n_test"].min()) if not sizes.empty else 0
    first_est = None
    first_X = None
    for spec in label_specs:
        name = spec[0]
        est = spec[1]
        Xm = spec[2] if len(spec) > 2 else X
        if first_est is None:
            first_est, first_X = est, Xm
        out = cv_eval(est, Xm, y, cv)
        rows.append(
            {
                "yöntem": name,
                "cv_rmse_mean": out["cv_rmse_mean"],
                "cv_rmse_std": out["cv_rmse_std"],
                "n_train_min": n_train_min,
                "n_test_min": n_test_min,
            }
        )
    tab = pd.DataFrame(rows)
    if plot and first_est is not None and first_X is not None:
        plot_all_cv_folds(first_X, y, first_est, cv, x_series=x_series)
    return tab


def materialize_fs_matrix(
    fs_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    bundles: Dict[str, Any],
) -> pd.DataFrame:
    """CV sırasında üretilen FS bundle'larından tam veri üzerinde özellik alt matrisi (tahmin için)."""
    if fs_name == "baseline":
        return X.copy()
    if fs_name == "corr_prune":
        return bundles["corr_prune"]["X"].copy()
    if fs_name not in bundles:
        return X.copy()
    pipe = bundles[fs_name]["model"]
    pipe.fit(X, y)
    if fs_name == "kbest":
        return X.loc[:, pipe.named_steps["kb"].get_support()].copy()
    if fs_name == "rfe":
        return X.loc[:, pipe.named_steps["rfe"].support_].copy()
    return X.copy()


def numeric_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", model),
        ]
    )


def cv_eval(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Union[TimeSeriesBlockCV, Any],
    n_jobs: int = 1,
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


def _is_degenerate_gbm(est, Xt: np.ndarray, y: np.ndarray) -> bool:
    """True when the fitted booster ignores features (constant predictions or zero FI)."""
    pred = np.asarray(est.predict(Xt), dtype=float)
    y_std = float(np.std(y))
    if y_std > 1e-12 and float(np.std(pred)) < max(1e-10, 1e-4 * y_std):
        return True
    fi = getattr(est, "feature_importances_", None)
    if fi is not None and fi.size and float(np.max(fi)) == 0.0:
        return True
    return False


def _interpretable_gbm(est):
    """Moderate regularization so trees split for permutation / importance plots."""
    if xgb is not None and isinstance(est, xgb.XGBRegressor):
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_weight=1,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
    if lgb is not None and isinstance(est, lgb.LGBMRegressor):
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_samples=5,
            n_jobs=-1,
            verbosity=-1,
        )
    return clone(est)


def gbm_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 10,
    random_state: int = 0,
    top_n: int = 15,
) -> Tuple[pd.Series, str]:
    """
    Permutation importance on imputed features (inner booster, not full Pipeline).

    If CV-tuned XGB/LGBM collapses to a constant predictor, refits a moderate
  model for interpretation only so importances are meaningful.
    """
    pipe = clone(pipeline)
    pipe.fit(X, y)
    Xt = pipe.named_steps["imp"].transform(X)
    cols = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(Xt.shape[1])]
    note = ""
    if _is_degenerate_gbm(pipe.named_steps["m"], Xt, y):
        reg = _interpretable_gbm(pipe.named_steps["m"])
        reg.fit(Xt, y)
        note = (
            "CV modeli özellikleri kullanmıyordu (sabit tahmin); "
            "permutation importance için yorumlanabilir parametrelerle yeniden eğitildi."
        )
    else:
        reg = pipe.named_steps["m"]
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
    pi = permutation_importance(
        reg,
        Xt,
        y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    imp = pd.Series(pi.importances_mean, index=cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    imp = imp.sort_values(ascending=False).head(top_n)
    return imp, note


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
        "m__reg_alpha": [0, 0.1, 1],
        "m__reg_lambda": [0, 0.1, 1],
        "m__min_child_weight": [1, 3],
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
        "m__reg_alpha": [0, 0.1, 1],
        "m__reg_lambda": [0, 0.1, 1],
        "m__min_child_samples": [1, 5, 10],
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
