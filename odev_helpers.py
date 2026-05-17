"""
Helpers for earnings post-event return regression project.
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


def configure_plot_style(*, dpi: int = 200, savefig_dpi: int = 400, retina: bool = True) -> None:
    """Notebook grafikleri: yüksek DPI, retina (Jupyter), biraz daha kalın çizgi/işaret."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": savefig_dpi,
            "figure.figsize": (12, 6),
            "lines.linewidth": 1.75,
            "lines.markersize": 6,
            "patch.linewidth": 0.8,
            "axes.linewidth": 1.1,
            "grid.linewidth": 0.9,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "figure.autolayout": False,
        }
    )
    if retina:
        try:
            from matplotlib_inline import backend_inline

            backend_inline.set_matplotlib_formats("retina", "png")
        except Exception:
            pass


# Varsayılan evren (tickers.csv yoksa kullanılır ve dosya oluşturulur)
TICKERS_CSV = Path(__file__).resolve().parent / "tickers.csv"
# Ham günlük panel (Adj Close + Volume); varsa yfinance tekrar çağrılmaz
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PRICES_LONG_CSV = DATA_DIR / "raw_prices_long.csv"
# yfinance get_earnings_dates (ticker başına ağır); evren + limit aynıysa tekrar çağrılmaz
EARNINGS_CACHE_CSV = DATA_DIR / "earnings_dates_cache.csv"
EARNINGS_CACHE_META = DATA_DIR / "earnings_cache_meta.json"

DEFAULT_SECTOR_TICKERS: Dict[str, List[str]] = {
    "Semiconductor": ["AMD", "INTC", "MU", "NVDA", "QCOM", "TXN"],
    "Tech_Cloud": ["MSFT", "GOOGL", "AMZN", "META"],
    "Healthcare": ["JNJ", "PFE", "UNH"],
    "Finance": ["JPM", "BAC", "GS"],
    "Energy": ["XOM", "CVX"],
}


def read_sector_tickers_csv(path: Path = TICKERS_CSV) -> Optional[Dict[str, List[str]]]:
    """tickers.csv varsa sector -> ticker listesi döner; yoksa None."""
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError(f"{path}: 'ticker' sütunu gerekli")
    if "sector" not in df.columns:
        df = df.copy()
        df["sector"] = "General"
    out: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        t = str(row["ticker"]).strip().upper()
        if not t or t.lower() in ("nan", "none", ""):
            continue
        s = str(row["sector"]).strip()
        if not s or s.lower() == "nan":
            s = "General"
        out[s].append(t)
    return {k: list(v) for k, v in out.items()}


def write_sector_tickers_csv(
    data: Dict[str, List[str]], path: Path = TICKERS_CSV
) -> None:
    """Sektör / ticker tablosunu CSV'ye yazar (sector, ticker sütunları)."""
    rows = []
    for sector in sorted(data.keys()):
        for ticker in data[sector]:
            rows.append(
                {"sector": sector, "ticker": str(ticker).strip().upper()}
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def load_sector_tickers(path: Path = TICKERS_CSV) -> Dict[str, List[str]]:
    """CSV varsa oradan; yoksa varsayılan sözlük ve tickers.csv oluşturulur."""
    loaded = read_sector_tickers_csv(path)
    if loaded is not None:
        return loaded
    data = {k: list(v) for k, v in DEFAULT_SECTOR_TICKERS.items()}
    write_sector_tickers_csv(data, path)
    return data


def reload_sector_tickers(path: Path = TICKERS_CSV) -> None:
    """Çalışma anında tickers.csv'yi yeniden yükler (not defteri hücresinden çağrılabilir)."""
    global SECTOR_TICKERS, ALL_STOCKS, ALL_EVENT_TICKERS
    loaded = read_sector_tickers_csv(path)
    if loaded is None:
        loaded = {k: list(v) for k, v in DEFAULT_SECTOR_TICKERS.items()}
    SECTOR_TICKERS = loaded
    ALL_STOCKS = sorted({t for v in SECTOR_TICKERS.values() for t in v})
    ALL_EVENT_TICKERS = list(ALL_STOCKS)


SECTOR_TICKERS = load_sector_tickers()
ALL_STOCKS = sorted({t for v in SECTOR_TICKERS.values() for t in v})
ALL_EVENT_TICKERS = list(ALL_STOCKS)


def _universe_cols(adj: pd.DataFrame) -> List[str]:
    return [c for c in ALL_EVENT_TICKERS if c in adj.columns]


def market_basket_close(adj: pd.DataFrame) -> pd.Series:
    """Equal-weight average close across all modeled stocks (ETF-free market proxy)."""
    cols = _universe_cols(adj)
    if not cols:
        raise ValueError("No universe columns in adj")
    return adj[cols].mean(axis=1, skipna=True)


def sector_peer_basket_close(adj: pd.DataFrame, sector: str, ticker: str) -> pd.Series:
    """Equal-weight average close of same-sector peers excluding `ticker`."""
    peers = [x for x in SECTOR_TICKERS[sector] if x != ticker and x in adj.columns]
    if peers:
        return adj[peers].mean(axis=1, skipna=True)
    return market_basket_close(adj)

# Trading-day horizons: target column names must not collide with lagged return features (ret_1d, ...)
HORIZON_DAYS: Dict[str, int] = {
    "y_1d": 1,
    "y_3d": 3,
    "y_1w": 5,
    "y_2w": 10,
    "y_3w": 15,
    "y_1m": 21,
    "y_3m": 63,
}


def flat_ticker_list() -> List[str]:
    """Sadece hisseler; ETF indirilmez."""
    return sorted(ALL_EVENT_TICKERS)


def ticker_sector_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for sector, tickers in SECTOR_TICKERS.items():
        for t in tickers:
            m[t] = sector
    return m


def download_ohlcv(
    tickers: List[str], progress: bool = False, auto_adjust: bool = True
) -> pd.DataFrame:
    """Multi-index columns (field, ticker): Adj Close, Close, Volume."""
    data = yf.download(
        tickers=tickers,
        period="max",
        interval="1d",
        group_by="ticker",
        auto_adjust=auto_adjust,
        threads=True,
        progress=progress,
    )
    if isinstance(data.columns, pd.MultiIndex):
        return data
    # single ticker
    t = tickers[0]
    out = pd.concat({t: data}, axis=1)
    out.columns = pd.MultiIndex.from_tuples([(t, c) for c in data.columns])
    return out


def wide_adj_close_volume(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide price and Volume matrices (dates x tickers). Uses Adj Close if present else Close."""
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        fields = df.columns.get_level_values(1).unique().tolist()
        price_key = "Adj Close" if "Adj Close" in fields else "Close"
        adj = df.xs(price_key, axis=1, level=1)
        vol = df.xs("Volume", axis=1, level=1)
        return adj, vol
    raise ValueError("Unexpected yfinance shape")


def _wide_to_long_raw_cache(adj: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    adj = adj.sort_index()
    vol = vol.reindex(adj.index)
    parts: List[pd.DataFrame] = []
    for t in adj.columns:
        parts.append(
            pd.DataFrame(
                {
                    "date": adj.index,
                    "ticker": str(t).upper(),
                    "adj_close": adj[t].values,
                    "volume": vol[t].values if t in vol.columns else np.nan,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _long_to_wide_raw(long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long = long.copy()
    long.columns = [str(c).strip().lower() for c in long.columns]
    if not {"date", "ticker", "adj_close", "volume"}.issubset(long.columns):
        raise ValueError("raw cache CSV must have columns: date, ticker, adj_close, volume")
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    long["ticker"] = long["ticker"].astype(str).str.upper()
    long = long.drop_duplicates(subset=["date", "ticker"], keep="last")
    piv_close = long.pivot(index="date", columns="ticker", values="adj_close")
    piv_vol = long.pivot(index="date", columns="ticker", values="volume")
    piv_close.index.name = None
    piv_close.columns.name = None
    piv_vol.index.name = None
    piv_vol.columns.name = None
    return piv_close.sort_index(), piv_vol.sort_index()


def load_raw_prices_from_csv(
    path: Path, required_tickers: List[str]
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Dosya yoksa veya gerekli ticker eksikse None."""
    if not path.is_file():
        return None
    try:
        long = pd.read_csv(path)
    except Exception:
        return None
    try:
        long.columns = [str(c).strip().lower() for c in long.columns]
        if not {"date", "ticker", "adj_close", "volume"}.issubset(set(long.columns)):
            return None
        have = set(long["ticker"].astype(str).str.upper())
        req = [str(t).upper() for t in required_tickers]
        if not set(req) <= have:
            return None
        long = long[long["ticker"].isin(req)]
        adj, vol = _long_to_wide_raw(long)
        missing = [t for t in req if t not in adj.columns]
        if missing:
            return None
        adj = adj[req]
        vol = vol.reindex(columns=req).reindex(adj.index)
        return adj, vol
    except Exception:
        return None


def save_raw_prices_to_csv(
    adj: pd.DataFrame,
    vol: pd.DataFrame,
    path: Path = RAW_PRICES_LONG_CSV,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _wide_to_long_raw_cache(adj, vol).to_csv(path, index=False)


def get_earnings_history(ticker: str, limit: int = 40) -> pd.DataFrame:
    """EPS actual vs estimate when available (yfinance)."""
    t = yf.Ticker(ticker)
    rows = []
    try:
        q = t.get_earnings_dates(limit=limit)
        if q is not None and len(q):
            q = q.copy()
            q["ticker"] = ticker
            rows.append(q)
    except Exception:
        pass
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out.sort_index()
    return out


def _earnings_df_to_long_rows(edf: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Wide earnings (index=date) → uzun satırlar (önbellek CSV)."""
    t = str(ticker).upper()
    if edf is None or edf.empty:
        return pd.DataFrame(columns=["ticker", "date", "eps_estimate", "eps_reported"])
    x = edf.reset_index()
    date_col = x.columns[0]
    x = x.rename(columns={date_col: "date"})
    est = x.get("EPS Estimate", pd.Series(np.nan, index=x.index))
    act = x.get("Reported EPS", pd.Series(np.nan, index=x.index))
    return pd.DataFrame(
        {
            "ticker": t,
            "date": pd.to_datetime(x["date"]).dt.tz_localize(None).dt.normalize(),
            "eps_estimate": pd.to_numeric(est, errors="coerce"),
            "eps_reported": pd.to_numeric(act, errors="coerce"),
        }
    )


def _long_rows_to_earnings_df(sub: pd.DataFrame) -> pd.DataFrame:
    """Önbellek satırları → build_event_table’ın beklediği sütun adları."""
    if sub is None or sub.empty:
        return pd.DataFrame()
    s = sub.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None).dt.normalize()
    out = s.set_index("date")[["eps_estimate", "eps_reported"]].copy()
    out.columns = ["EPS Estimate", "Reported EPS"]
    return out.sort_index()


def load_earnings_cache_map(
    path_csv: Path,
    path_meta: Path,
    required_tickers: List[str],
    limit: int,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Evren ve limit önbellekle uyuyorsa ticker → earnings DataFrame; aksi None."""
    if not path_csv.is_file() or not path_meta.is_file():
        return None
    try:
        meta = json.loads(path_meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    req = sorted({str(x).upper() for x in required_tickers})
    meta_tickers = sorted({str(x).upper() for x in meta.get("tickers", [])})
    if meta_tickers != req:
        return None
    if int(meta.get("limit", 0)) < int(limit):
        return None
    try:
        long = pd.read_csv(path_csv)
    except Exception:
        return None
    long.columns = [str(c).strip().lower() for c in long.columns]
    if not long.empty and not {"ticker", "date", "eps_estimate", "eps_reported"}.issubset(
        long.columns
    ):
        return None
    out: Dict[str, pd.DataFrame] = {}
    if long.empty:
        for t in req:
            out[t] = pd.DataFrame()
        return out
    long["ticker"] = long["ticker"].astype(str).str.upper()
    for t in req:
        sub = long.loc[long["ticker"] == t, ["date", "eps_estimate", "eps_reported"]]
        out[t] = _long_rows_to_earnings_df(sub)
    return out


def save_earnings_cache(
    by_ticker: Dict[str, pd.DataFrame],
    path_csv: Path,
    path_meta: Path,
    limit: int,
) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    tickers = sorted(by_ticker.keys())
    parts = [_earnings_df_to_long_rows(by_ticker.get(t, pd.DataFrame()), t) for t in tickers]
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["ticker", "date", "eps_estimate", "eps_reported"]
    )
    long.to_csv(path_csv, index=False)
    path_meta.write_text(
        json.dumps({"tickers": tickers, "limit": int(limit)}, indent=1),
        encoding="utf-8",
    )


def trading_day_offset(
    index: pd.DatetimeIndex, day: pd.Timestamp, offset: int
) -> Optional[pd.Timestamp]:
    """Return date at `offset` trading steps from `day` (day must exist in index). offset>0 = forward."""
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(index).unique()))
    day = pd.Timestamp(day).normalize()
    loc = int(idx.get_indexer([day], method="pad")[0])
    if loc < 0 or loc >= len(idx) or idx[loc] != day:
        return None
    j = loc + offset
    if j >= len(idx):
        return None
    return pd.Timestamp(idx[j])


def prev_trading_day(index: pd.DatetimeIndex, ref: pd.Timestamp) -> Optional[pd.Timestamp]:
    idx = index.sort_values()
    pos = idx.searchsorted(pd.Timestamp(ref).normalize(), side="left")
    if pos == 0:
        return None
    return idx[pos - 1]


def next_trading_day(index: pd.DatetimeIndex, ref: pd.Timestamp) -> Optional[pd.Timestamp]:
    idx = index.sort_values()
    pos = idx.searchsorted(pd.Timestamp(ref).normalize(), side="right")
    if pos >= len(idx):
        return None
    return idx[pos]


def add_technical_features(
    close: pd.Series,
    volume: pd.Series,
    market_close: pd.Series,
    sector_close: pd.Series,
) -> pd.DataFrame:
    """Daily features (no lookahead) for one ticker aligned on index.

    ``market_close`` / ``sector_close``: ETF değil; tüm evren (market) ve aynı sektördeki diğer
    hisselerin eşit ağırlıklı kapanış ortalaması (leave-one-out sektör sepeti).
    """
    from ta.momentum import RSIIndicator, ROCIndicator
    from ta.trend import MACD

    df = pd.DataFrame({"close": close, "vol": volume}).sort_index()
    mkt_c = market_close.reindex(df.index)
    sec_c = sector_close.reindex(df.index)
    df["ret_1d"] = df["close"].pct_change()
    for w in (2, 3, 5, 10, 20, 60):
        df[f"ret_{w}d"] = df["close"].pct_change(w)
    for w in (5, 10, 20, 60):
        df[f"vol_std_{w}"] = df["ret_1d"].rolling(w).std()
    df["vol_chg"] = df["vol"].pct_change()
    df["vol_ratio_20"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-9)
    z = (df["vol"] - df["vol"].rolling(20).mean()) / (df["vol"].rolling(20).std() + 1e-9)
    df["abnormal_vol_z"] = z
    rsi = RSIIndicator(df["close"], window=14)
    df["rsi_14"] = rsi.rsi()
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    roc = ROCIndicator(df["close"], window=10)
    df["roc_10"] = roc.roc()
    for w in (20, 50, 200):
        ma = df["close"].rolling(w).mean()
        df[f"dist_sma_{w}"] = (df["close"] - ma) / (ma + 1e-9)
    df["mkt_ret_1d"] = mkt_c.pct_change()
    df["mkt_ret_5d"] = mkt_c.pct_change(5)
    df["sector_ret_1d"] = sec_c.pct_change()
    df["sector_ret_5d"] = sec_c.pct_change(5)
    sret = df["ret_1d"]
    seret = sec_c.pct_change()
    df["roll_corr_stock_sector_60"] = sret.rolling(60).corr(seret)
    return df


def build_event_table(
    ticker: str,
    sector: str,
    close: pd.Series,
    volume: pd.Series,
    full_index: pd.DatetimeIndex,
    market_close: pd.Series,
    sector_peer_close: pd.Series,
    earnings_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per earnings date with features as-of prior close and forward returns from entry."""
    tech = add_technical_features(close, volume, market_close, sector_peer_close)

    events = []
    if earnings_df is None or earnings_df.empty:
        return pd.DataFrame()

    for dt, row in earnings_df.iterrows():
        dt = pd.Timestamp(dt).normalize()
        entry = next_trading_day(full_index, dt)
        if entry is None:
            continue
        feat_day = prev_trading_day(full_index, entry)
        if feat_day is None or feat_day not in tech.index:
            continue
        feats = tech.loc[feat_day].copy()
        # earnings-based features
        eps_est = row.get("EPS Estimate", np.nan)
        eps_act = row.get("Reported EPS", np.nan)
        surprise = np.nan
        if pd.notna(eps_est) and pd.notna(eps_act) and eps_est != 0:
            surprise = (eps_act - eps_est) / (abs(eps_est) + 1e-9)
        feats_dict = feats.to_dict()
        feats_dict.update(
            {
                "ticker": ticker,
                "sector": sector,
                "event_date": dt,
                "entry_date": entry,
                "feature_date": feat_day,
                "eps_estimate": eps_est,
                "eps_reported": eps_act,
                "eps_surprise_pct": surprise,
                "eps_beat": int(
                    pd.notna(eps_est)
                    and pd.notna(eps_act)
                    and (eps_act > eps_est)
                ),
            }
        )
        # forward returns from entry close to entry + h
        base_px = float(close.loc[entry])
        if not np.isfinite(base_px) or base_px == 0:
            continue
        for col, h in HORIZON_DAYS.items():
            fut = trading_day_offset(full_index, entry, h)
            if fut is None or fut not in close.index:
                feats_dict[col] = np.nan
                continue
            px = float(close.loc[fut])
            feats_dict[col] = (px / base_px) - 1.0
        events.append(feats_dict)

    return pd.DataFrame(events)


def attach_historical_earnings_features(events: pd.DataFrame) -> pd.DataFrame:
    """Rolling past surprise / post-event drift (same ticker, prior rows)."""
    if events.empty:
        return events
    ev = events.sort_values(["ticker", "entry_date"]).copy()
    g = ev.groupby("ticker", group_keys=False)
    ev["surprise_lag1"] = g["eps_surprise_pct"].shift(1)
    ev["surprise_roll4_mean"] = g["eps_surprise_pct"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    for c in list(HORIZON_DAYS.keys()):
        ev[f"prior_post_{c}_mean4"] = g[c].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )
    return ev


def try_implied_vol_snapshot(ticker: str, asof: pd.Timestamp) -> float:
    """Best-effort ATM IV from nearest expiry (may fail often)."""
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return np.nan
        exp = expiries[0]
        chain = t.option_chain(exp)
        und = float(t.history(start=asof - pd.Timedelta(days=5), end=asof + pd.Timedelta(days=5))["Close"].iloc[-1])
        calls = chain.calls
        if calls.empty:
            return np.nan
        calls = calls.copy()
        calls["diff"] = (calls["strike"] - und).abs()
        row = calls.sort_values("diff").iloc[0]
        return float(row.get("impliedVolatility", np.nan))
    except Exception:
        return np.nan


def correlation_bundle(returns_wide: pd.DataFrame) -> Dict[str, object]:
    corr = returns_wide.corr()
    return {"corr": corr}


# Takvim sonu → geriye N ay (iş günü getiri paneli üzerinde korelasyon)
CORRELATION_CALENDAR_WINDOWS: List[Tuple[str, int]] = [
    ("1m", 1),
    ("2m", 2),
    ("3m", 3),
    ("6m", 6),
    ("9m", 9),
    ("1y", 12),
    ("2y", 24),
]

CORR_WINDOW_TITLE_TR: Dict[str, str] = {
    "1m": "Son 1 ay",
    "2m": "Son 2 ay",
    "3m": "Son 3 ay",
    "6m": "Son 6 ay",
    "9m": "Son 9 ay",
    "1y": "Son 1 yıl",
    "2y": "Son 2 yıl",
}


def slice_returns_calendar_tail(
    returns_wide: pd.DataFrame,
    months: int,
    *,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Son `months` takvim ayı içindeki satırlar (end dahil, günlük / işgünü indeks)."""
    df = returns_wide.sort_index()
    if df.empty:
        return df.copy()
    idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df.copy()
    df.index = idx
    end_ts = pd.Timestamp(end if end is not None else df.index.max()).normalize()
    start_ts = (end_ts - pd.DateOffset(months=months)).normalize()
    sub = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return sub.dropna(how="all", axis=0)


def correlation_bundles_by_calendar_windows(
    returns_wide: pd.DataFrame,
    windows: Optional[List[Tuple[str, int]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Her pencere için Pearson korelasyon matrisi.

    Returns
    -------
    dict key -> {"corr", "n_obs", "months"}
    """
    spec = windows if windows is not None else CORRELATION_CALENDAR_WINDOWS
    out: Dict[str, Dict[str, Any]] = {}
    for key, months in spec:
        sub = slice_returns_calendar_tail(returns_wide, months)
        n = int(len(sub))
        if n < 2:
            out[key] = {"corr": pd.DataFrame(), "n_obs": n, "months": months}
        else:
            b = correlation_bundle(sub)
            out[key] = {**b, "n_obs": n, "months": months}
    return out


def primary_corr_from_window_bundles(
    bundles: Dict[str, Dict[str, Any]],
    preference: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Excel / özet için: tercih sırasına göre ilk dolu korelasyon matrisi."""
    order = preference or [k for k, _ in reversed(CORRELATION_CALENDAR_WINDOWS)]
    for k in order:
        if k not in bundles:
            continue
        c = bundles[k].get("corr")
        if isinstance(c, pd.DataFrame) and not c.empty:
            return c
    return None


def drop_high_corr_columns(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return X.drop(columns=to_drop, errors="ignore")


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape(yt, yp)}


META_COLS = [
    "ticker",
    "sector",
    "event_date",
    "entry_date",
    "feature_date",
    "eps_estimate",
    "eps_reported",
]


def target_columns() -> List[str]:
    return list(HORIZON_DAYS.keys())


def feature_columns(df: pd.DataFrame) -> List[str]:
    skip = set(META_COLS) | set(target_columns())
    num = []
    for c in df.columns:
        if c in skip:
            continue
        if df[c].dtype == object:
            continue
        num.append(c)
    return num


CLASSIFICATION_BEAT_EXCLUDE = [
    "eps_beat",
    "eps_surprise_pct",
    "eps_estimate",
    "eps_reported",
]


def classification_feature_columns(
    df: pd.DataFrame,
    label_kind: str = "beat",
    horizon: Optional[str] = None,
) -> List[str]:
    """Features for classification; excludes label leakage columns."""
    cols = feature_columns(df)
    if label_kind == "beat":
        excl = set(CLASSIFICATION_BEAT_EXCLUDE)
        return [c for c in cols if c not in excl]
    return cols


def make_direction_label(events: pd.DataFrame, horizon_col: str) -> pd.Series:
    """Binary 1 if forward return > 0, NaN if return missing."""
    if horizon_col not in events.columns:
        return pd.Series(np.nan, index=events.index)
    r = events[horizon_col].astype(float)
    out = pd.Series(np.nan, index=events.index, dtype=float)
    m = np.isfinite(r.values)
    out.loc[m] = (r.loc[m] > 0).astype(float)
    return out


def assemble_dataset(
    progress: bool = False,
    force_refresh_raw: bool = False,
    raw_cache_path: Optional[Path] = None,
    force_refresh_earnings: bool = False,
    earnings_limit: int = 60,
    earnings_cache_csv: Optional[Path] = None,
    earnings_cache_meta: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fiyatları yükle (önbellek CSV veya yfinance), getiri matrisi ve earnings paneli.

    Ham günlük veri ``data/raw_prices_long.csv`` (uzun format) olarak saklanır.
    Dosya mevcut ve ``tickers.csv`` evrenindeki tüm ticker'lar dosyada ise
    yfinance indirmesi atlanır. Yenilemek için ``force_refresh_raw=True`` veya
    CSV'yi silin.

    Kazanç tarihleri ``get_earnings_dates`` ile ticker başına çekilir; bu yavaştır.
    ``data/earnings_dates_cache.csv`` + ``earnings_cache_meta.json`` ile aynı
    ticker listesi ve ``earnings_limit`` için tekrar çağrılmaz. Yenilemek için
    ``force_refresh_earnings=True`` veya bu dosyaları silin.

    Returns
    -------
    events : event-level feature + target table
    adj : wide adj close (all tickers)
    returns : pct_change of adj
    """
    cache = raw_cache_path or RAW_PRICES_LONG_CSV
    ec = earnings_cache_csv or EARNINGS_CACHE_CSV
    em = earnings_cache_meta or EARNINGS_CACHE_META
    tickers = flat_ticker_list()
    cached = (
        None
        if force_refresh_raw
        else load_raw_prices_from_csv(cache, tickers)
    )
    downloaded = False
    if cached is not None:
        adj, vol = cached
    else:
        raw = download_ohlcv(tickers, progress=progress)
        adj, vol = wide_adj_close_volume(raw)
        downloaded = True
    adj = adj.sort_index()
    adj.index = pd.to_datetime(adj.index).tz_localize(None).normalize()
    vol = vol.reindex(adj.index).sort_index()
    if downloaded:
        save_raw_prices_to_csv(adj, vol, cache)
    rets = adj.pct_change()

    tmap = ticker_sector_map()
    market = market_basket_close(adj)
    event_tickers = [t for t in ALL_EVENT_TICKERS if t in adj.columns]

    earnings_map: Optional[Dict[str, pd.DataFrame]] = None
    if not force_refresh_earnings and event_tickers:
        earnings_map = load_earnings_cache_map(ec, em, event_tickers, earnings_limit)

    if earnings_map is None and event_tickers:
        earnings_map = {}
        for t in event_tickers:
            earnings_map[t] = get_earnings_history(t, limit=earnings_limit)
        save_earnings_cache(earnings_map, ec, em, earnings_limit)

    frames = []
    if event_tickers:
        assert earnings_map is not None
    for t in event_tickers:
        sector = tmap[t]
        peer = sector_peer_basket_close(adj, sector, t)
        edf = earnings_map[t]  # type: ignore[index]
        ev = build_event_table(
            ticker=t,
            sector=sector,
            close=adj[t],
            volume=vol[t],
            full_index=adj.index,
            market_close=market,
            sector_peer_close=peer,
            earnings_df=edf,
        )
        if not ev.empty:
            frames.append(ev)
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    events = attach_historical_earnings_features(events)
    if not events.empty:
        num_cols = events.select_dtypes(include=[np.number]).columns
        events[num_cols] = events[num_cols].replace([np.inf, -np.inf], np.nan)
    return events, adj, rets
