from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from IPython.display import display
from urllib.request import urlopen
from urllib.parse import quote
from networkx import display
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import os
try:
    from matplotlib import pyplot as plt
except Exception:  # pragma: no cover - notebooks usually provide matplotlib
    plt = None

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional at import time
    stats = None


def _as_dataframe(data: Any) -> pd.DataFrame:
    """This function converts supported input data into a DataFrame.

    Params:
        data: Input records or DataFrame.
    """
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    return pd.DataFrame(data)


def _read_market_csv_text(text: str, source_name: str="market data") -> pd.DataFrame:
    """This function validates and parses market CSV text.

    Params:
        text: Input text.
        source_name: Source label used in errors.
    """
    preview = text.strip()[:240].lower()

    if not text.strip():
        raise ValueError(f"{source_name} returned an empty response.")

    if "get your apikey" in preview or "captcha" in preview:
        raise ValueError(
            f"{source_name} requires an API key. "
            "Set STOOQ_API_KEY or use source='yahoo' / local_path."
        )

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise ValueError(
            f"{source_name} did not return a valid CSV. "
            f"First response characters were: {text[:120]!r}"
        ) from exc

    if df.empty:
        raise ValueError(f"{source_name} returned a valid CSV shape but no rows.")

    return df


def fetch_stooq_daily(
    symbol: str,
    start: str | pd.Timestamp | None=None,
    end: str | pd.Timestamp | None=None,
    cache_dir: str | os.PathLike[str]="data/real",
    force_download: bool=False,
    api_key: str | None=None,
    allow_download: bool=True,
) -> pd.DataFrame:
    """This function downloads or loads cached daily Stooq data.

    Params:
        symbol: Market ticker symbol.
        start: Optional start date.
        end: Optional end date.
        cache_dir: Local cache directory.
        force_download: Whether to replace cached data.
        api_key: Provider API key.
        allow_download: Whether remote downloads are allowed.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.lower().replace("/", "_")
    out_path = cache_path / f"stooq_{safe_symbol}.csv"

    api_key = api_key or os.getenv("STOOQ_API_KEY")
    df = None

    if out_path.exists() and not force_download:
        try:
            cached_text = out_path.read_text(encoding="utf-8", errors="replace")
            df = _read_market_csv_text(cached_text, source_name=f"cached Stooq file {out_path}")
        except ValueError:
            out_path.unlink(missing_ok=True)

    if df is None and not allow_download:
        raise FileNotFoundError(
            f"No valid cached Stooq CSV found at {out_path}. "
            "To avoid repeated web requests, either set local_path to a CSV file "
            "or rerun once with allow_download=True."
        )

    if df is None:
        url = f"https://stooq.com/q/d/l/?s={quote(symbol.lower())}&i=d"
        if api_key:
            url = f"{url}&apikey={quote(api_key)}"
        try:
            with urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 429:
                raise RuntimeError(
                    "Stooq returned HTTP 429 Too Many Requests. "
                    "Use a cached/local CSV, wait before retrying, or run once with a valid STOOQ_API_KEY."
                ) from exc
            raise
        except URLError as exc:
            raise RuntimeError(f"Stooq download failed: {exc}") from exc
        df = _read_market_csv_text(text, source_name=f"Stooq symbol {symbol}")
        out_path.write_text(text, encoding="utf-8")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        if start is not None:
            df = df[df["Date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["Date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def fetch_yahoo_daily(
    symbol: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    cache_dir: str | os.PathLike[str]="data/real",
    force_download: bool=False,
    allow_download: bool=True,
) -> pd.DataFrame:
    """This function downloads or loads cached daily Yahoo data.

    Params:
        symbol: Market ticker symbol.
        start: Optional start date.
        end: Optional end date.
        cache_dir: Local cache directory.
        force_download: Whether to replace cached data.
        allow_download: Whether remote downloads are allowed.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    safe_symbol = symbol.replace("/", "_")
    out_path = cache_path / f"yahoo_{safe_symbol}_{start_ts}_{end_ts}.csv"

    if out_path.exists() and not force_download:
        try:
            cached_text = out_path.read_text(encoding="utf-8", errors="replace")
            return _read_market_csv_text(cached_text, source_name=f"cached Yahoo file {out_path}")
        except ValueError:
            out_path.unlink(missing_ok=True)

    if not allow_download:
        raise FileNotFoundError(
            f"No valid cached Yahoo CSV found at {out_path}. "
            "To avoid repeated web requests, either set local_path to a CSV file "
            "or rerun once with allow_download=True."
        )

    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/"
        f"{quote(symbol)}?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
    )
    try:
        with urlopen(url, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "Yahoo returned HTTP 429 Too Many Requests. "
                "Use a cached/local CSV, wait before retrying, or rerun once later with allow_download=True."
            ) from exc
        raise
    except URLError as exc:
        raise RuntimeError(f"Yahoo download failed: {exc}") from exc
    df = _read_market_csv_text(text, source_name=f"Yahoo symbol {symbol}")
    out_path.write_text(text, encoding="utf-8")
    return df


def resolve_local_real_data_path(
    file_name: str="sp500_index.csv",
    local_path: str | os.PathLike[str] | None=None,
    search_dirs: tuple[str, ...] | list[str]=("data/real_data", "results/real_data", "data/real"),
) -> Path:
    """This function locates a manually downloaded market CSV.

    Params:
        file_name: Simulation or data file name.
        local_path: Optional local CSV path.
        search_dirs: Directories searched for local data.
    """
    if local_path:
        path = Path(local_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Local real-data file does not exist: {path}")

    if not file_name:
        raise ValueError("Provide local_path or file_name for local real data.")

    candidates = []
    for directory in search_dirs:
        candidate = Path(directory) / file_name
        candidates.append(candidate)
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find local real-data file '{file_name}'. Tried: {tried}")


def load_local_real_data(
    file_name: str="sp500_index.csv",
    local_path: str | os.PathLike[str] | None=None,
    search_dirs: tuple[str, ...] | list[str]=("data/real_data", "results/real_data", "data/real"),
    start: str | pd.Timestamp | None=None,
    end: str | pd.Timestamp | None=None,
) -> pd.DataFrame:
    """This function loads and filters a local market CSV.

    Params:
        file_name: Simulation or data file name.
        local_path: Optional local CSV path.
        search_dirs: Directories searched for local data.
        start: Optional start date.
        end: Optional end date.
    """
    path = resolve_local_real_data_path(
        file_name=file_name,
        local_path=local_path,
        search_dirs=search_dirs,
    )
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        if start is not None:
            df = df[df["Date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["Date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def load_or_fetch_real_data(
    symbol: Optional[str] = None,
    source: str = "local",
    start: str | pd.Timestamp | None=None,
    end: str | pd.Timestamp | None=None,
    local_path: str | os.PathLike[str] | None=None,
    file_name: str | None=None,
    search_dirs: tuple[str, ...] | list[str]=("data/real_data", "results/real_data", "data/real"),
    cache_dir: str | os.PathLike[str]="data/real",
    force_download: bool=False,
    api_key: str | None=None,
    allow_download: bool=True,
) -> pd.DataFrame:
    """This function loads market data from a local or remote source.

    Params:
        symbol: Market ticker symbol.
        source: Market-data source.
        start: Optional start date.
        end: Optional end date.
        local_path: Optional local CSV path.
        file_name: Simulation or data file name.
        search_dirs: Directories searched for local data.
        cache_dir: Local cache directory.
        force_download: Whether to replace cached data.
        api_key: Provider API key.
        allow_download: Whether remote downloads are allowed.
    """
    source = (source or "").lower()
    if source == "local" or local_path:
        return load_local_real_data(
            file_name=file_name or "sp500_index.csv",
            local_path=local_path,
            search_dirs=search_dirs,
            start=start,
            end=end,
        )

    if not symbol:
        raise ValueError("Provide either local_path or symbol.")

    if source == "stooq":
        return fetch_stooq_daily(
            symbol,
            start=start,
            end=end,
            cache_dir=cache_dir,
            force_download=force_download,
            api_key=api_key,
            allow_download=allow_download,
        )
    if source == "yahoo":
        if start is None or end is None:
            raise ValueError("Yahoo extraction requires start and end dates.")
        return fetch_yahoo_daily(
            symbol,
            start=start,
            end=end,
            cache_dir=cache_dir,
            force_download=force_download,
            allow_download=allow_download,
        )
    raise ValueError("source must be 'local', 'stooq', or 'yahoo'.")


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """This function returns the first candidate present in a column set.

    Params:
        columns: Available column names.
        candidates: Candidate column names.
    """
    columns = set(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def resolve_price_col(df: pd.DataFrame, price_col: Optional[str] = None) -> str:
    """This function identifies the price column in a market DataFrame.

    Params:
        df: Input DataFrame.
        price_col: Price column name.
    """
    if price_col and price_col in df.columns:
        return price_col
    candidate = _first_existing(
        df.columns,
        [
            "market_price",
            "close",
            "Close",
            "adj_close",
            "Adj Close",
            "price",
            "last",
            "S&P500",
            "S&P 500",
            "SP500",
            "sp500",
        ],
    )
    if candidate:
        return candidate
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric price column found.")
    return numeric_cols[0]


def prepare_price_frame(data: Any, date_col: Optional[str] = None, price_col: Optional[str] = None) -> pd.DataFrame:
    """This function standardizes dates and prices for analysis.

    Params:
        data: Input records or DataFrame.
        date_col: Date column name.
        price_col: Price column name.
    """
    df = _as_dataframe(data)
    if df.empty:
        return df

    price_col = resolve_price_col(df, price_col)
    out = df.copy()
    out["price"] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=["price"]).reset_index(drop=True)
    out = out[out["price"] > 0].reset_index(drop=True)

    if date_col is None:
        date_col = _first_existing(out.columns, ["date", "datetime", "timestamp", "Date", "market_datetime"])
    if date_col and date_col in out.columns:
        out["datetime"] = pd.to_datetime(out[date_col])
        out = out.sort_values("datetime").reset_index(drop=True)
    else:
        out["datetime"] = pd.RangeIndex(start=0, stop=len(out), step=1)

    if "market_time" in out.columns:
        out["tick"] = pd.to_numeric(out["market_time"], errors="coerce")
    else:
        out["tick"] = np.arange(len(out))

    out["log_return"] = np.log(out["price"]).diff()
    out["simple_return"] = out["price"].pct_change()
    return out


def split_real_data_for_backtest(
    real_data: Any,
    current_agent_knowledge: str | pd.Timestamp | None=None,
    holdout_steps: Optional[int] = None,
    date_col: Optional[str] = None,
    price_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This function creates known-history and future-holdout datasets.

    Params:
        real_data: Observed market data.
        current_agent_knowledge: Latest date known by the agent.
        holdout_steps: Reserved future observations.
        date_col: Date column name.
        price_col: Price column name.
    """
    df = prepare_price_frame(real_data, date_col=date_col, price_col=price_col)
    if df.empty:
        return df.copy(), df.copy()

    if current_agent_knowledge is None:
        cut_idx = max(1, int(len(df) * 0.8))
    elif isinstance(current_agent_knowledge, int):
        cut_idx = max(1, min(len(df), current_agent_knowledge))
    else:
        if not np.issubdtype(df["datetime"].dtype, np.datetime64):
            raise ValueError("Use an integer current_agent_knowledge when no datetime column exists.")
        knowledge_date = pd.to_datetime(current_agent_knowledge)
        cut_idx = int((df["datetime"] <= knowledge_date).sum())
        cut_idx = max(1, min(len(df), cut_idx))

    known = df.iloc[:cut_idx].copy()
    future = df.iloc[cut_idx:].copy()
    if holdout_steps is not None:
        future = future.iloc[:holdout_steps].copy()
    return known, future


def add_tick_time_index(
    simulated_data: Any,
    start_datetime: str | pd.Timestamp,
    end_datetime: str | pd.Timestamp | None=None,
    ticks_per_day: Optional[float] = None,
    trading_minutes_per_day: float = 390.0,
    tick_col: str = "market_time",
) -> pd.DataFrame:
    """This function maps artificial ticks to comparable timestamps.

    Params:
        simulated_data: Simulated market data.
        start_datetime: Start timestamp for tick mapping.
        end_datetime: End timestamp for tick mapping.
        ticks_per_day: Artificial ticks assigned to one day.
        trading_minutes_per_day: Trading minutes represented per day.
        tick_col: Simulation tick column name.
    """
    df = prepare_price_frame(simulated_data, price_col="market_price")
    if df.empty:
        return df

    if tick_col in df.columns:
        ticks = pd.to_numeric(df[tick_col], errors="coerce")
    else:
        ticks = pd.Series(np.arange(len(df)), index=df.index)
    ticks = ticks - ticks.min()

    start = pd.to_datetime(start_datetime)
    if end_datetime is not None:
        end = pd.to_datetime(end_datetime)
        span_seconds = max((end - start).total_seconds(), 0.0)
        denom = max(float(ticks.max()), 1.0)
        df["sim_datetime"] = start + pd.to_timedelta((ticks / denom) * span_seconds, unit="s")
        df["ticks_per_real_step"] = denom / max(len(df) - 1, 1)
    else:
        if ticks_per_day is None:
            ticks_per_day = max(float(ticks.max()), 1.0)
        minutes_per_tick = trading_minutes_per_day / ticks_per_day
        df["sim_datetime"] = start + pd.to_timedelta(ticks * minutes_per_tick, unit="m")
        df["ticks_per_day"] = ticks_per_day
        df["minutes_per_tick"] = minutes_per_tick

    return df


def align_simulation_to_real_holdout(
    real_future: pd.DataFrame,
    simulated_data: Any,
    real_price_col: Optional[str] = None,
    sim_price_col: Optional[str] = "market_price",
    scale_sim_to_real_start: bool = True,
) -> pd.DataFrame:
    """This function aligns simulated prices with real holdout observations.

    Params:
        real_future: Reserved real holdout data.
        simulated_data: Simulated market data.
        real_price_col: Real-data price column.
        sim_price_col: Simulation price column.
        scale_sim_to_real_start: Whether to align initial price levels.
    """
    real = prepare_price_frame(real_future, price_col=real_price_col)
    sim = prepare_price_frame(simulated_data, price_col=sim_price_col)
    if real.empty or sim.empty:
        return pd.DataFrame()

    real_x = np.linspace(0.0, 1.0, len(real))
    sim_x = np.linspace(0.0, 1.0, len(sim))
    sim_prices = np.interp(real_x, sim_x, sim["price"].to_numpy(dtype=float))
    if scale_sim_to_real_start and sim_prices[0] != 0:
        sim_prices = sim_prices * (real["price"].iloc[0] / sim_prices[0])

    aligned = pd.DataFrame({
        "datetime": real["datetime"].to_numpy(),
        "real_price": real["price"].to_numpy(dtype=float),
        "sim_price": sim_prices,
    })
    aligned["real_return"] = np.log(aligned["real_price"]).diff()
    aligned["sim_return"] = np.log(aligned["sim_price"]).diff()
    aligned["abs_error"] = (aligned["sim_price"] - aligned["real_price"]).abs()
    aligned["pct_error"] = aligned["abs_error"] / aligned["real_price"].replace(0, np.nan)
    aligned.attrs["ticks_per_real_observation"] = max(len(sim) - 1, 1) / max(len(real) - 1, 1)
    return aligned


def _safe_corr(left: Any, right: Any) -> float:
    """This function calculates correlation when enough valid data exists.

    Params:
        left: First numeric series.
        right: Second numeric series.
    """
    left = pd.Series(left).astype(float)
    right = pd.Series(right).astype(float)
    valid = left.notna() & right.notna()
    if valid.sum() < 2:
        return np.nan
    if left[valid].std() == 0 or right[valid].std() == 0:
        return np.nan
    return float(left[valid].corr(right[valid]))


def _scientific_error_metrics(aligned_df: pd.DataFrame) -> Dict[str, float]:
    """This function calculates scientific errors for aligned price series.

    Params:
        aligned_df: Aligned real and simulated observations.
    """
    real = aligned_df["real_price"].astype(float)
    sim = aligned_df["sim_price"].astype(float)
    errors = sim - real
    abs_errors = errors.abs()
    squared_errors = np.square(errors)

    real_range = real.max() - real.min()
    real_mean_abs = real.abs().mean()
    real_start = abs(real.iloc[0]) if len(real) else np.nan
    denominator = (real.abs() + sim.abs()).replace(0, np.nan)

    naive = real.shift(1)
    if len(real):
        naive.iloc[0] = real.iloc[0]
    naive_errors = naive - real
    naive_rmse = float(np.sqrt(np.nanmean(np.square(naive_errors)))) if len(real) > 1 else np.nan
    model_rmse = float(np.sqrt(np.nanmean(squared_errors)))
    theil_u = model_rmse / naive_rmse if naive_rmse and np.isfinite(naive_rmse) else np.nan

    metrics = {
        "mean_error_bias": float(errors.mean()),
        "median_absolute_error": float(abs_errors.median()),
        "nrmse_mean_price": float(model_rmse / real_mean_abs) if real_mean_abs else np.nan,
        "nrmse_price_range": float(model_rmse / real_range) if real_range else np.nan,
        "smape": float((2.0 * abs_errors / denominator).mean()),
        "wape": float(abs_errors.sum() / real.abs().sum()) if real.abs().sum() else np.nan,
        "mae_pct_of_start": float(abs_errors.mean() / real_start) if real_start else np.nan,
        "naive_random_walk_rmse": naive_rmse,
        "theil_u_vs_random_walk": float(theil_u) if np.isfinite(theil_u) else np.nan,
    }

    real_returns = aligned_df["real_return"].astype(float)
    sim_returns = aligned_df["sim_return"].astype(float)
    valid_returns = real_returns.notna() & sim_returns.notna()
    if valid_returns.any():
        return_errors = sim_returns[valid_returns] - real_returns[valid_returns]
        metrics.update({
            "return_mae": float(return_errors.abs().mean()),
            "return_rmse": float(np.sqrt(np.mean(np.square(return_errors)))),
            "return_bias": float(return_errors.mean()),
        })

    return metrics


def comparison_metrics(aligned_df: pd.DataFrame) -> Dict[str, float]:
    """This function summarizes real-versus-simulated comparison metrics.

    Params:
        aligned_df: Aligned real and simulated observations.
    """
    if aligned_df.empty:
        return {}

    errors = aligned_df["sim_price"] - aligned_df["real_price"]
    real_returns = aligned_df["real_return"].dropna()
    sim_returns = aligned_df["sim_return"].dropna()
    n_returns = min(len(real_returns), len(sim_returns))

    metrics = {
        "n_observations": float(len(aligned_df)),
        "ticks_per_real_observation": float(aligned_df.attrs.get("ticks_per_real_observation", np.nan)),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float(aligned_df["pct_error"].mean()),
        "price_correlation": _safe_corr(aligned_df["real_price"], aligned_df["sim_price"]),
    }
    metrics.update(_scientific_error_metrics(aligned_df))

    if n_returns > 0:
        rr = real_returns.iloc[:n_returns].to_numpy()
        sr = sim_returns.iloc[:n_returns].to_numpy()
        metrics["return_rmse"] = float(np.sqrt(np.mean(np.square(sr - rr))))
        metrics["return_correlation"] = _safe_corr(rr, sr)
        metrics["directional_accuracy"] = float(np.mean(np.sign(rr) == np.sign(sr)))
    return metrics


def compare_real_vs_simulated(
    real_data: Any,
    simulated_data: Any,
    current_agent_knowledge: str | pd.Timestamp | None=None,
    holdout_steps: Optional[int] = None,
    real_price_col: Optional[str] = None,
    sim_price_col: Optional[str] = "market_price",
    date_col: Optional[str] = None,
    scale_sim_to_real_start: bool = True,
) -> dict[str, Any]:
    """This function aligns and compares a simulation with future real data.

    Params:
        real_data: Observed market data.
        simulated_data: Simulated market data.
        current_agent_knowledge: Latest date known by the agent.
        holdout_steps: Reserved future observations.
        real_price_col: Real-data price column.
        sim_price_col: Simulation price column.
        date_col: Date column name.
        scale_sim_to_real_start: Whether to align initial price levels.
    """
    known, future = split_real_data_for_backtest(
        real_data,
        current_agent_knowledge=current_agent_knowledge,
        holdout_steps=holdout_steps,
        date_col=date_col,
        price_col=real_price_col,
    )
    aligned = align_simulation_to_real_holdout(
        future,
        simulated_data,
        real_price_col=real_price_col,
        sim_price_col=sim_price_col,
        scale_sim_to_real_start=scale_sim_to_real_start,
    )
    return {
        "known_real": known,
        "future_real": future,
        "aligned": aligned,
        "metrics": comparison_metrics(aligned),
    }


def compute_all_time_high_nearness(data: Any, price_col: Optional[str] = None) -> pd.DataFrame:
    """This function calculates price proximity to the running maximum.

    Params:
        data: Input records or DataFrame.
        price_col: Price column name.
    """
    df = prepare_price_frame(data, price_col=price_col)
    if df.empty:
        return df
    df["all_time_high"] = df["price"].cummax()
    df["all_time_high_nearness"] = df["price"] / df["all_time_high"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _resolved_horizon_shift(horizon: int, rows_per_day: int, n_obs: int, auto_rescale: bool=True) -> tuple[int, int, str]:
    """This function converts a horizon into a valid row shift.

    Params:
        horizon: Forecast horizon.
        rows_per_day: Rows representing one real day.
        n_obs: Number of available observations.
        auto_rescale: Whether to reduce oversized horizons.
    """
    requested_shift_rows = max(1, int(round(horizon * rows_per_day)))
    if requested_shift_rows < max(n_obs - 2, 1):
        return requested_shift_rows, requested_shift_rows, "days"
    if not auto_rescale:
        return requested_shift_rows, requested_shift_rows, "days"

    fallback_shift = max(1, int(round(horizon)))
    fallback_shift = min(fallback_shift, max(n_obs - 3, 1))
    return fallback_shift, requested_shift_rows, "rows_fallback"


def detect_historic_maxima(data: Any, price_col: Optional[str] = None, strict_new_high: bool = True) -> Dict[str, object]:
    """This function identifies simulated historical price maxima.

    Params:
        data: Input records or DataFrame.
        price_col: Price column name.
        strict_new_high: Whether only new records count as maxima.
    """
    df = compute_all_time_high_nearness(data, price_col=price_col)
    if df.empty:
        return {"has_data": False, "maxima": pd.DataFrame()}

    previous_high = df["all_time_high"].shift(1)
    if strict_new_high:
        is_new_high = previous_high.isna() | (df["price"] > previous_high)
    else:
        is_new_high = df["price"] >= df["all_time_high"]
    maxima = df[is_new_high].copy()
    peak_idx = df["price"].idxmax()
    return {
        "has_data": True,
        "has_historic_maximum": bool(len(maxima) > 0),
        "peak_price": float(df.loc[peak_idx, "price"]),
        "peak_tick": float(df.loc[peak_idx, "tick"]),
        "peak_datetime": df.loc[peak_idx, "datetime"],
        "n_new_highs": int(len(maxima)),
        "maxima": maxima,
    }


def ols_all_time_high_beta(
    data: Any,
    horizons: tuple[int, ...] | list[int]=(10, 15, 30),
    rows_per_day: float = 1.0,
    price_col: Optional[str] = None,
    auto_rescale: bool = True,
) -> pd.DataFrame:
    """This function estimates the all-time-high regression coefficient by horizon.

    Params:
        data: Input records or DataFrame.
        horizons: Horizons evaluated by the regression.
        rows_per_day: Rows representing one real day.
        price_col: Price column name.
        auto_rescale: Whether to reduce oversized horizons.
    """
    df = compute_all_time_high_nearness(data, price_col=price_col)
    df = df.dropna(subset=["price", "all_time_high_nearness"]).reset_index(drop=True)
    rows = []
    for horizon in horizons:
        shift_rows, requested_shift_rows, horizon_units = _resolved_horizon_shift(
            horizon=horizon,
            rows_per_day=rows_per_day,
            n_obs=len(df),
            auto_rescale=auto_rescale,
        )
        work = df.copy()
        work["future_gross_return"] = work["price"].shift(-shift_rows) / work["price"]
        work = work.replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=["future_gross_return", "all_time_high_nearness"])
        if len(work) < 3:
            rows.append({
                "horizon": horizon,
                "shift_rows": shift_rows,
                "requested_shift_rows": requested_shift_rows,
                "horizon_units": horizon_units,
                "beta_h": np.nan,
                "alpha": np.nan,
                "r2": np.nan,
                "p_value": np.nan,
                "n": len(work),
                "note": "not enough observations",
            })
            continue

        x = work["all_time_high_nearness"].to_numpy(dtype=float)
        y = work["future_gross_return"].to_numpy(dtype=float)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]
        if len(x) < 3:
            rows.append({
                "horizon": horizon,
                "shift_rows": shift_rows,
                "requested_shift_rows": requested_shift_rows,
                "horizon_units": horizon_units,
                "beta_h": np.nan,
                "alpha": np.nan,
                "r2": np.nan,
                "p_value": np.nan,
                "n": len(x),
                "note": "not enough finite observations",
            })
            continue
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            rows.append({
                "horizon": horizon,
                "shift_rows": shift_rows,
                "requested_shift_rows": requested_shift_rows,
                "horizon_units": horizon_units,
                "beta_h": np.nan,
                "alpha": float(np.nanmean(y)) if len(y) else np.nan,
                "r2": np.nan,
                "p_value": np.nan,
                "n": len(x),
                "note": "constant regressor or target",
            })
            continue

        if stats is not None:
            reg = stats.linregress(x, y)
            beta = reg.slope
            alpha = reg.intercept
            r2 = reg.rvalue ** 2
            p_value = reg.pvalue
        else:
            beta, alpha = np.polyfit(x, y, 1)
            y_hat = alpha + beta * x
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
            p_value = np.nan

        rows.append({
            "horizon": horizon,
            "shift_rows": shift_rows,
            "requested_shift_rows": requested_shift_rows,
            "horizon_units": horizon_units,
            "beta_h": float(beta),
            "alpha": float(alpha),
            "r2": float(r2),
            "p_value": float(p_value) if p_value is not None else np.nan,
            "n": len(x),
            "note": "",
        })
    return pd.DataFrame(rows)


def stylized_facts(
    data: Any,
    price_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    lags: tuple[int, ...] | list[int]=(1, 5, 10),
) -> Dict[str, float]:
    """This function calculates core financial-market stylized facts.

    Params:
        data: Input records or DataFrame.
        price_col: Price column name.
        volume_col: Volume column name.
        lags: Autocorrelation lags.
    """
    df = prepare_price_frame(data, price_col=price_col)
    if df.empty:
        return {}

    returns = df["log_return"].dropna()
    abs_returns = returns.abs()
    out = {
        "kurtosis_excess": float(returns.kurtosis()) if len(returns) > 3 else np.nan,
    }
    for lag in lags:
        out[f"abs_return_autocorr_{lag}"] = float(abs_returns.autocorr(lag=lag)) if len(abs_returns) > lag else np.nan

    if volume_col and volume_col in df.columns:
        aligned = pd.DataFrame({
            "abs_return": df["log_return"].abs(),
            "volume": pd.to_numeric(df[volume_col], errors="coerce"),
        }).dropna()
        out["abs_return_volume_corr"] = float(aligned["abs_return"].corr(aligned["volume"])) if len(aligned) > 2 else np.nan
    else:
        out["abs_return_volume_corr"] = np.nan
    return out


def orders_with_all_time_high_nearness(
    orders_data: Any,
    market_data: Any,
    agent_type: Optional[str] = "auto",
) -> pd.DataFrame:
    """This function attaches all-time-high proximity to submitted orders.

    Params:
        orders_data: Submitted order records.
        market_data: Simulated market records.
        agent_type: Agent class filter.
    """
    orders = _as_dataframe(orders_data)
    market = compute_all_time_high_nearness(market_data, price_col="market_price")
    if orders.empty or market.empty:
        return pd.DataFrame()

    orders = orders.copy()
    if "is_buy" in orders.columns and orders["is_buy"].dtype != bool:
        orders["is_buy"] = orders["is_buy"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "buy", "b"}
        )

    selected_agent_type = agent_type
    if agent_type == "auto":
        if "agent_type" in orders.columns and (orders["agent_type"] == "FCLAgent").any():
            selected_agent_type = "FCLAgent"
        else:
            selected_agent_type = None

    if selected_agent_type and "agent_type" in orders.columns:
        filtered = orders[orders["agent_type"] == selected_agent_type].copy()
        if not filtered.empty:
            orders = filtered

    market_cols = ["market_time", "market_id", "price", "all_time_high", "all_time_high_nearness"]
    market_view = market[[c for c in market_cols if c in market.columns]].copy()
    if "market_time" not in market_view.columns:
        market_view["market_time"] = market["tick"]

    for col in ["market_time", "market_id"]:
        if col in orders.columns:
            orders[col] = pd.to_numeric(orders[col], errors="coerce")
        if col in market_view.columns:
            market_view[col] = pd.to_numeric(market_view[col], errors="coerce")

    merge_keys = ["market_time"]
    if "market_id" in orders.columns and "market_id" in market_view.columns:
        merge_keys.append("market_id")

    merged = orders.merge(market_view, on=merge_keys, how="left", suffixes=("_order", "_market"))
    if "all_time_high_nearness" not in merged.columns or merged["all_time_high_nearness"].isna().all():
        merged = orders.merge(market_view.drop(columns=["market_id"], errors="ignore"), on="market_time", how="left")
    merged["selected_agent_type"] = selected_agent_type or "all"
    return merged


def asset_proportion_summary(portfolio_data: Any, agent_type: Optional[str] = "FCLAgent") -> Dict[str, float]:
    """This function summarizes risky-asset allocation across portfolios.

    Params:
        portfolio_data: Agent portfolio records.
        agent_type: Agent class filter.
    """
    df = _as_dataframe(portfolio_data)
    if df.empty:
        return {}
    if agent_type and "agent_type" in df.columns:
        df = df[df["agent_type"] == agent_type]
    series = pd.to_numeric(df["asset_proportion"], errors="coerce").dropna()
    if series.empty:
        return {}
    return {
        "n": float(len(series)),
        "p01": float(series.quantile(0.01)),
        "p50": float(series.quantile(0.50)),
        "p99": float(series.quantile(0.99)),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }


def order_nearness_tests(order_nearness_df: pd.DataFrame) -> Dict[str, float]:
    """This function compares buy and sell proximity distributions.

    Params:
        order_nearness_df: Orders with ATH proximity values.
    """
    if order_nearness_df.empty:
        return {}
    df = order_nearness_df.copy()
    if "is_buy" in df.columns and df["is_buy"].dtype != bool:
        df["is_buy"] = df["is_buy"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "buy", "b"}
        )
    buys = df[df["is_buy"] == True]["all_time_high_nearness"].dropna()
    sells = df[df["is_buy"] == False]["all_time_high_nearness"].dropna()
    out = {
        "n_buy": float(len(buys)),
        "n_sell": float(len(sells)),
        "buy_mean_nearness": float(buys.mean()) if len(buys) else np.nan,
        "sell_mean_nearness": float(sells.mean()) if len(sells) else np.nan,
        "buy_at_exact_high": float((buys == 1.0).sum()),
        "sell_at_exact_high": float((sells == 1.0).sum()),
    }
    if stats is not None and len(buys) > 1 and len(sells) > 1:
        out["ks_p_value"] = float(stats.ks_2samp(buys, sells).pvalue)
        out["mannwhitney_p_value"] = float(stats.mannwhitneyu(buys, sells, alternative="two-sided").pvalue)
    return out


def plot_real_vs_simulated(aligned_df: pd.DataFrame, title: str="Real vs simulated prices") -> tuple[Any, Any]:
    """This function plots aligned real and simulated prices.

    Params:
        aligned_df: Aligned real and simulated observations.
        title: Plot title.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(aligned_df["datetime"], aligned_df["real_price"], label="Real", linewidth=1.8)
    ax.plot(aligned_df["datetime"], aligned_df["sim_price"], label="Simulated", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Comparable time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_simulated_historic_maximum(simulated_data: Any, price_col: Optional[str] = "market_price") -> tuple[Any, Any, dict[str, Any]]:
    """This function plots the simulated path and its maximum.

    Params:
        simulated_data: Simulated market data.
        price_col: Price column name.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    report = detect_historic_maxima(simulated_data, price_col=price_col)
    df = compute_all_time_high_nearness(simulated_data, price_col=price_col)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["tick"], df["price"], label="Simulated price", linewidth=1.5)
    if report.get("has_data"):
        maxima = report["maxima"]
        ax.scatter(maxima["tick"], maxima["price"], color="crimson", s=24, label="New all-time high")
        ax.axhline(report["peak_price"], color="crimson", linestyle="--", linewidth=1, label="Historic maximum")
    ax.set_title("Simulated price and historic maxima")
    ax.set_xlabel("Simulation ticks")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax, report


def plot_asset_proportion_histogram(portfolio_data: Any, agent_type: Optional[str] = "FCLAgent", bins: int=60) -> tuple[Any, Any, dict[str, float]]:
    """This function plots the distribution of portfolio asset proportions.

    Params:
        portfolio_data: Agent portfolio records.
        agent_type: Agent class filter.
        bins: Histogram bin count.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    df = _as_dataframe(portfolio_data)
    if agent_type and "agent_type" in df.columns:
        df = df[df["agent_type"] == agent_type]
    values = pd.to_numeric(df["asset_proportion"], errors="coerce").dropna()
    summary = asset_proportion_summary(df, agent_type=None)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=bins, color="0.25", alpha=0.9)
    for key, color, label in [("p01", "blue", "1st percentile"), ("p50", "orange", "50th percentile"), ("p99", "green", "99th percentile")]:
        if key in summary:
            ax.axvline(summary[key], linestyle="--", color=color, linewidth=1, label=label)
    ax.set_xlabel("PA_t")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig, ax, summary


def plot_all_time_high_nearness_histogram(order_nearness_df: pd.DataFrame, bins: int=50) -> tuple[Any, Any, dict[str, float]]:
    """This function plots buy and sell proximity to the all-time high.

    Params:
        order_nearness_df: Orders with ATH proximity values.
        bins: Histogram bin count.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    df = order_nearness_df.copy()
    if "is_buy" in df.columns and df["is_buy"].dtype != bool:
        df["is_buy"] = df["is_buy"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "buy", "b"}
        )
    buys = df[df["is_buy"] == True]["all_time_high_nearness"].dropna()
    sells = df[df["is_buy"] == False]["all_time_high_nearness"].dropna()
    tests = order_nearness_tests(order_nearness_df)

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(buys):
        ax.hist(buys, bins=bins, alpha=0.55, color="#2ca02c", label="buy")
    if len(sells):
        ax.hist(sells, bins=bins, alpha=0.55, color="#ff6b6b", label="sell")
    if not len(buys) and not len(sells):
        ax.text(0.5, 0.5, "No buy/sell orders available", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("p_t / p^h_1:t")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig, ax, tests


def reference_paper_report(
    simulated_market_data: Any,
    real_data: Any=None,
    comparison_simulated_market_data: Any | None=None,
    orders_data: Any=None,
    portfolio_data: Any=None,
    executions_data: Any | None=None,
    current_agent_knowledge: str | pd.Timestamp | None=None,
    holdout_steps: Optional[int] = None,
    rows_per_day: float = 1.0,
    real_price_col: Optional[str] = None,
    real_date_col: Optional[str] = None,
    sim_price_col: Optional[str] = "market_price",
    volume_col: Optional[str] = None,
) -> dict[str, Any]:
    """This function builds the paper-style metrics for one simulation.

    Params:
        simulated_market_data: Simulated market records.
        real_data: Observed market data.
        comparison_simulated_market_data: Simulation window used for real-data comparison.
        orders_data: Submitted order records.
        portfolio_data: Agent portfolio records.
        executions_data: Executed trade records.
        current_agent_knowledge: Latest date known by the agent.
        holdout_steps: Reserved future observations.
        rows_per_day: Rows representing one real day.
        real_price_col: Real-data price column.
        real_date_col: Real-data date column.
        sim_price_col: Simulation price column.
        volume_col: Volume column name.
    """
    report = {
        "sim_historic_maximum": detect_historic_maxima(simulated_market_data, price_col=sim_price_col),
        "sim_beta_h": ols_all_time_high_beta(simulated_market_data, rows_per_day=rows_per_day, price_col=sim_price_col),
        "sim_stylized_facts": stylized_facts(simulated_market_data, price_col=sim_price_col, volume_col=volume_col),
    }

    if real_data is not None:
        comparison_market = (
            comparison_simulated_market_data
            if comparison_simulated_market_data is not None
            else simulated_market_data
        )
        comparison = compare_real_vs_simulated(
            real_data=real_data,
            simulated_data=comparison_market,
            current_agent_knowledge=current_agent_knowledge,
            holdout_steps=holdout_steps,
            real_price_col=real_price_col,
            sim_price_col=sim_price_col,
            date_col=real_date_col,
        )
        report["real_vs_simulated"] = comparison
        report["validation_metrics"] = comparison["metrics"]
        report["real_beta_h"] = ols_all_time_high_beta(
            comparison["known_real"],
            rows_per_day=rows_per_day,
            price_col="price",
        )
        report["real_stylized_facts"] = stylized_facts(
            comparison["known_real"],
            price_col="price",
            volume_col=volume_col,
        )

    if orders_data is not None:
        order_nearness = orders_with_all_time_high_nearness(orders_data, simulated_market_data)
        report["order_nearness"] = order_nearness
        report["order_nearness_tests"] = order_nearness_tests(order_nearness)

    if portfolio_data is not None:
        report["asset_proportion_summary"] = asset_proportion_summary(portfolio_data)

    if executions_data is not None and volume_col is None:
        executions = _as_dataframe(executions_data)
        if not executions.empty and {"market_time", "volume"}.issubset(executions.columns):
            volume_by_tick = executions.groupby("market_time", as_index=False)["volume"].sum()
            market = _as_dataframe(simulated_market_data).merge(volume_by_tick, on="market_time", how="left")
            market["volume"] = market["volume"].fillna(0)
            report["sim_stylized_facts_with_execution_volume"] = stylized_facts(
                market,
                price_col=sim_price_col,
                volume_col="volume",
            )

    return report


def list_market_simulations(result_root: str | os.PathLike[str]="results") -> pd.DataFrame:
    """This function lists saved simulations available for analysis.

    Params:
        result_root: Root directory containing results.
    """
    result_root = Path(result_root)
    rows = []
    for sim_type in ["classic", "llms"]:
        base_dir = result_root / sim_type
        if not base_dir.exists():
            continue
        for market_path in sorted(base_dir.glob("*_base_results_fcn.csv")):
            file_name = market_path.name.replace("_base_results_fcn.csv", "")
            rows.append({
                "sim_type": sim_type,
                "file_name": file_name,
                "market_path": str(market_path),
                "has_orders": (base_dir / f"{file_name}_orders.csv").exists(),
                "has_executions": (base_dir / f"{file_name}_executions.csv").exists(),
                "has_portfolios": (base_dir / f"{file_name}_agent_portfolios.csv").exists(),
            })
    return pd.DataFrame(rows)


def load_simulation_artifacts(sim_type: str, file_name: str, result_root: str | os.PathLike[str]="results") -> Dict[str, object]:
    """This function loads all CSV artifacts for one simulation.

    Params:
        sim_type: Simulation type: classic or llms.
        file_name: Simulation or data file name.
        result_root: Root directory containing results.
    """
    base_dir = Path(result_root) / sim_type
    paths = {
        "market": base_dir / f"{file_name}_base_results_fcn.csv",
        "orders": base_dir / f"{file_name}_orders.csv",
        "executions": base_dir / f"{file_name}_executions.csv",
        "portfolios": base_dir / f"{file_name}_agent_portfolios.csv",
    }
    if not paths["market"].exists():
        raise FileNotFoundError(f"Market result not found: {paths['market']}")

    return {
        "sim_type": sim_type,
        "file_name": file_name,
        "paths": paths,
        "market": pd.read_csv(paths["market"]),
        "orders": pd.read_csv(paths["orders"]) if paths["orders"].exists() else None,
        "executions": pd.read_csv(paths["executions"]) if paths["executions"].exists() else None,
        "portfolios": pd.read_csv(paths["portfolios"]) if paths["portfolios"].exists() else None,
    }


def simulated_trading_window(simulated_market_data: Any, warmup_ticks: int=0) -> pd.DataFrame:
    """This function returns simulated observations after warm-up.

    Params:
        simulated_market_data: Simulated market records.
        warmup_ticks: Ticks removed as warm-up.
    """
    df = _as_dataframe(simulated_market_data)
    if df.empty or "market_time" not in df.columns:
        return df
    market_time = pd.to_numeric(df["market_time"], errors="coerce")
    return df[market_time >= int(warmup_ticks)].reset_index(drop=True)


def analyze_simulation(
    sim_type: str,
    file_name: str,
    rows_per_day: int=1,
    real_data: Any=None,
    real_data_path: str | os.PathLike[str] | None=None,
    current_agent_knowledge: str | pd.Timestamp | None=None,
    holdout_steps: int | None=None,
    real_price_col: str | None=None,
    real_date_col: str | None=None,
    sim_price_col: str | None="market_price",
    comparison_market_data: Any | None=None,
    comparison_warmup_steps: int | None=None,
    result_root: str | os.PathLike[str]="results",
    print_summary: bool=True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """This function loads and analyzes one classical or hybrid simulation.

    Params:
        sim_type: Simulation type: classic or llms.
        file_name: Simulation or data file name.
        rows_per_day: Rows representing one real day.
        real_data: Observed market data.
        real_data_path: Optional real-data CSV path.
        current_agent_knowledge: Latest date known by the agent.
        holdout_steps: Reserved future observations.
        real_price_col: Real-data price column.
        real_date_col: Real-data date column.
        sim_price_col: Simulation price column.
        comparison_market_data: Optional post-warm-up market data.
        comparison_warmup_steps: Warm-up ticks removed before comparison.
        result_root: Root directory containing results.
        print_summary: Whether to print row counts.
    """
    artifacts = load_simulation_artifacts(sim_type, file_name, result_root=result_root)
    if real_data is None and real_data_path:
        real_data = pd.read_csv(real_data_path)

    if comparison_market_data is None and comparison_warmup_steps is not None:
        comparison_market_data = simulated_trading_window(
            artifacts["market"],
            warmup_ticks=comparison_warmup_steps,
        )

    report = reference_paper_report(
        simulated_market_data=artifacts["market"],
        comparison_simulated_market_data=comparison_market_data,
        real_data=real_data,
        orders_data=artifacts["orders"],
        portfolio_data=artifacts["portfolios"],
        executions_data=artifacts["executions"],
        current_agent_knowledge=current_agent_knowledge,
        holdout_steps=holdout_steps,
        rows_per_day=rows_per_day,
        real_price_col=real_price_col,
        real_date_col=real_date_col,
        sim_price_col=sim_price_col,
    )

    if print_summary:
        print(f"Analysis target: {sim_type}/{file_name}")
        print(f"Market rows: {len(artifacts['market'])}")
        print(f"Orders rows: {0 if artifacts['orders'] is None else len(artifacts['orders'])}")
        print(f"Executions rows: {0 if artifacts['executions'] is None else len(artifacts['executions'])}")
        print(f"Portfolio rows: {0 if artifacts['portfolios'] is None else len(artifacts['portfolios'])}")
        if comparison_market_data is not None:
            print(f"Comparison market rows after warmup trim: {len(comparison_market_data)}")

    return report, artifacts
def _safe_get(dct: dict[str, Any] | None, key: str, default: Any=np.nan) -> Any:
    """This function reads a dictionary value with a safe default.

    Params:
        dct: Source dictionary.
        key: Dictionary key.
        default: Fallback value.
    """
    return dct.get(key, default) if isinstance(dct, dict) else default

#This functions measures the mean squared error between the simulated and real beta_h values across horizons, handling various edge cases gracefully.
#Used for the llm paper's comparison of simulated vs real beta_h estimates.
def _beta_h_mse(report: dict[str, Any]) -> float:
    """This function calculates error between real and simulated ATH coefficients.

    Params:
        report: Simulation analysis report.
    """
    sim_beta = report.get("sim_beta_h")
    real_beta = report.get("real_beta_h")

    if not isinstance(sim_beta, pd.DataFrame) or not isinstance(real_beta, pd.DataFrame):
        return np.nan

    if sim_beta.empty or real_beta.empty:
        return np.nan

    merged = sim_beta[["horizon", "beta_h"]].merge(
        real_beta[["horizon", "beta_h"]],
        on="horizon",
        suffixes=("_sim", "_real"),
    )

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta_h_sim", "beta_h_real"])

    if merged.empty:
        return np.nan

    return float(np.mean((merged["beta_h_sim"] - merged["beta_h_real"]) ** 2))


def _return_distribution_distances(aligned_df: pd.DataFrame, bins: int=40, eps: float=1e-12) -> dict[str, float]:
    """This function compares real and simulated return distributions.

    Params:
        aligned_df: Aligned real and simulated observations.
        bins: Histogram bin count.
        eps: Small probability stabilizer.
    """
    if aligned_df is None or aligned_df.empty:
        return {"kl_divergence": np.nan, "hellinger_distance": np.nan}

    real_returns = pd.to_numeric(aligned_df["real_return"], errors="coerce").dropna()
    sim_returns = pd.to_numeric(aligned_df["sim_return"], errors="coerce").dropna()

    n = min(len(real_returns), len(sim_returns))
    if n < 5:
        return {"kl_divergence": np.nan, "hellinger_distance": np.nan}

    real_returns = real_returns.iloc[:n]
    sim_returns = sim_returns.iloc[:n]

    low = min(real_returns.min(), sim_returns.min())
    high = max(real_returns.max(), sim_returns.max())

    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return {"kl_divergence": np.nan, "hellinger_distance": np.nan}

    real_hist, edges = np.histogram(real_returns, bins=bins, range=(low, high), density=False)
    sim_hist, _ = np.histogram(sim_returns, bins=edges, density=False)

    p = real_hist.astype(float) + eps
    q = sim_hist.astype(float) + eps

    p = p / p.sum()
    q = q / q.sum()

    kl = float(np.sum(p * np.log(p / q)))
    hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    return {
        "kl_divergence": kl,
        "hellinger_distance": hellinger,
    }


def _llm_order_flow_metrics(artifacts: dict[str, Any], report: dict[str, Any]) -> dict[str, float]:
    """This function calculates LLM order-flow and portfolio metrics.

    Params:
        artifacts: Loaded simulation artifacts.
        report: Simulation analysis report.
    """
    orders = artifacts.get("orders")
    executions = artifacts.get("executions")
    market = artifacts.get("market")

    metrics = {
        "llm_orders": np.nan,
        "llm_buy_orders": np.nan,
        "llm_sell_orders": np.nan,
        "llm_order_flow_imbalance": np.nan,
        "ks_p_value_ath_nearness_buy_vs_sell": _safe_get(report.get("order_nearness_tests"), "ks_p_value"),
        "mannwhitney_p_value_buy_vs_sell": _safe_get(report.get("order_nearness_tests"), "mannwhitney_p_value"),
        "portfolio_p01": _safe_get(report.get("asset_proportion_summary"), "p01"),
        "portfolio_p50": _safe_get(report.get("asset_proportion_summary"), "p50"),
        "portfolio_p99": _safe_get(report.get("asset_proportion_summary"), "p99"),
        "portfolio_stability_width_p99_minus_p01": np.nan,
        "odean_proxy_pgr_minus_plr": np.nan,
    }

    if isinstance(orders, pd.DataFrame) and not orders.empty:
        llm_orders = orders.copy()

        if "agent_type" in llm_orders.columns:
            llm_orders = llm_orders[
                llm_orders["agent_type"].astype(str).str.contains("FCL|LLM", case=False, regex=True, na=False)
            ]

        if not llm_orders.empty and "is_buy" in llm_orders.columns:
            is_buy = llm_orders["is_buy"].map(
                lambda x: str(x).strip().lower() in {"true", "1", "buy", "b"}
            )
            buy_count = int(is_buy.sum())
            sell_count = int((~is_buy).sum())
            total = buy_count + sell_count

            metrics["llm_orders"] = total
            metrics["llm_buy_orders"] = buy_count
            metrics["llm_sell_orders"] = sell_count
            metrics["llm_order_flow_imbalance"] = (
                (buy_count - sell_count) / total if total else np.nan
            )

    p01 = metrics["portfolio_p01"]
    p99 = metrics["portfolio_p99"]
    if pd.notna(p01) and pd.notna(p99):
        metrics["portfolio_stability_width_p99_minus_p01"] = float(p99 - p01)

    if (
        isinstance(executions, pd.DataFrame)
        and not executions.empty
        and isinstance(market, pd.DataFrame)
        and not market.empty
        and {"market_time", "market_price"}.issubset(market.columns)
    ):
        sells = executions.copy()

        if "sell_agent_type" in sells.columns:
            sells = sells[
                sells["sell_agent_type"].astype(str).str.contains("FCL|LLM", case=False, regex=True, na=False)
            ]

        if not sells.empty and {"market_time", "price"}.issubset(sells.columns):
            market_ref = market[["market_time", "market_price"]].copy()
            market_ref["previous_market_price"] = market_ref["market_price"].shift(1)

            sells = sells.merge(
                market_ref[["market_time", "previous_market_price"]],
                on="market_time",
                how="left",
            )

            sells["price"] = pd.to_numeric(sells["price"], errors="coerce")
            sells["previous_market_price"] = pd.to_numeric(
                sells["previous_market_price"],
                errors="coerce",
            )

            valid = sells.dropna(subset=["price", "previous_market_price"])

            if not valid.empty:
                realized_gains = int((valid["price"] > valid["previous_market_price"]).sum())
                realized_losses = int((valid["price"] < valid["previous_market_price"]).sum())
                total_realized = realized_gains + realized_losses

                if total_realized > 0:
                    pgr = realized_gains / total_realized
                    plr = realized_losses / total_realized
                    metrics["odean_proxy_pgr_minus_plr"] = float(pgr - plr)

    return metrics
def macro_table(sim_facts: dict[str, Any], sim_facts_with_volume: dict[str, Any], historic_max: dict[str, Any], paper_report: dict[str, Any]) -> pd.DataFrame:
    """This function formats macro-level validation metrics as a table.

    Params:
        sim_facts: Simulation stylized facts.
        sim_facts_with_volume: Stylized facts using execution volume.
        historic_max: Historical-maximum summary.
        paper_report: Paper-style simulation report.
    """
    macro_market_validation = pd.DataFrame([
        {
            "metric": "fat_tails_excess_kurtosis",
            "value": _safe_get(sim_facts, "kurtosis_excess"),
            "principle": "Returns in real markets usually have fat tails; excess kurtosis above 0 supports this.",
        },
        {
            "metric": "volatility_clustering_abs_return_autocorr_1",
            "value": _safe_get(sim_facts, "abs_return_autocorr_1"),
            "principle": "Large price moves tend to cluster; positive autocorrelation of absolute returns supports this.",
        },
        {
            "metric": "volatility_clustering_abs_return_autocorr_5",
            "value": _safe_get(sim_facts, "abs_return_autocorr_5"),
            "principle": "Checks whether volatility persistence remains at a wider horizon.",
        },
        {
            "metric": "volatility_clustering_abs_return_autocorr_10",
            "value": _safe_get(sim_facts, "abs_return_autocorr_10"),
            "principle": "Checks whether volatility persistence remains across longer simulated memory.",
        },
        {
            "metric": "volume_volatility_correlation",
            "value": _safe_get(sim_facts_with_volume, "abs_return_volume_corr"),
            "principle": "Real markets often show higher volume during more volatile periods.",
        },
        {
            "metric": "ath_path_dependence_beta_h_mse_vs_real",
            "value": _beta_h_mse(paper_report),
            "principle": "Compares simulated and real beta_h; lower MSE means closer ATH/path-dependence behavior.",
        },
        {
            "metric": "has_simulated_historic_maximum",
            "value": _safe_get(historic_max, "has_historic_maximum"),
            "principle": "Confirms whether the simulated path creates new all-time highs.",
        },
        {
            "metric": "simulated_peak_price",
            "value": _safe_get(historic_max, "peak_price"),
            "principle": "Maximum simulated market price reached in the path.",
        },
        {
            "metric": "simulated_peak_tick",
            "value": _safe_get(historic_max, "peak_tick"),
            "principle": "Artificial tick where the simulated maximum appears.",
        },
    ])
    return macro_market_validation
def micro_table(micro_metrics: dict[str, Any]) -> pd.DataFrame:
    """This function formats agent-level validation metrics as a table.

    Params:
        micro_metrics: Agent-level validation metrics.
    """
    micro_llm_validation = pd.DataFrame([
        {
            "metric": "llm_orders",
            "value": micro_metrics["llm_orders"],
            "principle": "Number of observed LLM/FCL agent orders. NaN means this is a classical-only run or no LLM orders were saved.",
        },
        {
            "metric": "llm_order_flow_imbalance",
            "value": micro_metrics["llm_order_flow_imbalance"],
            "principle": "(buy - sell) / total. Values near 0 mean balanced order flow; positive means buy pressure.",
        },
        {
            "metric": "ks_p_value_ath_nearness_buy_vs_sell",
            "value": micro_metrics["ks_p_value_ath_nearness_buy_vs_sell"],
            "principle": "KS test comparing buy vs sell distributions near all-time highs.",
        },
        {
            "metric": "mannwhitney_p_value_buy_vs_sell",
            "value": micro_metrics["mannwhitney_p_value_buy_vs_sell"],
            "principle": "Mann-Whitney U test for contextual buy/sell differences near ATH.",
        },
        {
            "metric": "portfolio_p01",
            "value": micro_metrics["portfolio_p01"],
            "principle": "1st percentile of asset allocation proportion.",
        },
        {
            "metric": "portfolio_p50",
            "value": micro_metrics["portfolio_p50"],
            "principle": "Median asset allocation proportion.",
        },
        {
            "metric": "portfolio_p99",
            "value": micro_metrics["portfolio_p99"],
            "principle": "99th percentile of asset allocation proportion.",
        },
        {
            "metric": "portfolio_stability_width_p99_minus_p01",
            "value": micro_metrics["portfolio_stability_width_p99_minus_p01"],
            "principle": "Smaller width suggests more stable portfolio allocation.",
        },
        {
            "metric": "odean_proxy_pgr_minus_plr",
            "value": micro_metrics["odean_proxy_pgr_minus_plr"],
            "principle": "Proxy for disposition effect. Positive means more realized gains than realized losses among LLM/FCL sells.",
        },
    ])
    return micro_llm_validation
def validation_table_func(validation_metrics: dict[str, Any], distance_metrics: dict[str, float]) -> pd.DataFrame:
    """This function formats trajectory validation metrics as a table.

    Params:
        validation_metrics: Trajectory validation metrics.
        distance_metrics: Distribution-distance metrics.
    """
    real_vs_simulated_validation = pd.DataFrame([
        {
            "metric": "rmse",
            "value": _safe_get(validation_metrics, "rmse"),
            "principle": "Root mean squared price error between real holdout and simulated path.",
        },
        {
            "metric": "nrmse_mean_price",
            "value": _safe_get(validation_metrics, "nrmse_mean_price"),
            "principle": "RMSE normalized by mean real price.",
        },
        {
            "metric": "nrmse_price_range",
            "value": _safe_get(validation_metrics, "nrmse_price_range"),
            "principle": "RMSE normalized by real holdout price range.",
        },
        {
            "metric": "smape",
            "value": _safe_get(validation_metrics, "smape"),
            "principle": "Symmetric percentage error; more stable than plain MAPE near small denominators.",
        },
        {
            "metric": "wape",
            "value": _safe_get(validation_metrics, "wape"),
            "principle": "Weighted absolute percentage error.",
        },
        {
            "metric": "theil_u_vs_random_walk",
            "value": _safe_get(validation_metrics, "theil_u_vs_random_walk"),
            "principle": "Below 1 beats a naive random-walk baseline; above 1 underperforms it.",
        },
        {
            "metric": "directional_accuracy",
            "value": _safe_get(validation_metrics, "directional_accuracy"),
            "principle": "Share of days/ticks where simulated and real returns move in the same direction.",
        },
        {
            "metric": "return_correlation",
            "value": _safe_get(validation_metrics, "return_correlation"),
            "principle": "Correlation between real and simulated returns.",
        },
        {
            "metric": "kl_divergence_return_distribution",
            "value": distance_metrics["kl_divergence"],
            "principle": "Distribution distance between real and simulated returns. Lower is closer.",
        },
        {
            "metric": "hellinger_return_distribution",
            "value": distance_metrics["hellinger_distance"],
            "principle": "Bounded distribution distance between 0 and 1. Lower is closer.",
        },
    ])
    return real_vs_simulated_validation
def architecture_table() -> pd.DataFrame:
    """This function describes the simulation validation architecture.
    """
    architecture_safeguard = pd.DataFrame([
        {
            "component": "LLM/FCL agent",
            "responsibility": "Returns categorical trading intent and context-sensitive behavior.",
        },
        {
            "component": "Deterministic simulator",
            "responsibility": "Controls order pricing, sizing, matching, cash, inventory, and market state.",
        },
        {
            "component": "Validation layer",
            "responsibility": "Compares post-warmup simulated data with the real future holdout not shown to agents.",
        },
    ])
    return architecture_safeguard
def full_metric(paper_report: dict[str, Any],artifacts: dict[str, Any]) -> dict[str, pd.DataFrame]:
   
    """This function generates plots and validation tables for a report.

    Params:
        paper_report: Paper-style simulation report.
        artifacts: Loaded simulation artifacts.
    """
    plot_simulated_historic_maximum(
   artifacts["market"],
    price_col="market_price"
    )

    plot_all_time_high_nearness_histogram(
    paper_report["order_nearness"]
    )
   
    sim_facts = paper_report.get("sim_stylized_facts", {})
    sim_facts_with_volume = paper_report.get("sim_stylized_facts_with_execution_volume", sim_facts)
    historic_max = paper_report.get("sim_historic_maximum", {})
    validation_metrics = paper_report.get("validation_metrics", {})
    aligned = paper_report.get("real_vs_simulated", {}).get("aligned", pd.DataFrame())
    distance_metrics = _return_distribution_distances(aligned)
    #validation_table =  validation_table_func(validation_metrics, distance_metrics)
    macro_market_validation = macro_table(sim_facts, sim_facts_with_volume, historic_max, paper_report)
    micro_metrics = _llm_order_flow_metrics(artifacts, paper_report)
    micro_llm_validation = micro_table(micro_metrics)
    architecture_safeguard = architecture_table()



    validation_dashboard = {
        "macro": macro_market_validation,
        "micro": micro_llm_validation,
        "architecture": architecture_safeguard,
    }
    return validation_dashboard
