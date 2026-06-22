"""data/data_acquisition.py

Functions to acquire market data (for volatility prediction), download credit-risk data from Kaggle,
and helpers to install / load an open-source factor library for stock-picking strategies.

Notes:
- Volatility data fetching uses akshare (https://akshare.readthedocs.io/).
- Kaggle downloads use Kaggle API: set KAGGLE_USERNAME and KAGGLE_KEY in env or use `kaggle.json`.
- Factor library installation defaults to AlphaPurify (https://github.com/eliasswu/AlphaPurify) but
  the functions are generic and accept any GitHub repo URL or pip package name.

Place this file in the repository at data/data_acquisition.py.
"""

import os
import subprocess
import tempfile
from typing import Optional
import datetime

import pandas as pd
import numpy as np

# akshare is optional at runtime; import when needed to keep the module import-safe


def fetch_ak_daily(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """Fetch daily OHLCV from akshare and return a pandas DataFrame.

    Parameters
    - symbol: market symbol accepted by akshare, e.g. "sh600000" or "sz000001".
    - start_date / end_date: strings in "YYYY-MM-DD" format. If None, akshare will return available range.
    - adjust: adjust mode, e.g. 'qfq' (forward-adjust), 'hfq' (back-adjust), or None.

    Returns
    - DataFrame indexed by date with columns: open, high, low, close, volume.

    Example:
        df = fetch_ak_daily("sh600000", "2020-01-01", "2021-01-01")
    """
    try:
        import akshare as ak
    except Exception as e:
        raise ImportError("akshare is required for fetch_ak_daily. Install with `pip install akshare`") from e

    # ak.stock_zh_a_daily accepts symbol, start_date, end_date and adjust
    kwargs = {}
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date
    if adjust:
        kwargs["adjust"] = adjust

    # akshare method name may vary across versions; use the common one
    df = ak.stock_zh_a_daily(symbol=symbol, **kwargs)
    # akshare returns a DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
    if "date" in df.columns:
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
    return df


def realized_volatility_from_prices(df: pd.DataFrame, price_col: str = "close", window: int = 20, annualize: bool = True) -> pd.Series:
    """Compute rolling realized volatility from price series.

    - Uses log returns: r_t = ln(P_t / P_{t-1})
    - Rolling std over `window` periods, multiplied by sqrt(252) if annualize.

    Parameters
    - df: DataFrame containing price series
    - price_col: column name for price
    - window: rolling window size in trading days
    - annualize: whether to scale by sqrt(252)

    Returns
    - pandas Series of rolling volatility aligned to the right (same index as df)
    """
    prices = df[price_col].astype(float).dropna()
    logret = np.log(prices).diff().dropna()
    rv = logret.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    rv.name = f"rv_{window}d"
    return rv


def download_kaggle_dataset(dataset: str, target_path: str = "data/kaggle", unzip: bool = True) -> str:
    """Download a dataset from Kaggle using the Kaggle API.

    Parameters
    - dataset: Kaggle dataset slug in the form 'owner/dataset-name' (required).
      Example: 'crowdflower/twitter-airline-sentiment' or 'zynicide/wine-reviews'.
      For competitions you may need to use the competition dataset slug.
    - target_path: local folder to store dataset files.
    - unzip: whether to unzip the downloaded archive (if any).

    Returns
    - path to the folder containing downloaded files.

    Notes
    - The Kaggle API requires credentials. See https://www.kaggle.com/docs/api for setup.
    - This function uses the KaggleApi Python client bundled with the kaggle package.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise ImportError("kaggle package is required to download datasets. Install with `pip install kaggle`") from e

    api = KaggleApi()
    api.authenticate()

    os.makedirs(target_path, exist_ok=True)
    # This will download and extract if unzip is True
    api.dataset_download_files(dataset, path=target_path, unzip=unzip, quiet=False)
    return os.path.abspath(target_path)


def pip_install_package(pkg: str) -> None:
    """Install a pip package (can be a GitHub URL like git+https://...).

    This runs 'pip install --upgrade <pkg>' in a subprocess. It raises a CalledProcessError on failure.
    """
    cmd = ["python", "-m", "pip", "install", "--upgrade", pkg]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def install_factor_library(repo_url: str = "git+https://github.com/eliasswu/AlphaPurify.git") -> None:
    """Install a factor library from a pip-installable source (PyPI name or git+URL).

    Default installs AlphaPurify from GitHub. If you prefer another factor lib, pass its pip spec.
    """
    pip_install_package(repo_url)


def try_import_module(module_name: str):
    """Attempt to import a module and return it, or raise ImportError with guidance."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import {module_name}. Make sure it's installed, e.g. `pip install {module_name}`") from e


if __name__ == "__main__":
    # Quick demo: fetch a sample symbol, compute realized vol
    print("Demo: fetch data and compute realized volatility (requires akshare)")
    try:
        df = fetch_ak_daily("sh600000", start_date="2021-01-01", end_date="2021-12-31")
        rv = realized_volatility_from_prices(df, window=20)
        print(rv.dropna().tail())
    except Exception as e:
        print("Demo failed:", e)

    print("\nIf you want to download a Kaggle dataset, call download_kaggle_dataset('owner/dataset')")
    print("To install AlphaPurify for factor usage, call install_factor_library() or pip install it manually.")
