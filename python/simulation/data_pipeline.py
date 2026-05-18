"""Download Binance depth/trade data and extract EXP4 context features."""

from __future__ import annotations

import os
import io
import zipfile
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/daily"
BOOK_DEPTH_URL = f"{BASE_URL}/bookDepth"
KLINES_URL = f"{BASE_URL}/klines"
TRADES_URL = f"{BASE_URL}/trades"


class BinanceDataPipeline:
    def __init__(
        self,
        symbol:    str  = "BTCUSDT",
        data_dir:  str  = "data/binance_raw",
        processed: str  = "data/processed",
    ):
        self.symbol    = symbol
        self.data_dir  = Path(data_dir)
        self.processed = Path(processed)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Download
    # ================================================================

    def download_snapshot(self, date_str: str) -> Path:
        fname = f"{self.symbol}-bookDepth-{date_str}.zip"
        dest  = self.data_dir / fname
        if dest.exists():
            logger.info(f"Already downloaded: {dest}")
            return dest
        url = f"{BOOK_DEPTH_URL}/{self.symbol}/{fname}"
        self._download(url, dest)
        return dest

    def download_trades(self, date_str: str) -> Path:
        fname = f"{self.symbol}-trades-{date_str}.zip"
        dest  = self.data_dir / fname
        if dest.exists():
            logger.info(f"Already downloaded: {dest}")
            return dest
        url = f"{TRADES_URL}/{self.symbol}/{fname}"
        self._download(url, dest)
        return dest

    def download_klines(self, date_str: str, interval: str = "1m") -> Path:
        fname = f"{self.symbol}-{interval}-{date_str}.zip"
        dest = self.data_dir / fname
        if dest.exists():
            logger.info(f"Already downloaded: {dest}")
            return dest
        url = f"{KLINES_URL}/{self.symbol}/{interval}/{fname}"
        self._download(url, dest)
        return dest

    @staticmethod
    def _download(url: str, dest: Path):
        logger.info(f"Downloading {url} -> {dest}")
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            logger.info(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    # ================================================================
    # Parse raw files
    # ================================================================

    def load_snapshot(self, date_str: str) -> pd.DataFrame:
        """
        Parse the bookDepth zip.
        Format: timestamp, last_update_id, bids (json), asks (json)
        Binance bookDepth columns:
          symbol, timestamp, last_update_id, side, price, qty
        Returns a DataFrame with columns: timestamp, side, price, qty
        """
        path = self.data_dir / f"{self.symbol}-bookDepth-{date_str}.zip"
        if not path.exists():
            path = self.download_snapshot(date_str)

        with zipfile.ZipFile(path) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None)

        # Binance bookDepth format (as of 2023):
        # 0: symbol, 1: timestamp, 2: last_update_id, 3: side(BID/ASK), 4: price, 5: qty
        df.columns = ["symbol", "timestamp", "last_update_id", "side", "price", "qty"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["price"]     = df["price"].astype(float)
        df["qty"]       = df["qty"].astype(float)
        return df[["timestamp", "side", "price", "qty"]]

    def load_trades(self, date_str: str) -> pd.DataFrame:
        """
        Parse the trades zip.
        Binance trades format:
          id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
        """
        path = self.data_dir / f"{self.symbol}-trades-{date_str}.zip"
        if not path.exists():
            path = self.download_trades(date_str)

        with zipfile.ZipFile(path) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None)

        df.columns = ["id", "price", "qty", "quote_qty", "time",
                      "is_buyer_maker", "is_best_match"]
        df["time"]  = pd.to_datetime(df["time"], unit="ms")
        df["price"] = df["price"].astype(float)
        df["qty"]   = df["qty"].astype(float)
        # Derive side: if is_buyer_maker=True, the buyer rested and seller was aggressor => SELL aggression
        df["side"]  = np.where(df["is_buyer_maker"], "SELL", "BUY")
        return df[["time", "price", "qty", "side"]]

    def load_klines(self, date_str: str, interval: str = "1m") -> pd.DataFrame:
        path = self.data_dir / f"{self.symbol}-{interval}-{date_str}.zip"
        if not path.exists():
            path = self.download_klines(date_str, interval)

        with zipfile.ZipFile(path) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None)

        df = df.iloc[:, :12]
        df.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "n_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume", "n_trades"]]

    # ================================================================
    # LOB reconstruction
    # ================================================================

    def reconstruct_lob_snapshots(
        self,
        snapshot_df:  pd.DataFrame,
        freq:         str = "1min",
    ) -> pd.DataFrame:
        """
        Build a sequence of LOB snapshots at each `freq` interval.
        Returns a DataFrame with one row per timestamp containing:
          best_bid, best_ask, mid, spread,
          bid_vol_1, ask_vol_1, ..., bid_vol_5, ask_vol_5,
          ofi
        """
        records = []
        for ts, group in snapshot_df.groupby(
            pd.Grouper(key="timestamp", freq=freq)
        ):
            bids = group[group["side"] == "BID"].nlargest(5, "price")
            asks = group[group["side"] == "ASK"].nsmallest(5, "price")

            if bids.empty or asks.empty:
                continue

            best_bid = bids["price"].iloc[0]
            best_ask = asks["price"].iloc[0]
            mid      = (best_bid + best_ask) / 2
            spread   = best_ask - best_bid

            bid_vols = bids["qty"].values[:5]
            ask_vols = asks["qty"].values[:5]
            total_bid = bid_vols.sum()
            total_ask = ask_vols.sum()
            ofi = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)

            row = {
                "timestamp": ts,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "spread": spread,
                "ofi": ofi,
                "total_bid_vol": total_bid,
                "total_ask_vol": total_ask,
            }
            for i, (bv, av) in enumerate(
                zip(bid_vols, ask_vols), start=1
            ):
                row[f"bid_vol_{i}"] = bv
                row[f"ask_vol_{i}"] = av

            records.append(row)

        return pd.DataFrame(records).set_index("timestamp")

    def lob_from_klines(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        df = klines_df.copy()
        mid = df["close"].astype(float)
        spread = (df["high"] - df["low"]).clip(lower=0.01)
        out = pd.DataFrame(index=df["timestamp"])
        out["mid"] = mid.values
        out["spread"] = spread.values
        out["best_bid"] = out["mid"] - out["spread"] / 2
        out["best_ask"] = out["mid"] + out["spread"] / 2
        out["total_bid_vol"] = df["volume"].values / 2
        out["total_ask_vol"] = df["volume"].values / 2
        out["ofi"] = 0.0
        return out

    # ================================================================
    # Feature extraction
    # ================================================================

    def extract_features(
        self,
        lob_df:    pd.DataFrame,
        trades_df: pd.DataFrame,
        window:    int = 20,
    ) -> pd.DataFrame:
        """
        Compute EXP4 context features from LOB and trade data.

        Returns a DataFrame with per-row features suitable for use as
        context in EXP4MarketMaker.build_context().
        """
        df = lob_df.copy()

        # Realized volatility: std of log-returns over rolling window
        log_ret = np.log(df["mid"]).diff()
        df["realized_vol"] = log_ret.rolling(window).std()

        # Momentum: net return over window
        df["momentum"] = df["mid"].pct_change(window)

        # Merge trade features
        trades_df2 = trades_df.set_index("time").resample(
            df.index.freqstr or "1min"
        ).agg({"qty": "sum", "price": "mean"})
        trades_df2.columns = ["trade_volume", "vwap"]

        df = df.join(trades_df2, how="left")
        df["trade_volume"] = df["trade_volume"].fillna(0)
        df["arrival_rate"] = df["trade_volume"].rolling(window).mean()

        # Fill NaN from rolling
        df = df.bfill().ffill()

        return df

    def extract_features_for_day(self, date_str: str) -> pd.DataFrame:
        trades = self.load_trades(date_str)
        kline_path = self.data_dir / f"{self.symbol}-1m-{date_str}.zip"
        if kline_path.exists():
            klines = self.load_klines(date_str)
            lob_df = self.lob_from_klines(klines)
        else:
            try:
                snap = self.load_snapshot(date_str)
                lob_df = self.reconstruct_lob_snapshots(snap)
            except Exception:
                klines = self.load_klines(date_str)
                lob_df = self.lob_from_klines(klines)
        return self.extract_features(lob_df, trades)

    # ================================================================
    # Calibration helpers
    # ================================================================

    def calibrate_as_params(
        self, features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Estimate Avellaneda-Stoikov parameters from feature DataFrame.
        Returns dict with keys: sigma, kappa, gamma_suggested.
        """
        # sigma: median realized vol
        sigma = float(features["realized_vol"].median())

        # kappa: fit arrival_rate vs spread
        # lambda(delta) = lambda_0 * exp(-kappa * delta)
        # log lambda = log lambda_0 - kappa * delta
        spreads = features["spread"].dropna().values
        if "arrival_rate" in features:
            rates = features["arrival_rate"].dropna().values
            if len(rates) == len(spreads) and len(spreads) > 10:
                from python.algorithms.avellaneda_stoikov import AvellanedaStoikovMM
                kappa = AvellanedaStoikovMM.estimate_kappa(spreads, rates)
            else:
                kappa = 1.5
        else:
            kappa = 1.5

        return {"sigma": sigma, "kappa": kappa, "gamma_suggested": 0.1}

    def save_processed(self, features: pd.DataFrame, date_str: str):
        dest = self.processed / f"{self.symbol}-features-{date_str}.parquet"
        features.to_parquet(dest)
        logger.info(f"Saved processed features: {dest}")

    def load_processed(self, date_str: str) -> pd.DataFrame:
        dest = self.processed / f"{self.symbol}-features-{date_str}.parquet"
        if not dest.exists():
            raise FileNotFoundError(f"No processed file for {date_str}. Run extract_features_for_day() first.")
        return pd.read_parquet(dest)
