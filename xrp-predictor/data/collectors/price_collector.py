"""
Price data collector with full historical data support.

Fetches XRP, BTC, ETH OHLCV data from Binance with pagination
to get ALL available history (2+ years for most pairs).
"""

import time
import ccxt
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


# Folder where processed .parquet files will be stored
DATA_DIR = Path(__file__).resolve().parents[1] / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class PriceCollector:
    """
    Connects to Binance (via ccxt),
    downloads OHLCV data for XRP, BTC, ETH with full pagination,
    and combines it with BTC/ETH data from yfinance.
    """

    def __init__(self, exchange_name: str = "binance"):
        self.exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
        self.exchange.load_markets()

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        """
        Fetch a single batch of OHLCV data (for backward compatibility).
        """
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_full_history(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch COMPLETE historical OHLCV data by paginating through all available candles.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Candle interval ('1h', '4h', '1d', etc.)
            start_date: ISO date string (e.g., '2020-01-01') or None for 2 years back
            batch_size: Candles per request (max 1000 for Binance)
        
        Returns:
            DataFrame with ALL available historical data
        """
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        
        # Convert start_date to milliseconds
        if start_date:
            since = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        else:
            since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * 2)).timestamp() * 1000)
        
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        
        # Timeframe to milliseconds
        tf_map = {
            "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
            "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
        }
        tf_ms = tf_map.get(timeframe, 3_600_000)
        
        all_candles = []
        
        while since < end_ms:
            try:
                batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
                if not batch:
                    break
                
                all_candles.extend(batch)
                since = batch[-1][0] + tf_ms
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
                
                if len(batch) < batch_size:
                    break
                    
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
                continue
        
        if not all_candles:
            print("No data!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        
        print(f"{len(df):,} candles ({df.index.min().date()} to {df.index.max().date()})")
        return df

    def fetch_market_proxy(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch BTC-USD and ETH-USD from yfinance as a simple 'market context'.
        Note: yfinance only keeps 730 days of hourly data!
        """
        print("  Fetching market proxy from yfinance...", end=" ", flush=True)
        
        tickers = ["BTC-USD", "ETH-USD"]
        
        # yfinance hourly data is limited to 730 days
        # Use the later of: requested start_date or 729 days ago
        max_hourly_start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=729)).strftime("%Y-%m-%d")
        
        if start_date:
            effective_start = max(start_date, max_hourly_start)
        else:
            effective_start = max_hourly_start
        
        try:
            data = yf.download(
                " ".join(tickers), 
                start=effective_start,
                interval="1h", 
                group_by="ticker",
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            # Fallback to daily if hourly fails
            data = yf.download(
                " ".join(tickers),
                start=start_date, 
                interval="1d", 
                group_by="ticker",
                progress=False,
                auto_adjust=True,
            )
        
        if data.empty:
            print("No data!")
            return pd.DataFrame()
        
        # Flatten MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ["_".join(col).strip() for col in data.columns.to_flat_index()]

        # Ensure index is UTC
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")

        print(f"{len(data):,} rows")
        return data

    def run(self, timeframe: str = "1h", years: int = 2, start_date: Optional[str] = None):
        """
        Fetch full historical data and save to parquet.
        
        Args:
            timeframe: Candle interval ('1h', '4h', '1d')
            years: Years of history (ignored if start_date provided)
            start_date: Optional specific start date (YYYY-MM-DD)
        """
        if start_date is None:
            start_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * years)).strftime("%Y-%m-%d")
        
        print(f"\n[{datetime.now(timezone.utc)}] Fetching {timeframe} data from {start_date}\n")

        # Fetch with pagination
        xrp = self.fetch_full_history("XRP/USDT", timeframe, start_date)
        btc = self.fetch_full_history("BTC/USDT", timeframe, start_date)
        eth = self.fetch_full_history("ETH/USDT", timeframe, start_date)
        mkt = self.fetch_market_proxy(start_date)

        if xrp.empty:
            print("Error: No XRP data!")
            return

        # Combine with prefixes
        df = xrp.join(
            btc.add_prefix("btc_"), how="left"
        ).join(
            eth.add_prefix("eth_"), how="left"
        ).join(
            mkt, how="left"
        )
        
        # Forward-fill and drop NaN
        df = df.ffill().dropna()

        out_path = DATA_DIR / f"xrp_raw_{timeframe}.parquet"
        df.to_parquet(out_path)
        
        print(f"\n{'='*50}")
        print(f"Saved: {out_path}")
        print(f"Total: {len(df):,} candles")
        print(f"Range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Duration: {(df.index.max() - df.index.min()).days} days")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", "-t", default="1h")
    parser.add_argument("--years", "-y", type=int, default=2)
    parser.add_argument("--start", "-s", default=None, help="Start date YYYY-MM-DD")
    args = parser.parse_args()
    
    collector = PriceCollector()
    collector.run(timeframe=args.timeframe, years=args.years, start_date=args.start)