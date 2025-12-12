import ccxt
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone


# Folder where processed .parquet files will be stored
DATA_DIR = Path(__file__).resolve().parents[1] / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class PriceCollector:
    """
    Connects to Binance (via ccxt),
    downloads OHLCV data for XRP, BTC, ETH,
    and combines it with BTC/ETH data from yfinance.
    """

    def __init__(self, exchange_name: str = "binance"):
        self.exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 2000) -> pd.DataFrame:
        """
        Fetch OHLCV data and return DataFrame:
        index: timestamp (UTC), columns: open, high, low, close, volume
        """
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_market_proxy(self) -> pd.DataFrame:
        """
        Fetch BTC-USD and ETH-USD from yfinance as a simple 'market context'.
        """
        tickers = ["BTC-USD", "ETH-USD"]
        data = yf.download(" ".join(tickers), interval="60m", group_by="ticker")
        # Flatten MultiIndex columns
        data.columns = ["_".join(col).strip() for col in data.columns.to_flat_index()]

        # Ensure index is UTC, regardless of what yfinance returns
        if data.index.tz is None:
            # No timezone info: localize to UTC
            data.index = data.index.tz_localize("UTC")
        else:
            # Already tz-aware: convert to UTC (safe)
            data.index = data.index.tz_convert("UTC")

        return data

    def run(self, timeframe: str = "1h", limit: int = 2000):
        print(f"[{datetime.now(timezone.utc)}] Fetching OHLCV data...")

        xrp = self.fetch_ohlcv("XRP/USDT", timeframe, limit)
        btc = self.fetch_ohlcv("BTC/USDT", timeframe, limit)
        eth = self.fetch_ohlcv("ETH/USDT", timeframe, limit)
        mkt = self.fetch_market_proxy()

        # Add prefixes to avoid column name clashes
        df = xrp.join(
            btc.add_prefix("btc_"), how="inner"
        ).join(
            eth.add_prefix("eth_"), how="inner"
        ).join(
            mkt, how="inner"
        )

        out_path = DATA_DIR / f"xrp_raw_{timeframe}.parquet"
        df.to_parquet(out_path)
        print(f"Saved raw combined data to {out_path}")


if __name__ == "__main__":
    collector = PriceCollector()
    collector.run()