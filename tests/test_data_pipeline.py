"""
Tests for BinanceDataPipeline.

Run with: pytest tests/test_data_pipeline.py -v --tb=short

Note: download tests are skipped by default (require network).
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.simulation.data_pipeline import BinanceDataPipeline


class TestDataPipeline:

    def test_pipeline_init(self):
        pipe = BinanceDataPipeline(symbol="BTCUSDT")
        assert pipe.symbol == "BTCUSDT"
        assert pipe.data_dir.exists()

    def test_feature_extraction_from_synthetic(self):
        """Test feature extraction with synthetic LOB data."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")

        lob_df = pd.DataFrame({
            "best_bid": 100.0 - np.random.uniform(0, 0.5, n),
            "best_ask": 100.0 + np.random.uniform(0, 0.5, n),
            "mid": 100.0 + np.cumsum(np.random.normal(0, 0.01, n)),
            "spread": np.random.uniform(0.01, 0.5, n),
            "ofi": np.random.uniform(-1, 1, n),
            "total_bid_vol": np.random.uniform(10, 100, n),
            "total_ask_vol": np.random.uniform(10, 100, n),
        }, index=timestamps)

        trades_df = pd.DataFrame({
            "time": timestamps,
            "price": 100.0 + np.cumsum(np.random.normal(0, 0.01, n)),
            "qty": np.random.uniform(0.1, 10, n),
            "side": np.random.choice(["BUY", "SELL"], n),
        })

        pipe = BinanceDataPipeline()
        features = pipe.extract_features(lob_df, trades_df, window=10)
        assert "realized_vol" in features.columns
        assert "momentum" in features.columns
        assert len(features) == n

    @pytest.mark.skip(reason="Requires network access")
    def test_download_snapshot(self):
        pipe = BinanceDataPipeline(symbol="BTCUSDT")
        path = pipe.download_snapshot("2024-01-15")
        assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
