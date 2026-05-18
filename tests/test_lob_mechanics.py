"""
Tests for the C++ Limit Order Book mechanics.

Run with: pytest tests/test_lob_mechanics.py -v --tb=short

Requires the lob_engine C++ extension to be built:
    pip install -e .
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lob_engine
    HAS_LOB = True
except ImportError:
    HAS_LOB = False


@pytest.mark.skipif(not HAS_LOB, reason="lob_engine C++ extension not built")
class TestLOBMechanics:

    def test_empty_book_mid_price(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        # Empty book should return 0 or NaN for mid
        mid = engine.mid_price()
        assert mid == 0.0 or mid != mid  # 0 or NaN

    def test_limit_order_submission(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        fills = engine.submit_limit(lob_engine.Side.BID, 100.0, 10)
        assert isinstance(fills, list)

    def test_matching_at_same_price(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.BID, 100.0, 10)
        fills = engine.submit_limit(lob_engine.Side.ASK, 100.0, 5)
        assert len(fills) > 0
        assert fills[0].quantity == 5
        assert fills[0].price == 100.0

    def test_market_order_fills_resting(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 10)
        fills = engine.submit_market(lob_engine.Side.BID, 5)
        assert len(fills) == 1
        assert fills[0].quantity == 5

    def test_cancel_order(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.BID, 100.0, 10, trader_id=1)
        order_id = engine.last_order_id()
        assert engine.cancel(order_id)
        fills = engine.submit_limit(lob_engine.Side.ASK, 100.0, 10, trader_id=2)
        assert fills == []

    def test_price_time_priority(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 5, trader_id=1)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 5, trader_id=2)
        fills = engine.submit_market(lob_engine.Side.BID, 8, trader_id=3)
        assert fills[0].sell_trader_id == 1
        assert fills[0].quantity == 5
        assert fills[1].sell_trader_id == 2
        assert fills[1].quantity == 3

    def test_fill_cash_is_zero_sum(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 4, trader_id=1)
        fill = engine.submit_market(lob_engine.Side.BID, 4, trader_id=2)[0]
        buyer_cash = -fill.price * fill.quantity
        seller_cash = fill.price * fill.quantity
        assert buyer_cash + seller_cash == pytest.approx(0.0)

    def test_market_order_sweeps_multiple_levels(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 3)
        engine.submit_limit(lob_engine.Side.ASK, 102.0, 4)
        fills = engine.submit_market(lob_engine.Side.BID, 6)
        assert len(fills) == 2
        assert [f.price for f in fills] == [101.0, 102.0]
        assert [f.quantity for f in fills] == [3, 3]

    def test_spread_with_orders(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.BID, 99.0, 10)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 10)
        assert engine.spread() == pytest.approx(2.0, abs=0.02)
        assert engine.mid_price() == pytest.approx(100.0, abs=0.02)

    def test_ofi_symmetric(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.BID, 99.0, 100)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 100)
        ofi = engine.ofi()
        assert abs(ofi) < 0.1  # roughly balanced

    def test_ofi_sign_positive_when_bid_depth_larger(self):
        engine = lob_engine.MatchingEngine(tick_size=0.01)
        engine.submit_limit(lob_engine.Side.BID, 99.0, 200)
        engine.submit_limit(lob_engine.Side.ASK, 101.0, 50)
        assert engine.ofi() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
