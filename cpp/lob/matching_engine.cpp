#include "lob/matching_engine.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

MatchingEngine::MatchingEngine(double tick_size)
    : tick_size_(tick_size) {}

// ================================================================
// Internal helpers
// ================================================================

double MatchingEngine::round_to_tick(double price) const {
    return std::round(price / tick_size_) * tick_size_;
}

void MatchingEngine::record_fills(const std::vector<Fill>& fills) {
    fill_history_.insert(fill_history_.end(), fills.begin(), fills.end());
}

void MatchingEngine::record_mid() {
    double m = book_.mid_price();
    if (std::isfinite(m) && m > 0.0) {
        mid_history_.push_back(m);
        if (static_cast<int>(mid_history_.size()) > max_mid_history_)
            mid_history_.pop_front();
    }
    ++round_;
}

// ================================================================
// Order submission
// ================================================================

std::vector<Fill> MatchingEngine::submit_limit(Side side, double price,
                                                int qty, int trader_id) {
    price = round_to_tick(price);
    Order order(side, OrderType::LIMIT, price, qty, trader_id);
    auto fills = book_.add_limit_order(order);
    last_order_id_ = order.id;
    record_fills(fills);
    record_mid();
    return fills;
}

std::vector<Fill> MatchingEngine::submit_market(Side side, int qty,
                                                  int trader_id) {
    Order order(side, OrderType::MARKET, 0.0, qty, trader_id);
    auto fills = book_.add_market_order(order);
    last_order_id_ = order.id;
    record_fills(fills);
    record_mid();
    return fills;
}

bool MatchingEngine::cancel_order(uint64_t order_id) {
    bool ok = book_.cancel_order(order_id);
    record_mid();
    return ok;
}

std::vector<Fill> MatchingEngine::process(Order& order) {
    std::vector<Fill> fills;
    switch (order.type) {
        case OrderType::LIMIT:
            fills = book_.add_limit_order(order);
            break;
        case OrderType::MARKET:
            fills = book_.add_market_order(order);
            break;
        case OrderType::CANCEL:
            book_.cancel_order(order.id);
            break;
    }
    if (order.type != OrderType::CANCEL) {
        last_order_id_ = order.id;
    }
    record_fills(fills);
    record_mid();
    return fills;
}

// ================================================================
// Derived statistics
// ================================================================

double MatchingEngine::vwap(int n_last_fills) const {
    if (fill_history_.empty()) return book_.mid_price();
    int n = static_cast<int>(fill_history_.size());
    int start = std::max(0, n - n_last_fills);

    double total_value = 0.0;
    int    total_qty   = 0;
    for (int i = start; i < n; ++i) {
        total_value += fill_history_[i].price * fill_history_[i].quantity;
        total_qty   += fill_history_[i].quantity;
    }
    if (total_qty == 0) return book_.mid_price();
    return total_value / static_cast<double>(total_qty);
}

double MatchingEngine::realized_volatility(int window) const {
    int n = static_cast<int>(mid_history_.size());
    if (n < 2) return 0.0;

    int periods = std::min(n - 1, window);
    int start   = n - 1 - periods;   // start index in mid_history_

    std::vector<double> log_returns;
    log_returns.reserve(periods);

    for (int i = start; i < n - 1; ++i) {
        double m0 = mid_history_[i];
        double m1 = mid_history_[i + 1];
        if (m0 > 0.0 && m1 > 0.0)
            log_returns.push_back(std::log(m1 / m0));
    }
    if (log_returns.empty()) return 0.0;

    double mean = std::accumulate(log_returns.begin(),
                                   log_returns.end(), 0.0)
                  / static_cast<double>(log_returns.size());

    double var = 0.0;
    for (double r : log_returns) var += (r - mean) * (r - mean);
    var /= static_cast<double>(log_returns.size());

    return std::sqrt(var);
}
