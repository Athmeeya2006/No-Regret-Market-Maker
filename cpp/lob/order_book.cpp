#include "lob/order_book.hpp"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cassert>

// ================================================================
// Queries
// ================================================================

double OrderBook::best_bid() const {
    if (bids_.empty()) return 0.0;
    return bids_.begin()->first;
}

double OrderBook::best_ask() const {
    if (asks_.empty()) return std::numeric_limits<double>::infinity();
    return asks_.begin()->first;
}

double OrderBook::mid_price() const {
    double bb = best_bid();
    double ba = best_ask();
    if (bb <= 0.0 && ba == std::numeric_limits<double>::infinity())
        return 0.0;
    if (bb <= 0.0 || ba == std::numeric_limits<double>::infinity())
        return (bb <= 0.0 ? ba : bb);
    return (bb + ba) / 2.0;
}

double OrderBook::spread() const {
    double bb = best_bid();
    double ba = best_ask();
    if (bb <= 0.0 || ba == std::numeric_limits<double>::infinity())
        return std::numeric_limits<double>::infinity();
    return ba - bb;
}

int OrderBook::bid_depth(double price) const {
    auto it = bids_.find(price);
    if (it == bids_.end()) return 0;
    int total = 0;
    for (const auto& o : it->second) total += o.remaining_qty;
    return total;
}

int OrderBook::ask_depth(double price) const {
    auto it = asks_.find(price);
    if (it == asks_.end()) return 0;
    int total = 0;
    for (const auto& o : it->second) total += o.remaining_qty;
    return total;
}

double OrderBook::order_flow_imbalance(int levels) const {
    long long bid_vol = 0, ask_vol = 0;
    int cnt = 0;
    for (const auto& [price, deq] : bids_) {
        if (cnt++ >= levels) break;
        for (const auto& o : deq) bid_vol += o.remaining_qty;
    }
    cnt = 0;
    for (const auto& [price, deq] : asks_) {
        if (cnt++ >= levels) break;
        for (const auto& o : deq) ask_vol += o.remaining_qty;
    }
    long long total = bid_vol + ask_vol;
    if (total == 0) return 0.0;
    return static_cast<double>(bid_vol - ask_vol) / static_cast<double>(total);
}

int OrderBook::total_bid_volume(int levels) const {
    int total = 0, cnt = 0;
    for (const auto& [price, deq] : bids_) {
        if (cnt++ >= levels) break;
        for (const auto& o : deq) total += o.remaining_qty;
    }
    return total;
}

int OrderBook::total_ask_volume(int levels) const {
    int total = 0, cnt = 0;
    for (const auto& [price, deq] : asks_) {
        if (cnt++ >= levels) break;
        for (const auto& o : deq) total += o.remaining_qty;
    }
    return total;
}

std::vector<std::pair<double,int>> OrderBook::bid_levels(int depth) const {
    std::vector<std::pair<double,int>> result;
    result.reserve(depth);
    int cnt = 0;
    for (const auto& [price, deq] : bids_) {
        if (cnt++ >= depth) break;
        int vol = 0;
        for (const auto& o : deq) vol += o.remaining_qty;
        result.emplace_back(price, vol);
    }
    return result;
}

std::vector<std::pair<double,int>> OrderBook::ask_levels(int depth) const {
    std::vector<std::pair<double,int>> result;
    result.reserve(depth);
    int cnt = 0;
    for (const auto& [price, deq] : asks_) {
        if (cnt++ >= depth) break;
        int vol = 0;
        for (const auto& o : deq) vol += o.remaining_qty;
        result.emplace_back(price, vol);
    }
    return result;
}

// ================================================================
// Internal helpers
// ================================================================

Fill OrderBook::make_fill(const Order& buy, const Order& sell,
                           double exec_price, int exec_qty) {
    Fill f;
    f.buy_order_id   = buy.id;
    f.sell_order_id  = sell.id;
    f.price          = exec_price;
    f.quantity       = exec_qty;
    f.timestamp      = current_time_++;
    f.buy_trader_id  = buy.trader_id;
    f.sell_trader_id = sell.trader_id;
    return f;
}

// Aggressor is a BID (buy) - match against asks (ascending price order).
std::vector<Fill> OrderBook::match_against_asks(Order& aggressor) {
    std::vector<Fill> fills;

    auto it = asks_.begin();
    while (it != asks_.end() && aggressor.remaining_qty > 0) {
        double ask_price = it->first;

        // Price check: limit buy only crosses if price >= ask
        if (aggressor.type == OrderType::LIMIT &&
            aggressor.price < ask_price) break;

        auto& deq = it->second;
        while (!deq.empty() && aggressor.remaining_qty > 0) {
            Order& passive = deq.front();
            int traded = std::min(aggressor.remaining_qty,
                                  passive.remaining_qty);

            // Execution at passive (resting) price: price improvement for aggressor
            Fill f = make_fill(aggressor, passive, ask_price, traded);
            fills.push_back(f);

            aggressor.remaining_qty  -= traded;
            passive.remaining_qty    -= traded;

            if (passive.remaining_qty == 0) {
                order_index_.erase(passive.id);
                --order_index_asks_;
                deq.pop_front();
            }
        }
        if (deq.empty()) {
            it = asks_.erase(it);
        } else {
            break;   // level partially filled; best ask still has resting qty
        }
    }
    return fills;
}

// Aggressor is an ASK (sell) - match against bids (descending price order).
std::vector<Fill> OrderBook::match_against_bids(Order& aggressor) {
    std::vector<Fill> fills;

    auto it = bids_.begin();
    while (it != bids_.end() && aggressor.remaining_qty > 0) {
        double bid_price = it->first;

        if (aggressor.type == OrderType::LIMIT &&
            aggressor.price > bid_price) break;

        auto& deq = it->second;
        while (!deq.empty() && aggressor.remaining_qty > 0) {
            Order& passive = deq.front();
            int traded = std::min(aggressor.remaining_qty,
                                  passive.remaining_qty);

            Fill f = make_fill(passive, aggressor, bid_price, traded);
            fills.push_back(f);

            aggressor.remaining_qty  -= traded;
            passive.remaining_qty    -= traded;

            if (passive.remaining_qty == 0) {
                order_index_.erase(passive.id);
                --order_index_bids_;
                deq.pop_front();
            }
        }
        if (deq.empty()) {
            it = bids_.erase(it);
        } else {
            break;
        }
    }
    return fills;
}

// ================================================================
// Mutating operations
// ================================================================

std::vector<Fill> OrderBook::add_limit_order(Order& order) {
    order.id          = next_id_++;
    order.remaining_qty = order.quantity;
    order.timestamp   = current_time_;

    std::vector<Fill> fills;

    if (order.side == Side::BID) {
        fills = match_against_asks(order);
        if (order.remaining_qty > 0) {
            bids_[order.price].push_back(order);
            order_index_[order.id] = {Side::BID, order.price};
            ++order_index_bids_;
        }
    } else {
        fills = match_against_bids(order);
        if (order.remaining_qty > 0) {
            asks_[order.price].push_back(order);
            order_index_[order.id] = {Side::ASK, order.price};
            ++order_index_asks_;
        }
    }
    return fills;
}

std::vector<Fill> OrderBook::add_market_order(Order& order) {
    order.id            = next_id_++;
    order.remaining_qty = order.quantity;
    order.type          = OrderType::MARKET;
    order.timestamp     = current_time_;

    if (order.side == Side::BID) {
        return match_against_asks(order);
    } else {
        return match_against_bids(order);
    }
    // Note: if order.remaining_qty > 0 after matching, it is simply dropped
    // (market orders do not rest in the book).
}

bool OrderBook::cancel_order(uint64_t order_id) {
    auto idx_it = order_index_.find(order_id);
    if (idx_it == order_index_.end()) return false;

    auto [side, price] = idx_it->second;
    order_index_.erase(idx_it);

    auto remove_from_deque = [&](auto& price_map) {
        auto map_it = price_map.find(price);
        if (map_it == price_map.end()) return;
        auto& deq = map_it->second;
        for (auto it = deq.begin(); it != deq.end(); ++it) {
            if (it->id == order_id) {
                deq.erase(it);
                break;
            }
        }
        if (deq.empty()) price_map.erase(map_it);
    };

    if (side == Side::BID) {
        remove_from_deque(bids_);
        --order_index_bids_;
    } else {
        remove_from_deque(asks_);
        --order_index_asks_;
    }
    return true;
}
