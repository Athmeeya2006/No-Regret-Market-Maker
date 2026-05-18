#pragma once
#include "order.hpp"
#include <map>
#include <deque>
#include <unordered_map>
#include <vector>
#include <functional>
#include <limits>

// ----------------------------------------------------------------
// OrderBook
//
// Data structure:
//   bids_: std::map with std::greater -> best bid (highest price) first
//   asks_: std::map with default comparator -> best ask (lowest price) first
//
// Within each price level: std::deque<Order> in time-priority (FIFO).
//
// order_index_: maps order_id -> (side, price) for O(1) cancel lookup.
//
// Complexity:
//   insert  : O(log N)
//   cancel  : O(log N + L) where L = level depth (usually tiny)
//   best bid/ask: O(1)
// ----------------------------------------------------------------
class OrderBook {
public:
    // ------ Mutating operations ----------------------------------
    std::vector<Fill> add_limit_order (Order& order);
    std::vector<Fill> add_market_order(Order& order);
    bool              cancel_order    (uint64_t order_id);

    // ------ State queries ----------------------------------------
    double best_bid()  const;
    double best_ask()  const;
    double mid_price() const;
    double spread()    const;

    int    bid_depth(double price) const;
    int    ask_depth(double price) const;

    // OFI = (bid_vol - ask_vol) / (bid_vol + ask_vol) over top N levels
    double order_flow_imbalance(int levels = 5) const;

    // Total visible volume on each side (top `levels` levels)
    int total_bid_volume(int levels = 10) const;
    int total_ask_volume(int levels = 10) const;

    // Depth snapshots
    std::vector<std::pair<double,int>> bid_levels(int depth) const;
    std::vector<std::pair<double,int>> ask_levels(int depth) const;

    // Number of resting orders
    std::size_t n_bids() const { return order_index_bids_; }
    std::size_t n_asks() const { return order_index_asks_; }

    // Give MatchingEngine access to next_id counter
    uint64_t allocate_id() { return next_id_++; }

private:
    // Price-level maps
    std::map<double, std::deque<Order>, std::greater<double>> bids_;
    std::map<double, std::deque<Order>>                       asks_;

    // Cancel index: order_id -> (side, price)
    std::unordered_map<uint64_t, std::pair<Side,double>> order_index_;

    std::size_t order_index_bids_ {0};
    std::size_t order_index_asks_ {0};

    uint64_t next_id_      {1};
    uint64_t current_time_ {0};

    // ------ Internal matching helpers ----------------------------
    std::vector<Fill> match_against_asks(Order& aggressor);
    std::vector<Fill> match_against_bids(Order& aggressor);
    Fill make_fill(const Order& buy, const Order& sell,
                   double exec_price, int exec_qty);
};
