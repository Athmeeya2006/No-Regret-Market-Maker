#pragma once
#include <cstdint>
#include <limits>

// ----------------------------------------------------------------
// Enumerations
// ----------------------------------------------------------------
enum class Side     { BID = 0, ASK = 1 };
enum class OrderType{ LIMIT = 0, MARKET = 1, CANCEL = 2 };

// ----------------------------------------------------------------
// Order
// ----------------------------------------------------------------
struct Order {
    uint64_t  id                {0};
    Side      side              {Side::BID};
    OrderType type              {OrderType::LIMIT};
    double    price             {0.0};
    int       quantity          {0};
    int       remaining_qty     {0};
    uint64_t  timestamp         {0};
    int       trader_id         {-1};  // -1 = anonymous

    Order() = default;

    Order(Side s, OrderType t, double p, int q, int tid = -1)
        : side(s), type(t), price(p),
          quantity(q), remaining_qty(q), trader_id(tid) {}
};

// ----------------------------------------------------------------
// Fill  (execution report for a matched pair of orders)
// ----------------------------------------------------------------
struct Fill {
    uint64_t buy_order_id  {0};
    uint64_t sell_order_id {0};
    double   price         {0.0};
    int      quantity      {0};
    uint64_t timestamp     {0};
    int      buy_trader_id {-1};
    int      sell_trader_id{-1};

    // Convenience: did trader `tid` buy or sell in this fill?
    bool is_buyer(int tid)  const { return buy_trader_id  == tid; }
    bool is_seller(int tid) const { return sell_trader_id == tid; }
};
