#pragma once
#include "order.hpp"
#include <vector>

// ----------------------------------------------------------------
// Abstract base class for anything that can generate orders.
// Concrete implementations live in Python.
// This header exists so C++ tests can be written against the
// interface, and so the pybind11 binding can expose it as
// a trampoline class if needed.
// ----------------------------------------------------------------
class MarketParticipant {
public:
    explicit MarketParticipant(int id) : id_(id) {}
    virtual ~MarketParticipant() = default;

    // Generate orders for this time step.
    // `mid`  : current mid-price
    // `dt`   : time-step length
    // Returns a (possibly empty) vector of orders.
    virtual std::vector<Order> generate_orders(double mid, double dt) = 0;

    int id() const { return id_; }

protected:
    int id_;
};
