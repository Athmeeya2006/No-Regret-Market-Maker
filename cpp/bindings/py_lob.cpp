#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lob/order.hpp"
#include "lob/order_book.hpp"
#include "lob/matching_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lob_engine, m) {
    m.doc() = "C++ Limit Order Book engine";

    // ----------------------------------------------------------------
    // Side / OrderType enumerations
    // ----------------------------------------------------------------
    py::enum_<Side>(m, "Side")
        .value("BID", Side::BID)
        .value("ASK", Side::ASK)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("LIMIT",  OrderType::LIMIT)
        .value("MARKET", OrderType::MARKET)
        .value("CANCEL", OrderType::CANCEL)
        .export_values();

    // ----------------------------------------------------------------
    // Order
    // ----------------------------------------------------------------
    py::class_<Order>(m, "Order")
        .def(py::init<Side, OrderType, double, int, int>(),
             py::arg("side"), py::arg("type"), py::arg("price"),
             py::arg("quantity"), py::arg("trader_id") = -1)
        .def_readwrite("id",            &Order::id)
        .def_readwrite("side",          &Order::side)
        .def_readwrite("type",          &Order::type)
        .def_readwrite("price",         &Order::price)
        .def_readwrite("quantity",      &Order::quantity)
        .def_readwrite("remaining_qty", &Order::remaining_qty)
        .def_readwrite("timestamp",     &Order::timestamp)
        .def_readwrite("trader_id",     &Order::trader_id)
        .def("__repr__", [](const Order& o) {
            return "<Order id=" + std::to_string(o.id)
                 + " side=" + (o.side == Side::BID ? "BID" : "ASK")
                 + " price=" + std::to_string(o.price)
                 + " qty=" + std::to_string(o.quantity) + ">";
        });

    // ----------------------------------------------------------------
    // Fill
    // ----------------------------------------------------------------
    py::class_<Fill>(m, "Fill")
        .def_readonly("buy_order_id",   &Fill::buy_order_id)
        .def_readonly("sell_order_id",  &Fill::sell_order_id)
        .def_readonly("price",          &Fill::price)
        .def_readonly("quantity",       &Fill::quantity)
        .def_readonly("timestamp",      &Fill::timestamp)
        .def_readonly("buy_trader_id",  &Fill::buy_trader_id)
        .def_readonly("sell_trader_id", &Fill::sell_trader_id)
        .def("is_buyer",  &Fill::is_buyer)
        .def("is_seller", &Fill::is_seller)
        .def("__repr__", [](const Fill& f) {
            return "<Fill price=" + std::to_string(f.price)
                 + " qty=" + std::to_string(f.quantity)
                 + " ts=" + std::to_string(f.timestamp) + ">";
        });

    // ----------------------------------------------------------------
    // MatchingEngine   (primary Python-facing class)
    // ----------------------------------------------------------------
    py::class_<MatchingEngine>(m, "MatchingEngine")
        .def(py::init<double>(), py::arg("tick_size") = 0.01,
             "Construct a matching engine.\n\n"
             "tick_size : minimum price increment (default 0.01)")

        // Order submission
        .def("submit_limit",
             &MatchingEngine::submit_limit,
             py::arg("side"), py::arg("price"), py::arg("qty"),
             py::arg("trader_id") = -1,
             "Submit a limit order. Returns list of Fill objects.")
        .def("submit_market",
             &MatchingEngine::submit_market,
             py::arg("side"), py::arg("qty"), py::arg("trader_id") = -1,
             "Submit a market order. Returns list of Fill objects.")
        .def("cancel",
             &MatchingEngine::cancel_order,
             py::arg("order_id"),
             "Cancel a resting order by ID. Returns True if found.")
        .def("last_order_id",
             &MatchingEngine::last_order_id,
             "ID assigned to the most recently submitted order.")

        // State queries
        .def("best_bid",         &MatchingEngine::best_bid)
        .def("best_ask",         &MatchingEngine::best_ask)
        .def("mid_price",        &MatchingEngine::mid_price)
        .def("spread",           &MatchingEngine::spread)
        .def("ofi",              &MatchingEngine::ofi,
             "Order flow imbalance over top 5 levels.")
        .def("bid_levels",       &MatchingEngine::bid_levels, py::arg("depth") = 5)
        .def("ask_levels",       &MatchingEngine::ask_levels, py::arg("depth") = 5)
        .def("total_bid_volume", &MatchingEngine::total_bid_volume, py::arg("levels") = 10)
        .def("total_ask_volume", &MatchingEngine::total_ask_volume, py::arg("levels") = 10)

        // Derived statistics
        .def("vwap",
             &MatchingEngine::vwap,
             py::arg("n_last_fills") = 50)
        .def("realized_volatility",
             &MatchingEngine::realized_volatility,
             py::arg("window") = 20)

        // History
        .def("fill_history",
             &MatchingEngine::fill_history,
             py::return_value_policy::reference_internal)
        .def("mid_price_history",
             [](const MatchingEngine& e) {
                 // Convert deque to vector for Python
                 const auto& d = e.mid_price_history();
                 return std::vector<double>(d.begin(), d.end());
             })
        .def("__repr__", [](const MatchingEngine& e) {
            return "<MatchingEngine mid=" + std::to_string(e.mid_price())
                 + " spread=" + std::to_string(e.spread()) + ">";
        });
}
