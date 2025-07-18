#pragma once
#include <atomic>
#include <unordered_map>
#pragma once

#include <cstdint>
#include <chrono>
#include <string>
#include <array>
#include <variant>
#include <optional>

namespace darkpool {

// Basic types
using Price = int64_t;  // Price in 1/10000 of a cent (0.0001 cent precision)
using Quantity = uint64_t;
using OrderId = uint64_t;
using Symbol = uint32_t;  // Numeric symbol for fast comparison
using Timestamp = std::chrono::nanoseconds::rep;

// Constants
constexpr size_t MAX_SYMBOL_LENGTH = 12;
constexpr size_t MAX_VENUES = 32;
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr Price PRICE_MULTIPLIER = 10000;

// Enums
enum class Side : uint8_t {
    BUY = 0,
    SELL = 1,
    UNKNOWN = 2
};

enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1,
    STOP = 2,
    STOP_LIMIT = 3,
    PEGGED = 4,
    HIDDEN = 5
};

enum class ExecutionType : uint8_t {
    NEW = 0,
    PARTIAL_FILL = 1,
    FILL = 2,
    CANCEL = 3,
    REPLACE = 4,
    REJECT = 5,
    EXPIRED = 6
};

enum class Venue : uint8_t {
    NYSE = 0,
    NASDAQ = 1,
    BATS = 2,
    ARCA = 3,
    IEX = 4,
    DARK_POOL_1 = 16,
    DARK_POOL_2 = 17,
    DARK_POOL_3 = 18,
    UNKNOWN = 255
};

enum class AnomalyType : uint8_t {
    HIGH_TQR = 0,
    ABNORMAL_SLIPPAGE = 1,
    ORDER_BOOK_PRESSURE = 2,
    HIDDEN_REFILL = 3,
    TRADE_CLUSTERING = 4,
    POST_TRADE_DRIFT = 5,
    MULTI_VENUE_SWEEP = 6,
    ICEBERG_DETECTED = 7
};

// Core structures - cache-aligned and packed
struct alignas(CACHE_LINE_SIZE) Order {
    OrderId id;
    Symbol symbol;
    Price price;
    Quantity quantity;
    Quantity remaining_quantity;
    Timestamp timestamp;
    Side side;
    OrderType type;
    Venue venue;
    uint8_t _padding[5];  // Explicit padding
    
    constexpr Order() noexcept :
        id(0), symbol(0), price(0), quantity(0), 
        remaining_quantity(0), timestamp(0),
        side(Side::UNKNOWN), type(OrderType::LIMIT), 
        venue(Venue::UNKNOWN), _padding{} {}
};

struct alignas(32) Trade {
    OrderId order_id;
    Symbol symbol;
    Price price;
    Quantity quantity;
    Timestamp timestamp;
    Side aggressor_side;
    Venue venue;
    bool is_hidden;
    uint8_t _padding[4];
};

struct alignas(32) Quote {
    Symbol symbol;
    Price bid_price;
    Price ask_price;
    Quantity bid_size;
    Quantity ask_size;
    Timestamp timestamp;
    Venue venue;
    uint8_t _padding[7];
};

struct OrderBookLevel {
    Price price;
    Quantity quantity;
    uint32_t order_count;
    uint32_t _padding;
};

struct alignas(64) OrderBookSnapshot {
    Symbol symbol;
    Timestamp timestamp;
    std::array<OrderBookLevel, 10> bids;
    std::array<OrderBookLevel, 10> asks;
    Venue venue;
    uint8_t _padding[7];
};

// Anomaly detection results
struct Anomaly {
    Symbol symbol;
    Timestamp timestamp;
    AnomalyType type;
    double confidence;  // 0.0 to 1.0
    double magnitude;   // Severity measure
    Quantity estimated_hidden_size;
    Price expected_impact;
    std::array<char, 256> description;
};

// Message variant for protocol normalization
using MarketMessage = std::variant<Order, Trade, Quote, OrderBookSnapshot>;

// Metrics for performance monitoring
struct alignas(CACHE_LINE_SIZE) PerformanceMetrics {
    std::atomic<uint64_t> messages_processed{0};
    std::atomic<uint64_t> anomalies_detected{0};
    std::atomic<uint64_t> total_latency_ns{0};
    std::atomic<uint64_t> max_latency_ns{0};
    std::atomic<uint64_t> ml_inference_count{0};
    std::atomic<uint64_t> ml_total_latency_ns{0};
};

// Helper functions
constexpr Price to_price(double price) {
    return static_cast<Price>(price * PRICE_MULTIPLIER);
}

constexpr double from_price(Price price) {
    return static_cast<double>(price) / PRICE_MULTIPLIER;
}

inline const char* to_string(Side side) {
    switch(side) {
        case Side::BUY: return "BUY";
        case Side::SELL: return "SELL";
        default: return "UNKNOWN";
    }
}

inline const char* to_string(AnomalyType type) {
    switch(type) {
        case AnomalyType::HIGH_TQR: return "HIGH_TQR";
        case AnomalyType::ABNORMAL_SLIPPAGE: return "ABNORMAL_SLIPPAGE";
        case AnomalyType::ORDER_BOOK_PRESSURE: return "ORDER_BOOK_PRESSURE";
        case AnomalyType::HIDDEN_REFILL: return "HIDDEN_REFILL";
        case AnomalyType::TRADE_CLUSTERING: return "TRADE_CLUSTERING";
        case AnomalyType::POST_TRADE_DRIFT: return "POST_TRADE_DRIFT";
        case AnomalyType::MULTI_VENUE_SWEEP: return "MULTI_VENUE_SWEEP";
        case AnomalyType::ICEBERG_DETECTED: return "ICEBERG_DETECTED";
        default: return "UNKNOWN";
    }
}

} 

