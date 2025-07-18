#pragma once
#include <unordered_map>

#include <cstring>
#include "darkpool/types.hpp"
#include "darkpool/utils/memory_pool.hpp"

namespace darkpool {

// OUCH 4.2 message types
enum class OUCHMessageType : char {
    // Inbound (client to exchange)
    EnterOrder = 'O',
    ReplaceOrder = 'U',
    CancelOrder = 'X',
    ModifyOrder = 'M',
    
    // Outbound (exchange to client)
    SystemEvent = 'S',
    Accepted = 'A',
    Replaced = 'U',
    Canceled = 'C',
    AIQCanceled = 'D',
    Executed = 'E',
    BrokenTrade = 'B',
    ExecutedWithReferencePrice = 'G',
    TradeCorrected = 'F',
    Rejected = 'J',
    CancelPending = 'P',
    CancelReject = 'I',
    OrderPriorityUpdate = 'T',
    OrderModified = 'M',
    OrderRestated = 'R'
};

#pragma pack(push, 1)

// OUCH message header
struct OUCHHeader {
    uint16_t packet_length;  // Big-endian
    uint8_t packet_type;     // Always 'U' for unsequenced data
    OUCHMessageType message_type;
};

// Enter Order message
struct OUCHEnterOrder {
    OUCHHeader header;
    uint32_t order_token;
    char buy_sell_indicator;
    uint32_t shares;
    char stock[8];
    uint32_t price;  // Price * 10000
    uint32_t time_in_force;
    char firm[4];
    char display;
    uint64_t capacity;
    char intermarket_sweep;
    uint32_t minimum_quantity;
    char cross_type;
    char customer_type;
};

// Order Accepted message
struct OUCHAccepted {
    OUCHHeader header;
    uint64_t timestamp;  // Nanoseconds since midnight
    uint32_t order_token;
    char buy_sell_indicator;
    uint32_t shares;
    char stock[8];
    uint32_t price;
    uint32_t time_in_force;
    char firm[4];
    char display;
    uint64_t order_reference_number;
    uint64_t capacity;
    char intermarket_sweep;
    uint32_t minimum_quantity;
    char cross_type;
    char order_state;
    char bbo_weight_indicator;
};

// Order Executed message
struct OUCHExecuted {
    OUCHHeader header;
    uint64_t timestamp;
    uint32_t order_token;
    uint32_t executed_shares;
    uint32_t execution_price;
    char liquidity_flag;
    uint64_t match_number;
};

#pragma pack(pop)

class OUCHParser {
public:
    explicit OUCHParser(MemoryPool<Order>& order_pool, MemoryPool<Trade>& trade_pool);
    
    // Parse OUCH message
    std::optional<MarketMessage> parse(const uint8_t* buffer, size_t& bytes_consumed);
    
    // Parse specific message types
    std::optional<Order> parse_enter_order(const OUCHEnterOrder* msg);
    std::optional<Order> parse_accepted(const OUCHAccepted* msg);
    std::optional<Trade> parse_executed(const OUCHExecuted* msg);
    
    // Symbol conversion
    void add_symbol_mapping(const std::string& ouch_symbol, Symbol numeric_symbol);
    
    // Order token tracking
    void track_order(uint32_t token, Symbol symbol, Side side, Price price);
    
    // Stats
    struct Stats {
        uint64_t messages_parsed = 0;
        uint64_t orders_entered = 0;
        uint64_t orders_accepted = 0;
        uint64_t orders_executed = 0;
        uint64_t orders_canceled = 0;
        uint64_t parse_errors = 0;
    };
    
    const Stats& stats() const { return stats_; }
    
private:
    // Convert big-endian to host byte order
    template<typename T>
    static T from_be(T value) {
        if constexpr (sizeof(T) == 2) {
            return __builtin_bswap16(value);
        } else if constexpr (sizeof(T) == 4) {
            return __builtin_bswap32(value);
        } else if constexpr (sizeof(T) == 8) {
            return __builtin_bswap64(value);
        }
        return value;
    }
    
    // Parse OUCH symbol (8 bytes, right-padded with spaces)
    Symbol parse_symbol(const char stock[8]);
    
    // Convert side indicator
    static Side parse_side(char indicator) {
        return indicator == 'B' ? Side::BUY : Side::SELL;
    }
    
    // Convert timestamp
    static Timestamp convert_timestamp(uint64_t ouch_timestamp) {
        return static_cast<Timestamp>(ouch_timestamp);
    }
    
    // Memory pools
    MemoryPool<Order>& order_pool_;
    MemoryPool<Trade>& trade_pool_;
    
    // Symbol mapping
    std::unordered_map<std::string, Symbol> symbol_map_;
    
    // Order tracking for execution matching
    struct OrderInfo {
        Symbol symbol;
        Side side;
        Price price;
        Quantity quantity;
    };
    std::unordered_map<uint32_t, OrderInfo> order_map_;
    
    // Stats
    Stats stats_;
};

}


