#pragma once
#include <unordered_map>
#pragma once

#include <cstring>
#include <array>
#include "darkpool/types.hpp"
#include "darkpool/utils/memory_pool.hpp"

namespace darkpool {

// ITCH 5.0 message types
enum class ITCHMessageType : char {
    SystemEvent = 'S',
    StockDirectory = 'R',
    StockTradingAction = 'H',
    RegSHORestriction = 'Y',
    MarketParticipantPosition = 'L',
    MWCBDeclineLevel = 'V',
    MWCBStatus = 'W',
    IPOQuotingPeriod = 'K',
    AddOrder = 'A',
    AddOrderMPID = 'F',
    ExecutedOrder = 'E',
    ExecutedOrderWithPrice = 'C',
    OrderCancel = 'X',
    OrderDelete = 'D',
    OrderReplace = 'U',
    Trade = 'P',
    CrossTrade = 'Q',
    BrokenTrade = 'B',
    NOII = 'I',
    RPII = 'N',
    LULDAuctionCollar = 'J'
};

#pragma pack(push, 1)

// Base ITCH message header
struct ITCHMessageHeader {
    uint16_t length;  // Big-endian
    ITCHMessageType type;
};

// Add Order message
struct ITCHAddOrder {
    ITCHMessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint8_t timestamp[6];  // Nanoseconds since midnight
    uint64_t order_reference_number;
    char buy_sell_indicator;
    uint32_t shares;
    char stock[8];
    uint32_t price;  // Price * 10000
};

// Add Order with MPID
struct ITCHAddOrderMPID {
    ITCHAddOrder base;
    char mpid[4];
};

// Executed Order
struct ITCHExecutedOrder {
    ITCHMessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint8_t timestamp[6];
    uint64_t order_reference_number;
    uint32_t executed_shares;
    uint64_t match_number;
};

// Trade message
struct ITCHTrade {
    ITCHMessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint8_t timestamp[6];
    uint64_t order_reference_number;
    char buy_sell_indicator;
    uint32_t shares;
    char stock[8];
    uint32_t price;
    uint64_t match_number;
};

#pragma pack(pop)

class ITCHParser {
public:
    explicit ITCHParser(MemoryPool<Order>& order_pool, MemoryPool<Trade>& trade_pool);
    
    // Parse ITCH message
    std::optional<MarketMessage> parse(const uint8_t* buffer, size_t& bytes_consumed);
    
    // Parse specific message types
    std::optional<Order> parse_add_order(const ITCHAddOrder* msg);
    std::optional<Trade> parse_trade(const ITCHTrade* msg);
    std::optional<Trade> parse_executed_order(const ITCHExecutedOrder* msg);
    
    // Symbol mapping
    void add_symbol_mapping(uint16_t stock_locate, const std::string& symbol);
    
    // Stats
    struct Stats {
        uint64_t messages_parsed = 0;
        uint64_t add_orders = 0;
        uint64_t executions = 0;
        uint64_t trades = 0;
        uint64_t cancels = 0;
        uint64_t unknown_types = 0;
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
    
    // Parse 48-bit timestamp (nanoseconds since midnight)
    static Timestamp parse_timestamp(const uint8_t timestamp[6]) {
        uint64_t ns = 0;
        for (int i = 0; i < 6; ++i) {
            ns = (ns << 8) | timestamp[i];
        }
        return static_cast<Timestamp>(ns);
    }
    
    // Convert ITCH stock symbol to internal format
    Symbol parse_symbol(const char stock[8], uint16_t stock_locate);
    
    // Convert ITCH side to internal format
    static Side parse_side(char buy_sell_indicator) {
        return buy_sell_indicator == 'B' ? Side::BUY : Side::SELL;
    }
    
    // Memory pools
    MemoryPool<Order>& order_pool_;
    MemoryPool<Trade>& trade_pool_;
    
    // Symbol mapping: stock_locate -> Symbol
    std::array<Symbol, 10000> symbol_map_{};  // Max stock_locate value
    
    // Order tracking for execution matching
    struct OrderInfo {
        Symbol symbol;
        Price price;
        Side side;
    };
    std::unordered_map<uint64_t, OrderInfo> order_map_;
    
    // Stats
    Stats stats_;
};

// Fast inline parsing for hot path
inline std::optional<MarketMessage> ITCHParser::parse(const uint8_t* buffer, size_t& bytes_consumed) {
    if (bytes_consumed < sizeof(ITCHMessageHeader)) {
        return std::nullopt;
    }
    
    auto* header = reinterpret_cast<const ITCHMessageHeader*>(buffer);
    uint16_t msg_length = from_be(header->length) + 2;  // Length doesn't include itself
    
    if (bytes_consumed < msg_length) {
        return std::nullopt;
    }
    
    bytes_consumed = msg_length;
    
    switch (header->type) {
        case ITCHMessageType::AddOrder:
        case ITCHMessageType::AddOrderMPID: {
            auto order = parse_add_order(reinterpret_cast<const ITCHAddOrder*>(buffer));
            if (order) {
                stats_.add_orders++;
                return *order;
            }
            break;
        }
        
        case ITCHMessageType::Trade: {
            auto trade = parse_trade(reinterpret_cast<const ITCHTrade*>(buffer));
            if (trade) {
                stats_.trades++;
                return *trade;
            }
            break;
        }
        
        case ITCHMessageType::ExecutedOrder:
        case ITCHMessageType::ExecutedOrderWithPrice: {
            auto trade = parse_executed_order(reinterpret_cast<const ITCHExecutedOrder*>(buffer));
            if (trade) {
                stats_.executions++;
                return *trade;
            }
            break;
        }
        
        default:
            stats_.unknown_types++;
            break;
    }
    
    stats_.messages_parsed++;
    return std::nullopt;
}

} 

