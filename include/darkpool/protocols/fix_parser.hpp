#pragma once

#include <string_view>
#include <unordered_map>
#include <charconv>
#include "darkpool/types.hpp"
#include "darkpool/utils/memory_pool.hpp"

namespace darkpool {

class FIXParser {
public:
    static constexpr char SOH = '\x01';  // FIX field delimiter
    
    explicit FIXParser(MemoryPool<Order>& order_pool, MemoryPool<Trade>& trade_pool);
    
    // Parse FIX message and return market message
    std::optional<MarketMessage> parse(const char* buffer, size_t length);
    
    // Zero-copy parsing for specific message types
    std::optional<Order> parse_new_order(std::string_view message);
    std::optional<Trade> parse_execution_report(std::string_view message);
    std::optional<Quote> parse_quote(std::string_view message);
    
    // Symbol mapping
    void add_symbol_mapping(const std::string& fix_symbol, Symbol numeric_symbol);
    
    // Performance stats
    struct Stats {
        uint64_t messages_parsed = 0;
        uint64_t parse_errors = 0;
        uint64_t unknown_msg_types = 0;
    };
    
    const Stats& stats() const { return stats_; }
    
private:
    // FIX field tags
    enum class Tag : uint16_t {
        BeginString = 8,
        BodyLength = 9,
        MsgType = 35,
        SenderCompID = 49,
        TargetCompID = 56,
        MsgSeqNum = 34,
        SendingTime = 52,
        
        // Order fields
        ClOrdID = 11,
        Symbol = 55,
        Side = 54,
        OrderQty = 38,
        Price = 44,
        OrdType = 40,
        TimeInForce = 59,
        
        // Execution fields
        OrderID = 37,
        ExecID = 17,
        ExecType = 150,
        OrdStatus = 39,
        LastPx = 31,
        LastQty = 32,
        CumQty = 14,
        AvgPx = 6,
        
        // Quote fields
        QuoteID = 117,
        BidPx = 132,
        OfferPx = 133,
        BidSize = 134,
        OfferSize = 135,
        
        CheckSum = 10
    };
    
    // Fast field extraction
    template<typename T>
    bool extract_field(std::string_view& message, Tag tag, T& value);
    
    bool extract_string_field(std::string_view& message, Tag tag, std::string_view& value);
    
    // Parse timestamp in FIX format (YYYYMMDD-HH:MM:SS.sss)
    Timestamp parse_timestamp(std::string_view timestamp_str);
    
    // Symbol lookup
    Symbol lookup_symbol(std::string_view fix_symbol);
    
    // Validate checksum
    bool validate_checksum(std::string_view message, uint8_t expected_checksum);
    
    // Memory pools
    MemoryPool<Order>& order_pool_;
    MemoryPool<Trade>& trade_pool_;
    
    // Symbol mapping
    std::unordered_map<std::string, Symbol> symbol_map_;
    
    // Stats
    Stats stats_;
    
    // Reusable buffers
    char parse_buffer_[4096];
};

// Optimized field extraction implementation
template<typename T>
inline bool FIXParser::extract_field(std::string_view& message, Tag tag, T& value) {
    char tag_buffer[16];
    int tag_len = snprintf(tag_buffer, sizeof(tag_buffer), "%d=", static_cast<int>(tag));
    
    size_t pos = message.find(tag_buffer);
    if (pos == std::string_view::npos) {
        return false;
    }
    
    pos += tag_len;
    size_t end = message.find(SOH, pos);
    if (end == std::string_view::npos) {
        return false;
    }
    
    if constexpr (std::is_integral_v<T>) {
        auto result = std::from_chars(message.data() + pos, message.data() + end, value);
        return result.ec == std::errc{};
    } else if constexpr (std::is_floating_point_v<T>) {
        // Fast float parsing
        char buffer[32];
        size_t len = std::min(end - pos, sizeof(buffer) - 1);
        std::memcpy(buffer, message.data() + pos, len);
        buffer[len] = '\0';
        
        char* endptr;
        if constexpr (std::is_same_v<T, double>) {
            value = std::strtod(buffer, &endptr);
        } else {
            value = std::strtof(buffer, &endptr);
        }
        return endptr != buffer;
    }
    
    return false;
}

} 
