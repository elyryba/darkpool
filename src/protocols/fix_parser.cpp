#include "darkpool/protocols/fix_parser.hpp"
#include <sstream>
#include <algorithm>

namespace darkpool {

FIXParser::FIXParser(MemoryPool<Order>& order_pool, MemoryPool<Trade>& trade_pool)
    : order_pool_(order_pool), trade_pool_(trade_pool) {
    // Pre-populate common symbol mappings
    symbol_map_.reserve(1000);
}

std::optional<MarketMessage> FIXParser::parse(const char* buffer, size_t length) {
    std::string_view message(buffer, length);
    
    // Extract message type
    char msg_type;
    if (!extract_field(message, Tag::MsgType, msg_type)) {
        stats_.parse_errors++;
        return std::nullopt;
    }
    
    stats_.messages_parsed++;
    
    switch (msg_type) {
        case 'D':  // New Order Single
            return parse_new_order(message);
            
        case '8':  // Execution Report
            return parse_execution_report(message);
            
        case 'S':  // Quote
            return parse_quote(message);
            
        default:
            stats_.unknown_msg_types++;
            return std::nullopt;
    }
}

std::optional<Order> FIXParser::parse_new_order(std::string_view message) {
    auto* order = order_pool_.allocate();
    if (!order) {
        return std::nullopt;
    }
    
    // Extract order fields
    uint64_t cl_ord_id;
    if (!extract_field(message, Tag::ClOrdID, cl_ord_id)) {
        order_pool_.deallocate(order);
        return std::nullopt;
    }
    order->id = cl_ord_id;
    
    // Symbol
    std::string_view symbol_str;
    if (!extract_string_field(message, Tag::Symbol, symbol_str)) {
        order_pool_.deallocate(order);
        return std::nullopt;
    }
    order->symbol = lookup_symbol(symbol_str);
    
    // Side
    char side_char;
    if (!extract_field(message, Tag::Side, side_char)) {
        order_pool_.deallocate(order);
        return std::nullopt;
    }
    order->side = (side_char == '1') ? Side::BUY : Side::SELL;
    
    // Quantity
    if (!extract_field(message, Tag::OrderQty, order->quantity)) {
        order_pool_.deallocate(order);
        return std::nullopt;
    }
    order->remaining_quantity = order->quantity;
    
    // Price (convert to internal format)
    double price_double;
    if (!extract_field(message, Tag::Price, price_double)) {
        order_pool_.deallocate(order);
        return std::nullopt;
    }
    order->price = to_price(price_double);
    
    // Order type
    char ord_type;
    if (!extract_field(message, Tag::OrdType, ord_type)) {
        order_pool_.deallocate(order);
        return std::nullopt;
    }
    
    switch (ord_type) {
        case '1': order->type = OrderType::MARKET; break;
        case '2': order->type = OrderType::LIMIT; break;
        case '3': order->type = OrderType::STOP; break;
        case '4': order->type = OrderType::STOP_LIMIT; break;
        case 'P': order->type = OrderType::PEGGED; break;
        default: order->type = OrderType::LIMIT; break;
    }
    
    // Timestamp
    std::string_view timestamp_str;
    if (extract_string_field(message, Tag::SendingTime, timestamp_str)) {
        order->timestamp = parse_timestamp(timestamp_str);
    } else {
        order->timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    
    // Default venue (would be set by connection context in real implementation)
    order->venue = Venue::UNKNOWN;
    
    return *order;
}

std::optional<Trade> FIXParser::parse_execution_report(std::string_view message) {
    auto* trade = trade_pool_.allocate();
    if (!trade) {
        return std::nullopt;
    }
    
    // Extract execution fields
    char exec_type;
    if (!extract_field(message, Tag::ExecType, exec_type)) {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    
    // Only process fills and partial fills
    if (exec_type != '1' && exec_type != '2') {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    
    // Order ID
    if (!extract_field(message, Tag::OrderID, trade->order_id)) {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    
    // Symbol
    std::string_view symbol_str;
    if (!extract_string_field(message, Tag::Symbol, symbol_str)) {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    trade->symbol = lookup_symbol(symbol_str);
    
    // Last price
    double last_px;
    if (!extract_field(message, Tag::LastPx, last_px)) {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    trade->price = to_price(last_px);
    
    // Last quantity
    if (!extract_field(message, Tag::LastQty, trade->quantity)) {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    
    // Side
    char side_char;
    if (!extract_field(message, Tag::Side, side_char)) {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    trade->aggressor_side = (side_char == '1') ? Side::BUY : Side::SELL;
    
    // Timestamp
    std::string_view timestamp_str;
    if (extract_string_field(message, Tag::SendingTime, timestamp_str)) {
        trade->timestamp = parse_timestamp(timestamp_str);
    } else {
        trade->timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    
    trade->venue = Venue::UNKNOWN;
    trade->is_hidden = false;  // Could be determined from execution type
    
    return *trade;
}

std::optional<Quote> FIXParser::parse_quote(std::string_view message) {
    Quote quote;
    
    // Symbol
    std::string_view symbol_str;
    if (!extract_string_field(message, Tag::Symbol, symbol_str)) {
        return std::nullopt;
    }
    quote.symbol = lookup_symbol(symbol_str);
    
    // Bid price and size
    double bid_px;
    if (extract_field(message, Tag::BidPx, bid_px)) {
        quote.bid_price = to_price(bid_px);
    }
    
    if (!extract_field(message, Tag::BidSize, quote.bid_size)) {
        quote.bid_size = 0;
    }
    
    // Ask price and size
    double offer_px;
    if (extract_field(message, Tag::OfferPx, offer_px)) {
        quote.ask_price = to_price(offer_px);
    }
    
    if (!extract_field(message, Tag::OfferSize, quote.ask_size)) {
        quote.ask_size = 0;
    }
    
    // Timestamp
    std::string_view timestamp_str;
    if (extract_string_field(message, Tag::SendingTime, timestamp_str)) {
        quote.timestamp = parse_timestamp(timestamp_str);
    } else {
        quote.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    
    quote.venue = Venue::UNKNOWN;
    
    return quote;
}

bool FIXParser::extract_string_field(std::string_view& message, Tag tag, std::string_view& value) {
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
    
    value = message.substr(pos, end - pos);
    return true;
}

Timestamp FIXParser::parse_timestamp(std::string_view timestamp_str) {
    // FIX timestamp format: YYYYMMDD-HH:MM:SS.sss
    if (timestamp_str.length() < 17) {
        return 0;
    }
    
    // Simple parsing - in production would use more robust method
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0, millis = 0;
    
    // Parse components
    std::from_chars(timestamp_str.data(), timestamp_str.data() + 4, year);
    std::from_chars(timestamp_str.data() + 4, timestamp_str.data() + 6, month);
    std::from_chars(timestamp_str.data() + 6, timestamp_str.data() + 8, day);
    std::from_chars(timestamp_str.data() + 9, timestamp_str.data() + 11, hour);
    std::from_chars(timestamp_str.data() + 12, timestamp_str.data() + 14, minute);
    std::from_chars(timestamp_str.data() + 15, timestamp_str.data() + 17, second);
    
    if (timestamp_str.length() >= 21) {
        std::from_chars(timestamp_str.data() + 18, timestamp_str.data() + 21, millis);
    }
    
    // Convert to nanoseconds since epoch (simplified)
    // In production, would use proper date/time library
    uint64_t ns = second * 1000000000ULL + millis * 1000000ULL;
    ns += minute * 60ULL * 1000000000ULL;
    ns += hour * 3600ULL * 1000000000ULL;
    
    return static_cast<Timestamp>(ns);
}

Symbol FIXParser::lookup_symbol(std::string_view fix_symbol) {
    std::string symbol_str(fix_symbol);
    
    auto it = symbol_map_.find(symbol_str);
    if (it != symbol_map_.end()) {
        return it->second;
    }
    
    // Generate new symbol ID
    static std::atomic<Symbol> next_symbol_id{1};
    Symbol new_id = next_symbol_id.fetch_add(1, std::memory_order_relaxed);
    symbol_map_[symbol_str] = new_id;
    
    return new_id;
}

void FIXParser::add_symbol_mapping(const std::string& fix_symbol, Symbol numeric_symbol) {
    symbol_map_[fix_symbol] = numeric_symbol;
}

bool FIXParser::validate_checksum(std::string_view message, uint8_t expected_checksum) {
    uint32_t sum = 0;
    for (char c : message) {
        sum += static_cast<uint8_t>(c);
    }
    return (sum % 256) == expected_checksum;
}

} 
