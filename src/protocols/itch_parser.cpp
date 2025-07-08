#include "darkpool/protocols/itch_parser.hpp"
#include <algorithm>

namespace darkpool {

ITCHParser::ITCHParser(MemoryPool<Order>& order_pool, MemoryPool<Trade>& trade_pool)
    : order_pool_(order_pool), trade_pool_(trade_pool) {
    // Initialize symbol map with zeros
    std::fill(symbol_map_.begin(), symbol_map_.end(), 0);
    
    // Reserve space for order tracking
    order_map_.reserve(100000);
}

std::optional<Order> ITCHParser::parse_add_order(const ITCHAddOrder* msg) {
    auto* order = order_pool_.allocate();
    if (!order) {
        return std::nullopt;
    }
    
    order->id = from_be(msg->order_reference_number);
    order->symbol = parse_symbol(msg->stock, from_be(msg->stock_locate));
    order->side = parse_side(msg->buy_sell_indicator);
    order->quantity = from_be(msg->shares);
    order->remaining_quantity = order->quantity;
    order->price = from_be(msg->price);
    order->timestamp = parse_timestamp(msg->timestamp);
    order->type = OrderType::LIMIT;
    order->venue = Venue::NASDAQ;
    
    // Track order for execution matching
    order_map_[order->id] = {order->symbol, order->price, order->side};
    
    return *order;
}

std::optional<Trade> ITCHParser::parse_trade(const ITCHTrade* msg) {
    auto* trade = trade_pool_.allocate();
    if (!trade) {
        return std::nullopt;
    }
    
    trade->order_id = from_be(msg->order_reference_number);
    trade->symbol = parse_symbol(msg->stock, from_be(msg->stock_locate));
    trade->price = from_be(msg->price);
    trade->quantity = from_be(msg->shares);
    trade->timestamp = parse_timestamp(msg->timestamp);
    trade->aggressor_side = parse_side(msg->buy_sell_indicator);
    trade->venue = Venue::NASDAQ;
    trade->is_hidden = false;
    
    return *trade;
}

std::optional<Trade> ITCHParser::parse_executed_order(const ITCHExecutedOrder* msg) {
    auto* trade = trade_pool_.allocate();
    if (!trade) {
        return std::nullopt;
    }
    
    uint64_t order_ref = from_be(msg->order_reference_number);
    auto it = order_map_.find(order_ref);
    if (it == order_map_.end()) {
        trade_pool_.deallocate(trade);
        return std::nullopt;
    }
    
    trade->order_id = order_ref;
    trade->symbol = it->second.symbol;
    trade->price = it->second.price;
    trade->quantity = from_be(msg->executed_shares);
    trade->timestamp = parse_timestamp(msg->timestamp);
    trade->aggressor_side = it->second.side;
    trade->venue = Venue::NASDAQ;
    trade->is_hidden = false;
    
    return *trade;
}

Symbol ITCHParser::parse_symbol(const char stock[8], uint16_t stock_locate) {
    // First check if we have a mapping for this stock_locate
    if (stock_locate < symbol_map_.size() && symbol_map_[stock_locate] != 0) {
        return symbol_map_[stock_locate];
    }
    
    // Parse the symbol string (right-padded with spaces)
    char symbol_str[9] = {0};
    for (int i = 0; i < 8 && stock[i] != ' '; ++i) {
        symbol_str[i] = stock[i];
    }
    
    // Generate numeric symbol ID
    static std::atomic<Symbol> next_symbol_id{1};
    Symbol symbol_id = next_symbol_id.fetch_add(1, std::memory_order_relaxed);
    
    // Cache the mapping
    if (stock_locate < symbol_map_.size()) {
        symbol_map_[stock_locate] = symbol_id;
    }
    
    return symbol_id;
}

void ITCHParser::add_symbol_mapping(uint16_t stock_locate, const std::string& symbol) {
    if (stock_locate < symbol_map_.size()) {
        // In real implementation, would need to convert string symbol to numeric ID
        static std::atomic<Symbol> next_symbol_id{1};
        symbol_map_[stock_locate] = next_symbol_id.fetch_add(1, std::memory_order_relaxed);
    }
}

} 
