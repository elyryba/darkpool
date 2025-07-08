#include "darkpool/protocols/ouch_parser.hpp"
#include <algorithm>

namespace darkpool {

OUCHParser::OUCHParser(MemoryPool<Order>& order_pool, MemoryPool<Trade>& trade_pool)
    : order_pool_(order_pool), trade_pool_(trade_pool) {
    order_map_.reserve(50000);
}

std::optional<MarketMessage> OUCHParser::parse(const uint8_t* buffer, size_t& bytes_consumed) {
    if (bytes_consumed < sizeof(OUCHHeader)) {
        return std::nullopt;
    }
    
    auto* header = reinterpret_cast<const OUCHHeader*>(buffer);
    uint16_t packet_length = from_be(header->packet_length);
    
    if (bytes_consumed < packet_length + 2) {
        return std::nullopt;
    }
    
    bytes_consumed = packet_length + 2;
    stats_.messages_parsed++;
    
    switch (header->message_type) {
        case OUCHMessageType::EnterOrder:
            return parse_enter_order(reinterpret_cast<const OUCHEnterOrder*>(buffer));
            
        case OUCHMessageType::Accepted:
            return parse_accepted(reinterpret_cast<const OUCHAccepted*>(buffer));
            
        case OUCHMessageType::Executed:
            return parse_executed(reinterpret_cast<const OUCHExecuted*>(buffer));
            
        default:
            return std::nullopt;
    }
}

std::optional<Order> OUCHParser::parse_enter_order(const OUCHEnterOrder* msg) {
    auto* order = order_pool_.allocate();
    if (!order) {
        return std::nullopt;
    }
    
    uint32_t token = from_be(msg->order_token);
    order->id = token;
    order->symbol = parse_symbol(msg->stock);
    order->side = parse_side(msg->buy_sell_indicator);
    order->quantity = from_be(msg->shares);
    order->remaining_quantity = order->quantity;
    order->price = from_be(msg->price);
    order->timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    order->type = OrderType::LIMIT;
    order->venue = Venue::NASDAQ;
    
    track_order(token, order->symbol, order->side, order->price);
    stats_.orders_entered++;
    
    return *order;
}

std::optional<Order> OUCHParser::parse_accepted(const OUCHAccepted* msg) {
    auto* order = order_pool_.allocate();
    if (!order) {
        return std::nullopt;
    }
    
    order->id = from_be(msg->order_reference_number);
    order->symbol = parse_symbol(msg->stock);
    order->side = parse_side(msg->buy_sell_indicator);
    order->quantity = from_be(msg->shares);
    order->remaining_quantity = order->quantity;
    order->price = from_be(msg->price);
    order->timestamp = convert_timestamp(from_be(msg->timestamp));
    order->type = OrderType::LIMIT;
    order->venue = Venue::NASDAQ;
    
    uint32_t token = from_be(msg->order_token);
    track_order(token, order->symbol, order->side, order->price);
    stats_.orders_accepted++;
    
    return *order;
}

std::optional<Trade> OUCHParser::parse_executed(const OUCHExecuted* msg) {
    auto* trade = trade_pool_.allocate();
    if (!trade) {
        return std::nullopt;
    }
    
    uint32_t token = from_be(msg->order_token);
    auto it = order_map_.find(token);
    if (it == order_map_.end()) {
        trade_pool_.deallocate(trade);
        stats_.parse_errors++;
        return std::nullopt;
    }
    
    trade->order_id = token;
    trade->symbol = it->second.symbol;
    trade->price = from_be(msg->execution_price);
    trade->quantity = from_be(msg->executed_shares);
    trade->timestamp = convert_timestamp(from_be(msg->timestamp));
    trade->aggressor_side = it->second.side;
    trade->venue = Venue::NASDAQ;
    trade->is_hidden = (msg->liquidity_flag == 'R' || msg->liquidity_flag == 'K');
    
    stats_.orders_executed++;
    return *trade;
}

Symbol OUCHParser::parse_symbol(const char stock[8]) {
    // Extract symbol (right-padded with spaces)
    std::string symbol_str;
    for (int i = 0; i < 8 && stock[i] != ' '; ++i) {
        symbol_str += stock[i];
    }
    
    auto it = symbol_map_.find(symbol_str);
    if (it != symbol_map_.end()) {
        return it->second;
    }
    
    static std::atomic<Symbol> next_symbol_id{1};
    Symbol new_id = next_symbol_id.fetch_add(1, std::memory_order_relaxed);
    symbol_map_[symbol_str] = new_id;
    
    return new_id;
}

void OUCHParser::add_symbol_mapping(const std::string& ouch_symbol, Symbol numeric_symbol) {
    symbol_map_[ouch_symbol] = numeric_symbol;
}

void OUCHParser::track_order(uint32_t token, Symbol symbol, Side side, Price price) {
    order_map_[token] = {symbol, side, price, 0};
}

} 
