#include "darkpool/visualization/websocket_server.hpp"
#include <chrono>
#include <sstream>

namespace darkpool::visualization {

WebSocketServer::WebSocketServer(const Config& config)
    : config_(config) {
    
    // Configure server
    ws_server_.set_access_channels(websocketpp::log::alevel::none);
    ws_server_.set_error_channels(websocketpp::log::elevel::warn);
    
    // Set handlers
    ws_server_.set_open_handler(
        [this](connection_hdl hdl) { on_open(hdl); });
    ws_server_.set_close_handler(
        [this](connection_hdl hdl) { on_close(hdl); });
    ws_server_.set_message_handler(
        [this](connection_hdl hdl, message_ptr msg) { on_message(hdl, msg); });
    ws_server_.set_fail_handler(
        [this](connection_hdl hdl) { on_error(hdl); });
    
    // Initialize Asio
    ws_server_.init_asio();
    ws_server_.set_reuse_addr(true);
    
    // Configure compression if enabled
    if (config_.enable_compression) {
        // WebSocket++ compression settings would go here
    }
}

WebSocketServer::~WebSocketServer() {
    stop();
}

bool WebSocketServer::start() noexcept {
    if (running_.exchange(true)) {
        return false; // Already running
    }
    
    try {
        // Listen on port
        ws_server_.listen(config_.port);
        ws_server_.start_accept();
        
        // Start server thread
        server_thread_ = std::thread([this]() {
            try {
                ws_server_.run();
            } catch (const std::exception& e) {
                // Log error
                running_ = false;
            }
        });
        
        // Start broadcast threads
        for (size_t i = 0; i < config_.worker_threads; ++i) {
            broadcast_threads_.emplace_back(&WebSocketServer::broadcast_loop, this);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        running_ = false;
        return false;
    }
}

void WebSocketServer::stop() noexcept {
    if (!running_.exchange(false)) {
        return;
    }
    
    // Stop accepting new connections
    ws_server_.stop_listening();
    
    // Close all connections
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto& [hdl, client] : clients_) {
            try {
                ws_server_.close(hdl, websocketpp::close::status::going_away, "Server shutting down");
            } catch (...) {}
        }
        clients_.clear();
    }
    
    // Stop server
    ws_server_.stop();
    
    // Wake up broadcast threads
    queue_cv_.notify_all();
    
    // Join threads
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    
    for (auto& thread : broadcast_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

bool WebSocketServer::is_running() const noexcept {
    return running_.load();
}

void WebSocketServer::broadcast_anomaly(const Anomaly& anomaly) noexcept {
    if (!running_) return;
    
    BroadcastMessage msg;
    msg.type = MessageType::ANOMALY;
    msg.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    msg.data = create_anomaly_json(anomaly);
    
    // Add to queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (message_queue_.size() < config_.message_queue_size) {
            message_queue_.push(msg);
            queue_cv_.notify_one();
        } else {
            stats_.messages_dropped.fetch_add(1);
        }
    }
}

void WebSocketServer::broadcast_market_data(const MarketMessage& market_msg) noexcept {
    if (!running_) return;
    
    BroadcastMessage msg;
    msg.type = MessageType::MARKET_DATA;
    msg.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    msg.data = create_market_json(market_msg);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (message_queue_.size() < config_.message_queue_size) {
            message_queue_.push(msg);
            queue_cv_.notify_one();
        } else {
            stats_.messages_dropped.fetch_add(1);
        }
    }
}

void WebSocketServer::broadcast_heatmap(const std::string& heatmap_json) noexcept {
    if (!running_) return;
    
    BroadcastMessage msg;
    msg.type = MessageType::HEATMAP_DATA;
    msg.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    msg.data = heatmap_json;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (message_queue_.size() < config_.message_queue_size) {
            message_queue_.push(msg);
            queue_cv_.notify_one();
        }
    }
}

size_t WebSocketServer::get_client_count() const noexcept {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return clients_.size();
}

WebSocketServer::ServerStats WebSocketServer::get_stats() const noexcept {
    return stats_;
}

void WebSocketServer::on_open(connection_hdl hdl) {
    auto conn = ws_server_.get_con_from_hdl(hdl);
    
    ClientInfo client;
    client.handle = hdl;
    client.last_activity = std::chrono::high_resolution_clock::now()
                          .time_since_epoch().count();
    
    // Default subscriptions
    client.subscriptions.insert(MessageType::ANOMALY);
    client.subscriptions.insert(MessageType::SYSTEM_STATUS);
    
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        if (clients_.size() >= config_.max_clients) {
            // Reject connection
            ws_server_.close(hdl, websocketpp::close::status::try_again_later, 
                           "Server full");
            return;
        }
        clients_[hdl] = client;
    }
    
    // Send welcome message
    FastJSONBuilder json;
    json.start_object()
        .key("type").value("welcome")
        .key("server_time").value(client.last_activity)
        .key("version").value("1.0.0")
        .end_object();
    
    try {
        ws_server_.send(hdl, json.build(), websocketpp::frame::opcode::text);
    } catch (...) {}
}

void WebSocketServer::on_close(connection_hdl hdl) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    clients_.erase(hdl);
}

void WebSocketServer::on_message(connection_hdl hdl, message_ptr msg) {
    // Update client activity
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto it = clients_.find(hdl);
        if (it != clients_.end()) {
            it->second.last_activity = std::chrono::high_resolution_clock::now()
                                      .time_since_epoch().count();
        }
    }
    
    // Process message
    try {
        process_client_message(hdl, msg->get_payload());
    } catch (const std::exception& e) {
        stats_.client_errors.fetch_add(1);
    }
}

void WebSocketServer::on_error(connection_hdl hdl) {
    stats_.client_errors.fetch_add(1);
    
    std::lock_guard<std::mutex> lock(clients_mutex_);
    clients_.erase(hdl);
}

void WebSocketServer::process_client_message(connection_hdl hdl, 
                                           const std::string& message) {
    // Simple JSON parsing (in production would use proper parser)
    if (message.find("\"type\":\"subscribe\"") != std::string::npos) {
        handle_subscription(hdl, message);
    } else if (message.find("\"type\":\"ping\"") != std::string::npos) {
        // Send pong
        try {
            ws_server_.send(hdl, "{\"type\":\"pong\"}", websocketpp::frame::opcode::text);
        } catch (...) {}
    }
}

void WebSocketServer::handle_subscription(connection_hdl hdl, 
                                        const std::string& request) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = clients_.find(hdl);
    if (it == clients_.end()) return;
    
    // Parse subscription request (simplified)
    if (request.find("\"channel\":\"market_data\"") != std::string::npos) {
        it->second.subscriptions.insert(MessageType::MARKET_DATA);
    }
    if (request.find("\"channel\":\"execution\"") != std::string::npos) {
        it->second.subscriptions.insert(MessageType::EXECUTION_UPDATE);
    }
    if (request.find("\"channel\":\"heatmap\"") != std::string::npos) {
        it->second.subscriptions.insert(MessageType::HEATMAP_DATA);
    }
    
    // Send confirmation
    FastJSONBuilder json;
    json.start_object()
        .key("type").value("subscription_confirmed")
        .key("channels").start_array();
    
    for (auto type : it->second.subscriptions) {
        json.value(static_cast<int>(type));
    }
    
    json.end_array().end_object();
    
    try {
        ws_server_.send(hdl, json.build(), websocketpp::frame::opcode::text);
    } catch (...) {}
}

void WebSocketServer::broadcast_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for messages
        queue_cv_.wait(lock, [this]() {
            return !message_queue_.empty() || !running_;
        });
        
        if (!running_) break;
        
        // Process batch of messages
        std::vector<BroadcastMessage> batch;
        size_t batch_size = std::min(size_t(100), message_queue_.size());
        
        for (size_t i = 0; i < batch_size && !message_queue_.empty(); ++i) {
            batch.push_back(std::move(message_queue_.front()));
            message_queue_.pop();
        }
        
        lock.unlock();
        
        // Send to clients
        std::lock_guard<std::mutex> client_lock(clients_mutex_);
        
        for (const auto& msg : batch) {
            // Serialize once
            std::string serialized;
            std::shared_ptr<std::vector<uint8_t>> binary;
            
            if (config_.enable_binary_protocol) {
                binary = serialize_binary(msg);
            } else {
                serialized = serialize_message(msg);
            }
            
            // Send to subscribed clients
            for (auto& [hdl, client] : clients_) {
                if (client.subscriptions.count(msg.type) > 0 && !client.is_slow) {
                    try {
                        if (binary) {
                            ws_server_.send(hdl, binary->data(), binary->size(),
                                          websocketpp::frame::opcode::binary);
                        } else {
                            ws_server_.send(hdl, serialized, 
                                          websocketpp::frame::opcode::text);
                        }
                        
                        stats_.messages_sent.fetch_add(1);
                        stats_.bytes_sent.fetch_add(
                            binary ? binary->size() : serialized.size()
                        );
                        
                    } catch (const websocketpp::exception& e) {
                        // Mark as slow client
                        client.is_slow = true;
                        stats_.client_errors.fetch_add(1);
                    }
                }
            }
        }
        
        // Periodic cleanup
        static auto last_cleanup = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - last_cleanup > std::chrono::seconds(10)) {
            cleanup_slow_clients();
            last_cleanup = std::chrono::steady_clock::now();
        }
    }
}

std::string WebSocketServer::serialize_message(const BroadcastMessage& msg) {
    FastJSONBuilder json;
    json.start_object()
        .key("type").value(static_cast<int>(msg.type))
        .key("timestamp").value(msg.timestamp)
        .key("sequence").value(sequence_number_.fetch_add(1))
        .key("data");
    
    // Data is already JSON
    return json.build().substr(0, json.build().size() - 1) + msg.data + "}";
}

std::shared_ptr<std::vector<uint8_t>> WebSocketServer::serialize_binary(
    const BroadcastMessage& msg) {
    
    auto buffer = std::make_shared<std::vector<uint8_t>>();
    buffer->reserve(msg.data.size() + 32);
    
    // Simple binary protocol: [type:1][seq:4][timestamp:8][length:4][data:N]
    buffer->push_back(static_cast<uint8_t>(msg.type));
    
    uint32_t seq = sequence_number_.fetch_add(1);
    buffer->insert(buffer->end(), reinterpret_cast<uint8_t*>(&seq), 
                  reinterpret_cast<uint8_t*>(&seq) + 4);
    
    buffer->insert(buffer->end(), reinterpret_cast<const uint8_t*>(&msg.timestamp),
                  reinterpret_cast<const uint8_t*>(&msg.timestamp) + 8);
    
    uint32_t length = msg.data.size();
    buffer->insert(buffer->end(), reinterpret_cast<uint8_t*>(&length),
                  reinterpret_cast<uint8_t*>(&length) + 4);
    
    buffer->insert(buffer->end(), msg.data.begin(), msg.data.end());
    
    return buffer;
}

std::string WebSocketServer::create_anomaly_json(const Anomaly& anomaly) {
    FastJSONBuilder json;
    json.start_object()
        .key("symbol").value(static_cast<int64_t>(anomaly.symbol))
        .key("timestamp").value(anomaly.timestamp)
        .key("type").value(static_cast<int>(anomaly.type))
        .key("confidence").value(anomaly.confidence)
        .key("magnitude").value(anomaly.magnitude)
        .key("metadata").start_object();
    
    for (const auto& [key, value] : anomaly.metadata) {
        json.key(key).value(value);
    }
    
    json.end_object().end_object();
    
    return json.build();
}

std::string WebSocketServer::create_market_json(const MarketMessage& msg) {
    FastJSONBuilder json;
    json.start_object();
    
    switch (msg.index()) {
        case 0: { // Quote
            const auto& quote = std::get<Quote>(msg);
            json.key("msg_type").value("quote")
                .key("symbol").value(static_cast<int64_t>(quote.symbol))
                .key("bid_price").value(quote.bid_price / 10000.0)
                .key("ask_price").value(quote.ask_price / 10000.0)
                .key("bid_size").value(static_cast<int64_t>(quote.bid_size))
                .key("ask_size").value(static_cast<int64_t>(quote.ask_size))
                .key("timestamp").value(quote.timestamp);
            break;
        }
        case 1: { // Trade
            const auto& trade = std::get<Trade>(msg);
            json.key("msg_type").value("trade")
                .key("symbol").value(static_cast<int64_t>(trade.symbol))
                .key("price").value(trade.price / 10000.0)
                .key("quantity").value(static_cast<int64_t>(trade.quantity))
                .key("side").value(trade.side == Side::BUY ? "BUY" : "SELL")
                .key("timestamp").value(trade.timestamp);
            break;
        }
    }
    
    json.end_object();
    return json.build();
}

std::string WebSocketServer::create_execution_json(const ExecutionOrder& order) {
    FastJSONBuilder json;
    json.start_object()
        .key("order_id").value(static_cast<int64_t>(order.order_id))
        .key("symbol").value(static_cast<int64_t>(order.symbol))
        .key("side").value(order.side == Side::BUY ? "BUY" : "SELL")
        .key("quantity").value(static_cast<int64_t>(order.quantity))
        .key("filled").value(static_cast<int64_t>(order.filled_quantity))
        .key("venue").value(order.venue_id)
        .key("avg_price").value(order.avg_fill_price)
        .key("status").value(order.filled_quantity >= order.quantity ? "FILLED" : "PARTIAL")
        .end_object();
    
    return json.build();
}

void WebSocketServer::cleanup_slow_clients() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::vector<connection_hdl> to_disconnect;
    
    for (auto& [hdl, client] : clients_) {
        if (client.is_slow || now - client.last_activity > config_.client_timeout_ms * 1000000) {
            to_disconnect.push_back(hdl);
        }
    }
    
    for (auto hdl : to_disconnect) {
        try {
            ws_server_.close(hdl, websocketpp::close::status::going_away, "Timeout");
        } catch (...) {}
        clients_.erase(hdl);
    }
}

} 
