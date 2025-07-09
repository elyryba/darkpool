#pragma once

#include "darkpool/types.hpp"
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <thread>
#include <unordered_set>
#include <atomic>
#include <queue>

namespace darkpool::visualization {

class WebSocketServer {
public:
    using server = websocketpp::server<websocketpp::config::asio>;
    using connection_hdl = websocketpp::connection_hdl;
    using message_ptr = server::message_ptr;
    
    struct Config {
        uint16_t port = 9002;
        size_t max_clients = 100;
        int64_t update_interval_ms = 100;
        size_t message_queue_size = 10000;
        bool enable_compression = true;
        size_t max_message_size = 1048576; // 1MB
        int64_t client_timeout_ms = 30000;
        bool enable_binary_protocol = true;
        size_t worker_threads = 2;
    };
    
    // Message types for client communication
    enum class MessageType : uint8_t {
        ANOMALY = 1,
        MARKET_DATA = 2,
        EXECUTION_UPDATE = 3,
        PERFORMANCE_METRICS = 4,
        HEATMAP_DATA = 5,
        SYSTEM_STATUS = 6,
        ERROR = 7,
        SUBSCRIPTION = 8
    };
    
    struct ClientMessage {
        MessageType type;
        std::string payload;
        int64_t timestamp;
        uint32_t sequence_num;
    };
    
    explicit WebSocketServer(const Config& config = {});
    ~WebSocketServer();
    
    // Server control
    bool start() noexcept;
    void stop() noexcept;
    bool is_running() const noexcept;
    
    // Data broadcasting
    void broadcast_anomaly(const Anomaly& anomaly) noexcept;
    void broadcast_market_data(const MarketMessage& msg) noexcept;
    void broadcast_execution(const ExecutionOrder& order) noexcept;
    void broadcast_metrics(const std::string& metrics_json) noexcept;
    void broadcast_heatmap(const std::string& heatmap_json) noexcept;
    
    // Client management
    size_t get_client_count() const noexcept;
    void disconnect_client(connection_hdl hdl) noexcept;
    
    // Performance stats
    struct ServerStats {
        std::atomic<uint64_t> messages_sent{0};
        std::atomic<uint64_t> bytes_sent{0};
        std::atomic<uint64_t> messages_dropped{0};
        std::atomic<uint64_t> client_errors{0};
        std::atomic<uint64_t> compression_ratio{0};
        std::atomic<int64_t> avg_latency_us{0};
    };
    
    ServerStats get_stats() const noexcept;
    
private:
    struct ClientInfo {
        connection_hdl handle;
        std::unordered_set<MessageType> subscriptions;
        int64_t last_activity;
        std::atomic<uint32_t> pending_messages{0};
        bool is_slow = false;
    };
    
    struct BroadcastMessage {
        MessageType type;
        std::string data;
        int64_t timestamp;
        std::shared_ptr<std::vector<uint8_t>> binary_data;
    };
    
    // WebSocket handlers
    void on_open(connection_hdl hdl);
    void on_close(connection_hdl hdl);
    void on_message(connection_hdl hdl, message_ptr msg);
    void on_error(connection_hdl hdl);
    
    // Message processing
    void process_client_message(connection_hdl hdl, const std::string& message);
    void handle_subscription(connection_hdl hdl, const std::string& request);
    
    // Broadcasting
    void broadcast_loop();
    void send_to_client(const ClientInfo& client, const BroadcastMessage& msg);
    std::string serialize_message(const BroadcastMessage& msg);
    std::shared_ptr<std::vector<uint8_t>> serialize_binary(const BroadcastMessage& msg);
    
    // Utilities
    std::string create_anomaly_json(const Anomaly& anomaly);
    std::string create_market_json(const MarketMessage& msg);
    std::string create_execution_json(const ExecutionOrder& order);
    void cleanup_slow_clients();
    
    Config config_;
    server ws_server_;
    std::thread server_thread_;
    std::vector<std::thread> broadcast_threads_;
    std::atomic<bool> running_{false};
    
    // Client tracking
    std::unordered_map<connection_hdl, ClientInfo, 
                       std::owner_less<connection_hdl>> clients_;
    mutable std::mutex clients_mutex_;
    
    // Message queue
    std::queue<BroadcastMessage> message_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Statistics
    mutable ServerStats stats_;
    std::atomic<uint32_t> sequence_number_{0};
};

// Specialized JSON serializer for performance
class FastJSONBuilder {
public:
    FastJSONBuilder() { buffer_.reserve(4096); }
    
    FastJSONBuilder& start_object() {
        buffer_.push_back('{');
        return *this;
    }
    
    FastJSONBuilder& end_object() {
        if (buffer_.back() == ',') buffer_.pop_back();
        buffer_.push_back('}');
        return *this;
    }
    
    FastJSONBuilder& key(const std::string& k) {
        buffer_.push_back('"');
        buffer_.append(k);
        buffer_.append("\":");
        return *this;
    }
    
    FastJSONBuilder& value(const std::string& v) {
        buffer_.push_back('"');
        escape_string(v);
        buffer_.push_back('"');
        buffer_.push_back(',');
        return *this;
    }
    
    FastJSONBuilder& value(int64_t v) {
        buffer_.append(std::to_string(v));
        buffer_.push_back(',');
        return *this;
    }
    
    FastJSONBuilder& value(double v) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.6f", v);
        buffer_.append(buf);
        buffer_.push_back(',');
        return *this;
    }
    
    FastJSONBuilder& null() {
        buffer_.append("null,");
        return *this;
    }
    
    FastJSONBuilder& start_array() {
        buffer_.push_back('[');
        return *this;
    }
    
    FastJSONBuilder& end_array() {
        if (buffer_.back() == ',') buffer_.pop_back();
        buffer_.push_back(']');
        buffer_.push_back(',');
        return *this;
    }
    
    std::string build() {
        if (buffer_.back() == ',') buffer_.pop_back();
        return std::move(buffer_);
    }
    
    void reset() {
        buffer_.clear();
    }
    
private:
    std::string buffer_;
    
    void escape_string(const std::string& s) {
        for (char c : s) {
            switch (c) {
                case '"': buffer_.append("\\\""); break;
                case '\\': buffer_.append("\\\\"); break;
                case '\n': buffer_.append("\\n"); break;
                case '\r': buffer_.append("\\r"); break;
                case '\t': buffer_.append("\\t"); break;
                default: buffer_.push_back(c);
            }
        }
    }
};

} 
