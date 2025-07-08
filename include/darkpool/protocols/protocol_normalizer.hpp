#pragma once

#include <memory>
#include <thread>
#include <vector>
#include "darkpool/types.hpp"
#include "darkpool/protocols/fix_parser.hpp"
#include "darkpool/protocols/itch_parser.hpp"
#include "darkpool/protocols/ouch_parser.hpp"
#include "darkpool/utils/lock_free_queue.hpp"
#include "darkpool/utils/ring_buffer.hpp"

namespace darkpool {

// Protocol-agnostic message handler
class MessageHandler {
public:
    virtual ~MessageHandler() = default;
    virtual void on_message(const MarketMessage& message) = 0;
    virtual void on_error(const std::string& error) = 0;
};

// Base class for protocol-specific receivers
class ProtocolReceiver {
public:
    virtual ~ProtocolReceiver() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool is_running() const = 0;
    virtual size_t messages_received() const = 0;
};

// TCP receiver for FIX/OUCH protocols
class TCPReceiver : public ProtocolReceiver {
public:
    TCPReceiver(const std::string& host, uint16_t port, size_t buffer_size = 65536);
    ~TCPReceiver();
    
    void start() override;
    void stop() override;
    bool is_running() const override { return running_.load(); }
    size_t messages_received() const override { return messages_received_.load(); }
    
    // Set message callback
    void set_data_callback(std::function<void(const uint8_t*, size_t)> callback) {
        data_callback_ = callback;
    }
    
private:
    void receive_loop();
    bool connect();
    void disconnect();
    
    std::string host_;
    uint16_t port_;
    size_t buffer_size_;
    int socket_fd_ = -1;
    
    std::atomic<bool> running_{false};
    std::atomic<size_t> messages_received_{0};
    std::thread receive_thread_;
    
    std::function<void(const uint8_t*, size_t)> data_callback_;
    std::vector<uint8_t> receive_buffer_;
};

// UDP multicast receiver for ITCH
class MulticastReceiver : public ProtocolReceiver {
public:
    MulticastReceiver(const std::string& multicast_group, uint16_t port, 
                     const std::string& interface = "");
    ~MulticastReceiver();
    
    void start() override;
    void stop() override;
    bool is_running() const override { return running_.load(); }
    size_t messages_received() const override { return messages_received_.load(); }
    
    void set_data_callback(std::function<void(const uint8_t*, size_t)> callback) {
        data_callback_ = callback;
    }
    
private:
    void receive_loop();
    bool join_multicast_group();
    
    std::string multicast_group_;
    uint16_t port_;
    std::string interface_;
    int socket_fd_ = -1;
    
    std::atomic<bool> running_{false};
    std::atomic<size_t> messages_received_{0};
    std::thread receive_thread_;
    
    std::function<void(const uint8_t*, size_t)> data_callback_;
    std::vector<uint8_t> receive_buffer_;
};

// Main protocol normalizer that handles multiple data sources
class ProtocolNormalizer {
public:
    ProtocolNormalizer(size_t output_queue_size = 1048576);
    ~ProtocolNormalizer();
    
    // Add data sources
    void add_fix_source(const std::string& host, uint16_t port, 
                       const std::vector<std::string>& symbols);
    
    void add_itch_source(const std::string& multicast_group, uint16_t port,
                        const std::vector<std::string>& symbols);
    
    void add_ouch_source(const std::string& host, uint16_t port,
                        const std::vector<std::string>& symbols);
    
    // Start/stop all receivers
    void start();
    void stop();
    bool is_running() const;
    
    // Get normalized messages
    std::optional<MarketMessage> get_message() {
        return output_queue_.dequeue();
    }
    
    // Set handler for normalized messages
    void set_handler(std::shared_ptr<MessageHandler> handler) {
        handler_ = handler;
    }
    
    // Stats
    struct Stats {
        size_t total_messages = 0;
        size_t fix_messages = 0;
        size_t itch_messages = 0;
        size_t ouch_messages = 0;
        size_t parse_errors = 0;
        size_t queue_full_drops = 0;
    };
    
    Stats get_stats() const;
    
private:
    // Protocol-specific message processors
    void process_fix_data(const uint8_t* data, size_t length);
    void process_itch_data(const uint8_t* data, size_t length);
    void process_ouch_data(const uint8_t* data, size_t length);
    
    // Output normalized messages
    void output_message(MarketMessage&& message);
    
    // Memory pools
    std::unique_ptr<MemoryPool<Order>> order_pool_;
    std::unique_ptr<MemoryPool<Trade>> trade_pool_;
    std::unique_ptr<MemoryPool<Quote>> quote_pool_;
    
    // Protocol parsers
    std::unique_ptr<FIXParser> fix_parser_;
    std::unique_ptr<ITCHParser> itch_parser_;
    std::unique_ptr<OUCHParser> ouch_parser_;
    
    // Network receivers
    std::vector<std::unique_ptr<ProtocolReceiver>> receivers_;
    
    // Output queue
    LockFreeQueue<MarketMessage, 1048576> output_queue_;
    
    // Message handler
    std::shared_ptr<MessageHandler> handler_;
    
    // Processing thread
    std::thread processing_thread_;
    std::atomic<bool> processing_{false};
    
    // Stats
    mutable std::mutex stats_mutex_;
    Stats stats_;
    
    // Symbol mappings
    Symbol next_symbol_id_ = 1;
    std::unordered_map<std::string, Symbol> symbol_map_;
    
    Symbol get_or_create_symbol(const std::string& symbol_str);
};

} 
