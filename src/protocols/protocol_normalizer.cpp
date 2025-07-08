#include "darkpool/protocols/protocol_normalizer.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

namespace darkpool {

// TCPReceiver Implementation
TCPReceiver::TCPReceiver(const std::string& host, uint16_t port, size_t buffer_size)
    : host_(host), port_(port), buffer_size_(buffer_size) {
    receive_buffer_.resize(buffer_size);
}

TCPReceiver::~TCPReceiver() {
    stop();
}

void TCPReceiver::start() {
    if (running_.load()) return;
    
    if (!connect()) {
        throw std::runtime_error("Failed to connect to " + host_ + ":" + std::to_string(port_));
    }
    
    running_.store(true);
    receive_thread_ = std::thread(&TCPReceiver::receive_loop, this);
}

void TCPReceiver::stop() {
    running_.store(false);
    disconnect();
    
    if (receive_thread_.joinable()) {
        receive_thread_.join();
    }
}

bool TCPReceiver::connect() {
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) return false;
    
    // Set non-blocking
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
    
    // Set TCP_NODELAY
    int nodelay = 1;
    setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    
    // Set receive buffer size
    int rcvbuf = buffer_size_;
    setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
    
    if (::connect(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (errno != EINPROGRESS) {
            close(socket_fd_);
            socket_fd_ = -1;
            return false;
        }
    }
    
    return true;
}

void TCPReceiver::disconnect() {
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
}

void TCPReceiver::receive_loop() {
    while (running_.load()) {
        ssize_t bytes_received = recv(socket_fd_, receive_buffer_.data(), 
                                     receive_buffer_.size(), 0);
        
        if (bytes_received > 0) {
            messages_received_.fetch_add(1, std::memory_order_relaxed);
            
            if (data_callback_) {
                data_callback_(receive_buffer_.data(), bytes_received);
            }
        } else if (bytes_received == 0) {
            // Connection closed
            break;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            // Error
            break;
        }
    }
}

// MulticastReceiver Implementation
MulticastReceiver::MulticastReceiver(const std::string& multicast_group, 
                                   uint16_t port, const std::string& interface)
    : multicast_group_(multicast_group), port_(port), interface_(interface) {
    receive_buffer_.resize(65536);
}

MulticastReceiver::~MulticastReceiver() {
    stop();
}

void MulticastReceiver::start() {
    if (running_.load()) return;
    
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        throw std::runtime_error("Failed to create multicast socket");
    }
    
    // Allow multiple receivers
    int reuse = 1;
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    // Bind to port
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    addr.sin_addr.s_addr = INADDR_ANY;
    
    if (bind(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(socket_fd_);
        throw std::runtime_error("Failed to bind multicast socket");
    }
    
    if (!join_multicast_group()) {
        close(socket_fd_);
        throw std::runtime_error("Failed to join multicast group");
    }
    
    running_.store(true);
    receive_thread_ = std::thread(&MulticastReceiver::receive_loop, this);
}

void MulticastReceiver::stop() {
    running_.store(false);
    
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
    
    if (receive_thread_.joinable()) {
        receive_thread_.join();
    }
}

bool MulticastReceiver::join_multicast_group() {
    struct ip_mreq mreq{};
    inet_pton(AF_INET, multicast_group_.c_str(), &mreq.imr_multiaddr);
    
    if (!interface_.empty()) {
        inet_pton(AF_INET, interface_.c_str(), &mreq.imr_interface);
    } else {
        mreq.imr_interface.s_addr = INADDR_ANY;
    }
    
    return setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, 
                     &mreq, sizeof(mreq)) == 0;
}

void MulticastReceiver::receive_loop() {
    while (running_.load()) {
        ssize_t bytes_received = recv(socket_fd_, receive_buffer_.data(), 
                                     receive_buffer_.size(), 0);
        
        if (bytes_received > 0) {
            messages_received_.fetch_add(1, std::memory_order_relaxed);
            
            if (data_callback_) {
                data_callback_(receive_buffer_.data(), bytes_received);
            }
        }
    }
}

// ProtocolNormalizer Implementation
ProtocolNormalizer::ProtocolNormalizer(size_t output_queue_size) {
    // Initialize memory pools
    order_pool_ = std::make_unique<MemoryPool<Order>>(100000);
    trade_pool_ = std::make_unique<MemoryPool<Trade>>(100000);
    quote_pool_ = std::make_unique<MemoryPool<Quote>>(50000);
    
    // Initialize parsers
    fix_parser_ = std::make_unique<FIXParser>(*order_pool_, *trade_pool_);
    itch_parser_ = std::make_unique<ITCHParser>(*order_pool_, *trade_pool_);
    ouch_parser_ = std::make_unique<OUCHParser>(*order_pool_, *trade_pool_);
}

ProtocolNormalizer::~ProtocolNormalizer() {
    stop();
}

void ProtocolNormalizer::add_fix_source(const std::string& host, uint16_t port,
                                       const std::vector<std::string>& symbols) {
    auto receiver = std::make_unique<TCPReceiver>(host, port);
    
    receiver->set_data_callback([this](const uint8_t* data, size_t length) {
        process_fix_data(data, length);
    });
    
    // Add symbol mappings
    for (const auto& symbol : symbols) {
        Symbol id = get_or_create_symbol(symbol);
        fix_parser_->add_symbol_mapping(symbol, id);
    }
    
    receivers_.push_back(std::move(receiver));
}

void ProtocolNormalizer::add_itch_source(const std::string& multicast_group, uint16_t port,
                                        const std::vector<std::string>& symbols) {
    auto receiver = std::make_unique<MulticastReceiver>(multicast_group, port);
    
    receiver->set_data_callback([this](const uint8_t* data, size_t length) {
        process_itch_data(data, length);
    });
    
    receivers_.push_back(std::move(receiver));
}

void ProtocolNormalizer::add_ouch_source(const std::string& host, uint16_t port,
                                        const std::vector<std::string>& symbols) {
    auto receiver = std::make_unique<TCPReceiver>(host, port);
    
    receiver->set_data_callback([this](const uint8_t* data, size_t length) {
        process_ouch_data(data, length);
    });
    
    // Add symbol mappings
    for (const auto& symbol : symbols) {
        Symbol id = get_or_create_symbol(symbol);
        ouch_parser_->add_symbol_mapping(symbol, id);
    }
    
    receivers_.push_back(std::move(receiver));
}

void ProtocolNormalizer::start() {
    for (auto& receiver : receivers_) {
        receiver->start();
    }
    
    processing_.store(true);
    processing_thread_ = std::thread([this] {
        while (processing_.load()) {
            // Process messages from queue if using async processing
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
}

void ProtocolNormalizer::stop() {
    processing_.store(false);
    
    for (auto& receiver : receivers_) {
        receiver->stop();
    }
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

bool ProtocolNormalizer::is_running() const {
    for (const auto& receiver : receivers_) {
        if (receiver->is_running()) return true;
    }
    return false;
}

void ProtocolNormalizer::process_fix_data(const uint8_t* data, size_t length) {
    auto message = fix_parser_->parse(reinterpret_cast<const char*>(data), length);
    if (message) {
        output_message(std::move(*message));
        stats_.fix_messages++;
    } else {
        stats_.parse_errors++;
    }
}

void ProtocolNormalizer::process_itch_data(const uint8_t* data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        size_t bytes_consumed = length - offset;
        auto message = itch_parser_->parse(data + offset, bytes_consumed);
        
        if (message) {
            output_message(std::move(*message));
            stats_.itch_messages++;
            offset += bytes_consumed;
        } else {
            break;
        }
    }
}

void ProtocolNormalizer::process_ouch_data(const uint8_t* data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        size_t bytes_consumed = length - offset;
        auto message = ouch_parser_->parse(data + offset, bytes_consumed);
        
        if (message) {
            output_message(std::move(*message));
            stats_.ouch_messages++;
            offset += bytes_consumed;
        } else {
            break;
        }
    }
}

void ProtocolNormalizer::output_message(MarketMessage&& message) {
    stats_.total_messages++;
    
    if (!output_queue_.enqueue(std::move(message))) {
        stats_.queue_full_drops++;
    }
    
    if (handler_) {
        handler_->on_message(message);
    }
}

Symbol ProtocolNormalizer::get_or_create_symbol(const std::string& symbol_str) {
    auto it = symbol_map_.find(symbol_str);
    if (it != symbol_map_.end()) {
        return it->second;
    }
    
    Symbol new_id = next_symbol_id_++;
    symbol_map_[symbol_str] = new_id;
    return new_id;
}

ProtocolNormalizer::Stats ProtocolNormalizer::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

} 
