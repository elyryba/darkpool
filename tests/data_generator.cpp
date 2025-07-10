#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <vector>
#include <map>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <arpa/inet.h>
#include "darkpool/types.hpp"

using namespace darkpool;

class MarketDataGenerator {
public:
    struct Config {
        std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"};
        int duration_seconds = 3600;  // 1 hour
        int messages_per_second = 10000;
        double anomaly_rate = 0.02;  // 2%
        std::string output_dir = "generated_data/";
        
        // Market conditions
        enum MarketCondition {
            NORMAL,
            TRENDING,
            VOLATILE,
            CALM
        } market_condition = NORMAL;
        
        // Anomaly types to inject
        bool enable_iceberg = true;
        bool enable_spoofing = true;
        bool enable_dark_pool = true;
        bool enable_sweep = true;
    };
    
    MarketDataGenerator(const Config& config) 
        : config_(config), gen_(std::random_device{}()) {
        
        // Initialize symbol data
        for (const auto& symbol : config.symbols) {
            SymbolData data;
            data.symbol = symbol;
            data.base_price = 100.0 + std::hash<std::string>{}(symbol) % 400;
            data.volatility = 0.001 + (std::hash<std::string>{}(symbol) % 10) * 0.001;
            data.avg_spread = 0.01 + (std::hash<std::string>{}(symbol) % 5) * 0.01;
            data.avg_trade_size = 100 + (std::hash<std::string>{}(symbol) % 9) * 100;
            data.current_price = data.base_price;
            data.bid_price = data.current_price - data.avg_spread / 2;
            data.ask_price = data.current_price + data.avg_spread / 2;
            
            symbol_data_[symbol] = data;
        }
        
        // Create output directory
        std::system(("mkdir -p " + config_.output_dir).c_str());
    }
    
    void generate() {
        std::cout << "Generating market data...\n";
        std::cout << "Symbols: " << config_.symbols.size() << "\n";
        std::cout << "Duration: " << config_.duration_seconds << " seconds\n";
        std::cout << "Rate: " << config_.messages_per_second << " msgs/sec\n";
        std::cout << "Anomaly rate: " << config_.anomaly_rate * 100 << "%\n";
        
        // Open output files
        std::ofstream fix_file(config_.output_dir + "messages.fix");
        std::ofstream itch_file(config_.output_dir + "messages.itch", std::ios::binary);
        std::ofstream ouch_file(config_.output_dir + "messages.ouch", std::ios::binary);
        std::ofstream anomaly_file(config_.output_dir + "anomalies.csv");
        
        // Write anomaly header
        anomaly_file << "timestamp,symbol,type,description,severity\n";
        
        // Generate messages
        auto start_time = std::chrono::system_clock::now();
        int total_messages = 0;
        int total_anomalies = 0;
        
        for (int second = 0; second < config_.duration_seconds; ++second) {
            auto current_time = start_time + std::chrono::seconds(second);
            
            // Update market conditions
            update_market_conditions(second);
            
            // Generate messages for this second
            for (int i = 0; i < config_.messages_per_second; ++i) {
                auto timestamp = current_time + 
                    std::chrono::microseconds(i * 1000000 / config_.messages_per_second);
                
                // Select random symbol
                const auto& symbol = config_.symbols[gen_() % config_.symbols.size()];
                
                // Decide message type
                double r = uniform_dist_(gen_);
                
                if (r < 0.7) {
                    // 70% quotes
                    generate_quote(symbol, timestamp, fix_file, itch_file);
                } else if (r < 0.9) {
                    // 20% trades
                    generate_trade(symbol, timestamp, fix_file, itch_file);
                } else {
                    // 10% orders
                    generate_order(symbol, timestamp, fix_file, ouch_file);
                }
                
                // Inject anomalies
                if (uniform_dist_(gen_) < config_.anomaly_rate) {
                    inject_anomaly(symbol, timestamp, fix_file, itch_file, anomaly_file);
                    total_anomalies++;
                }
                
                total_messages++;
            }
            
            // Progress update
            if (second % 60 == 0) {
                std::cout << "Progress: " << second << "/" << config_.duration_seconds 
                          << " seconds\n";
            }
        }
        
        std::cout << "\nGeneration complete!\n";
        std::cout << "Total messages: " << total_messages << "\n";
        std::cout << "Total anomalies: " << total_anomalies << "\n";
        std::cout << "Files written to: " << config_.output_dir << "\n";
    }
    
private:
    struct SymbolData {
        std::string symbol;
        double base_price;
        double current_price;
        double bid_price;
        double ask_price;
        double volatility;
        double avg_spread;
        double avg_trade_size;
        uint64_t sequence_number = 1;
        uint64_t order_id = 1;
    };
    
    void update_market_conditions(int second) {
        // Update prices based on market condition
        for (auto& [symbol, data] : symbol_data_) {
            double drift = 0.0;
            double vol_mult = 1.0;
            
            switch (config_.market_condition) {
                case Config::TRENDING:
                    drift = 0.0001 * ((second / 60) % 2 == 0 ? 1 : -1);
                    break;
                case Config::VOLATILE:
                    vol_mult = 3.0;
                    break;
                case Config::CALM:
                    vol_mult = 0.3;
                    break;
                default:
                    break;
            }
            
            // Random walk with drift
            std::normal_distribution<double> dist(drift, data.volatility * vol_mult);
            double change = dist(gen_);
            
            data.current_price *= (1.0 + change);
            data.current_price = std::max(0.01, data.current_price); // Floor at 1 cent
            
            // Update bid/ask
            double spread = data.avg_spread * (1.0 + 0.2 * uniform_dist_(gen_));
            data.bid_price = data.current_price - spread / 2;
            data.ask_price = data.current_price + spread / 2;
        }
    }
    
    void generate_quote(const std::string& symbol,
                       const std::chrono::system_clock::time_point& timestamp,
                       std::ofstream& fix_file,
                       std::ofstream& itch_file) {
        auto& data = symbol_data_[symbol];
        
        // Randomize sizes
        uint32_t bid_size = 100 + gen_() % 9900;
        uint32_t ask_size = 100 + gen_() % 9900;
        
        // Write FIX quote
        write_fix_quote(fix_file, symbol, data.bid_price, data.ask_price, 
                       bid_size, ask_size, timestamp);
        
        // Write ITCH Add Order messages (bid and ask)
        write_itch_add_order(itch_file, symbol, true, data.bid_price, bid_size, 
                            data.order_id++, timestamp);
        write_itch_add_order(itch_file, symbol, false, data.ask_price, ask_size, 
                            data.order_id++, timestamp);
    }
    
    void generate_trade(const std::string& symbol,
                       const std::chrono::system_clock::time_point& timestamp,
                       std::ofstream& fix_file,
                       std::ofstream& itch_file) {
        auto& data = symbol_data_[symbol];
        
        // Trade at mid or slightly through
        double price = data.current_price + 
                      (uniform_dist_(gen_) - 0.5) * data.avg_spread;
        
        // Trade size (exponential distribution)
        std::exponential_distribution<double> size_dist(1.0 / data.avg_trade_size);
        uint32_t size = std::max(1u, static_cast<uint32_t>(size_dist(gen_)));
        
        // Write FIX execution report
        write_fix_trade(fix_file, symbol, price, size, timestamp);
        
        // Write ITCH trade
        write_itch_trade(itch_file, symbol, price, size, timestamp);
    }
    
    void generate_order(const std::string& symbol,
                       const std::chrono::system_clock::time_point& timestamp,
                       std::ofstream& fix_file,
                       std::ofstream& ouch_file) {
        auto& data = symbol_data_[symbol];
        
        // Order type distribution
        bool is_buy = uniform_dist_(gen_) < 0.5;
        double price = is_buy ? data.bid_price : data.ask_price;
        
        // Add some price improvement
        if (uniform_dist_(gen_) < 0.3) {
            price += (is_buy ? 0.01 : -0.01);
        }
        
        uint32_t size = 100 + gen_() % 900;
        
        // Write FIX new order
        write_fix_new_order(fix_file, symbol, is_buy, price, size, 
                           "ORD" + std::to_string(data.order_id++), timestamp);
        
        // Write OUCH order
        write_ouch_order(ouch_file, symbol, is_buy, price, size, timestamp);
    }
    
    void inject_anomaly(const std::string& symbol,
                       const std::chrono::system_clock::time_point& timestamp,
                       std::ofstream& fix_file,
                       std::ofstream& itch_file,
                       std::ofstream& anomaly_file) {
        // Select anomaly type
        std::vector<std::string> enabled_anomalies;
        if (config_.enable_iceberg) enabled_anomalies.push_back("iceberg");
        if (config_.enable_spoofing) enabled_anomalies.push_back("spoofing");
        if (config_.enable_dark_pool) enabled_anomalies.push_back("darkpool");
        if (config_.enable_sweep) enabled_anomalies.push_back("sweep");
        
        if (enabled_anomalies.empty()) return;
        
        const auto& anomaly_type = enabled_anomalies[gen_() % enabled_anomalies.size()];
        auto& data = symbol_data_[symbol];
        
        if (anomaly_type == "iceberg") {
            inject_iceberg_order(symbol, timestamp, fix_file, itch_file, anomaly_file);
        } else if (anomaly_type == "spoofing") {
            inject_spoofing(symbol, timestamp, fix_file, itch_file, anomaly_file);
        } else if (anomaly_type == "darkpool") {
            inject_dark_pool_print(symbol, timestamp, fix_file, itch_file, anomaly_file);
        } else if (anomaly_type == "sweep") {
            inject_sweep(symbol, timestamp, fix_file, itch_file, anomaly_file);
        }
    }
    
    void inject_iceberg_order(const std::string& symbol,
                             const std::chrono::system_clock::time_point& timestamp,
                             std::ofstream& fix_file,
                             std::ofstream& itch_file,
                             std::ofstream& anomaly_file) {
        auto& data = symbol_data_[symbol];
        
        // Large hidden order with small visible part
        uint32_t total_size = 10000 + gen_() % 40000;
        uint32_t visible_size = 100 + gen_() % 400;
        bool is_buy = uniform_dist_(gen_) < 0.5;
        double price = is_buy ? data.bid_price : data.ask_price;
        
        // Log anomaly
        log_anomaly(anomaly_file, timestamp, symbol, "ICEBERG", 
                   "Hidden order " + std::to_string(total_size) + 
                   " shares, showing " + std::to_string(visible_size), "HIGH");
        
        // Generate visible part
        write_itch_add_order(itch_file, symbol, is_buy, price, visible_size,
                            data.order_id++, timestamp);
        
        // Simulate refills
        uint32_t remaining = total_size - visible_size;
        auto refill_time = timestamp;
        
        while (remaining > 0) {
            refill_time += std::chrono::milliseconds(100 + gen_() % 900);
            uint32_t refill_size = std::min(visible_size, remaining);
            
            // Execute visible part
            write_itch_trade(itch_file, symbol, price, visible_size, refill_time);
            
            // Add new visible part
            if (remaining > visible_size) {
                write_itch_add_order(itch_file, symbol, is_buy, price, refill_size,
                                    data.order_id++, refill_time);
            }
            
            remaining -= refill_size;
        }
    }
    
    void inject_spoofing(const std::string& symbol,
                        const std::chrono::system_clock::time_point& timestamp,
                        std::ofstream& fix_file,
                        std::ofstream& itch_file,
                        std::ofstream& anomaly_file) {
        auto& data = symbol_data_[symbol];
        
        // Place large orders away from market
        bool spoof_buy_side = uniform_dist_(gen_) < 0.5;
        
        log_anomaly(anomaly_file, timestamp, symbol, "SPOOFING",
                   "Layered orders on " + std::string(spoof_buy_side ? "bid" : "ask"), 
                   "MEDIUM");
        
        // Create layers
        for (int i = 0; i < 5; ++i) {
            double price = spoof_buy_side 
                ? data.bid_price - 0.02 * (i + 1)
                : data.ask_price + 0.02 * (i + 1);
            
            uint32_t size = 5000 + gen_() % 5000;
            uint64_t order_id = data.order_id++;
            
            // Add order
            write_itch_add_order(itch_file, symbol, spoof_buy_side, price, size,
                                order_id, timestamp);
            
            // Cancel after short time
            auto cancel_time = timestamp + std::chrono::milliseconds(500 + gen_() % 1500);
            write_itch_order_cancel(itch_file, symbol, order_id, size, cancel_time);
        }
    }
    
    void inject_dark_pool_print(const std::string& symbol,
                               const std::chrono::system_clock::time_point& timestamp,
                               std::ofstream& fix_file,
                               std::ofstream& itch_file,
                               std::ofstream& anomaly_file) {
        auto& data = symbol_data_[symbol];
        
        // Large trade at mid
        uint32_t size = 50000 + gen_() % 150000;
        double price = data.current_price;
        
        log_anomaly(anomaly_file, timestamp, symbol, "DARK_POOL",
                   "Large print " + std::to_string(size) + " @ " + 
                   std::to_string(price), "HIGH");
        
        // Write as trade with special condition
        write_fix_trade(fix_file, symbol, price, size, timestamp, "4"); // Dark pool
        write_itch_trade(itch_file, symbol, price, size, timestamp);
    }
    
    void inject_sweep(const std::string& symbol,
                     const std::chrono::system_clock::time_point& timestamp,
                     std::ofstream& fix_file,
                     std::ofstream& itch_file,
                     std::ofstream& anomaly_file) {
        auto& data = symbol_data_[symbol];
        
        // Rapid series of aggressive orders
        bool sweep_buy = uniform_dist_(gen_) < 0.5;
        
        log_anomaly(anomaly_file, timestamp, symbol, "SWEEP",
                   std::string(sweep_buy ? "Buy" : "Sell") + " side sweep", "HIGH");
        
        // Generate rapid trades
        auto sweep_time = timestamp;
        for (int i = 0; i < 20; ++i) {
            double price = sweep_buy 
                ? data.ask_price + 0.01 * i
                : data.bid_price - 0.01 * i;
            
            uint32_t size = 1000 + gen_() % 4000;
            
            write_itch_trade(itch_file, symbol, price, size, sweep_time);
            write_fix_trade(fix_file, symbol, price, size, sweep_time);
            
            sweep_time += std::chrono::microseconds(100 + gen_() % 400);
        }
        
        // Update price after sweep
        data.current_price += (sweep_buy ? 0.20 : -0.20);
    }
    
    // FIX message writers
    void write_fix_quote(std::ofstream& file,
                        const std::string& symbol,
                        double bid, double ask,
                        uint32_t bid_size, uint32_t ask_size,
                        const std::chrono::system_clock::time_point& timestamp) {
        std::stringstream msg;
        msg << "8=FIX.4.2\x01"
            << "35=S\x01"  // Quote
            << "49=GENERATOR\x01"
            << "56=DARKPOOL\x01"
            << "34=" << fix_sequence_++ << "\x01"
            << "52=" << format_timestamp(timestamp) << "\x01"
            << "55=" << symbol << "\x01"
            << "132=" << std::fixed << std::setprecision(2) << bid << "\x01"
            << "133=" << std::fixed << std::setprecision(2) << ask << "\x01"
            << "134=" << bid_size << "\x01"
            << "135=" << ask_size << "\x01";
        
        add_fix_checksum(msg);
        file << msg.str() << "\n";
    }
    
    void write_fix_trade(std::ofstream& file,
                        const std::string& symbol,
                        double price, uint32_t size,
                        const std::chrono::system_clock::time_point& timestamp,
                        const std::string& exec_type = "F") {
        std::stringstream msg;
        msg << "8=FIX.4.2\x01"
            << "35=8\x01"  // Execution Report
            << "49=GENERATOR\x01"
            << "56=DARKPOOL\x01"
            << "34=" << fix_sequence_++ << "\x01"
            << "52=" << format_timestamp(timestamp) << "\x01"
            << "17=EXEC" << fix_sequence_ << "\x01"
            << "11=ORD" << fix_sequence_ << "\x01"
            << "55=" << symbol << "\x01"
            << "54=1\x01"  // Buy
            << "31=" << std::fixed << std::setprecision(2) << price << "\x01"
            << "32=" << size << "\x01"
            << "14=" << size << "\x01"  // Cum qty
            << "151=0\x01"  // Leaves qty
            << "39=2\x01"   // Filled
            << "150=" << exec_type << "\x01";
        
        add_fix_checksum(msg);
        file << msg.str() << "\n";
    }
    
    void write_fix_new_order(std::ofstream& file,
                            const std::string& symbol,
                            bool is_buy, double price, uint32_t size,
                            const std::string& order_id,
                            const std::chrono::system_clock::time_point& timestamp) {
        std::stringstream msg;
        msg << "8=FIX.4.2\x01"
            << "35=D\x01"  // New Order Single
            << "49=GENERATOR\x01"
            << "56=DARKPOOL\x01"
            << "34=" << fix_sequence_++ << "\x01"
            << "52=" << format_timestamp(timestamp) << "\x01"
            << "11=" << order_id << "\x01"
            << "55=" << symbol << "\x01"
            << "54=" << (is_buy ? "1" : "2") << "\x01"
            << "38=" << size << "\x01"
            << "44=" << std::fixed << std::setprecision(2) << price << "\x01"
            << "40=2\x01"  // Limit order
            << "59=0\x01"; // Day order
        
        add_fix_checksum(msg);
        file << msg.str() << "\n";
    }
    
    // ITCH message writers
    void write_itch_add_order(std::ofstream& file,
                             const std::string& symbol,
                             bool is_buy, double price, uint32_t size,
                             uint64_t order_ref,
                             const std::chrono::system_clock::time_point& timestamp) {
        std::vector<uint8_t> msg(36);
        msg[0] = 0;
        msg[1] = 35;  // Message length
        msg[2] = 'A'; // Add Order
        
        // Stock locate
        uint16_t stock_locate = std::hash<std::string>{}(symbol) % 1000;
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(stock_locate);
        
        // Tracking number
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(itch_tracking_++);
        
        // Timestamp (nanoseconds since midnight)
        uint64_t nanos = get_nanos_since_midnight(timestamp);
        write_itch_timestamp(&msg[7], nanos);
        
        // Order reference
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(order_ref);
        
        // Buy/sell
        msg[21] = is_buy ? 'B' : 'S';
        
        // Shares
        *reinterpret_cast<uint32_t*>(&msg[22]) = htonl(size);
        
        // Stock symbol (8 bytes, padded)
        std::string padded = symbol;
        padded.resize(8, ' ');
        std::memcpy(&msg[26], padded.c_str(), 8);
        
        // Price (in hundredths)
        *reinterpret_cast<uint32_t*>(&msg[34]) = htonl(static_cast<uint32_t>(price * 100));
        
        file.write(reinterpret_cast<char*>(msg.data()), msg.size());
    }
    
    void write_itch_trade(std::ofstream& file,
                         const std::string& symbol,
                         double price, uint32_t size,
                         const std::chrono::system_clock::time_point& timestamp) {
        // Use Order Executed message
        std::vector<uint8_t> msg(31);
        msg[0] = 0;
        msg[1] = 30;  // Message length
        msg[2] = 'E'; // Order Executed
        
        uint16_t stock_locate = std::hash<std::string>{}(symbol) % 1000;
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(stock_locate);
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(itch_tracking_++);
        
        uint64_t nanos = get_nanos_since_midnight(timestamp);
        write_itch_timestamp(&msg[7], nanos);
        
        // Order reference (dummy)
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(itch_order_ref_++);
        
        // Executed shares
        *reinterpret_cast<uint32_t*>(&msg[21]) = htonl(size);
        
        // Match number
        *reinterpret_cast<uint64_t*>(&msg[25]) = htobe64(itch_match_++);
        
        file.write(reinterpret_cast<char*>(msg.data()), msg.size());
    }
    
    void write_itch_order_cancel(std::ofstream& file,
                                const std::string& symbol,
                                uint64_t order_ref,
                                uint32_t size,
                                const std::chrono::system_clock::time_point& timestamp) {
        std::vector<uint8_t> msg(23);
        msg[0] = 0;
        msg[1] = 22;  // Message length
        msg[2] = 'X'; // Order Cancel
        
        uint16_t stock_locate = std::hash<std::string>{}(symbol) % 1000;
        *reinterpret_cast<uint16_t*>(&msg[3]) = htons(stock_locate);
        *reinterpret_cast<uint16_t*>(&msg[5]) = htons(itch_tracking_++);
        
        uint64_t nanos = get_nanos_since_midnight(timestamp);
        write_itch_timestamp(&msg[7], nanos);
        
        *reinterpret_cast<uint64_t*>(&msg[13]) = htobe64(order_ref);
        *reinterpret_cast<uint32_t*>(&msg[21]) = htonl(size);
        
        file.write(reinterpret_cast<char*>(msg.data()), msg.size());
    }
    
    // OUCH message writer
    void write_ouch_order(std::ofstream& file,
                         const std::string& symbol,
                         bool is_buy, double price, uint32_t size,
                         const std::chrono::system_clock::time_point& timestamp) {
        // Simplified OUCH enter order message
        std::vector<uint8_t> msg(49);
        msg[0] = 0;
        msg[1] = 48;  // Message length
        msg[2] = 'O'; // Enter Order
        
        // Order token (14 bytes)
        std::string token = "ORD" + std::to_string(ouch_token_++);
        token.resize(14, ' ');
        std::memcpy(&msg[3], token.c_str(), 14);
        
        // Buy/sell
        msg[17] = is_buy ? 'B' : 'S';
        
        // Shares
        *reinterpret_cast<uint32_t*>(&msg[18]) = htonl(size);
        
        // Stock (8 bytes)
        std::string padded = symbol;
        padded.resize(8, ' ');
        std::memcpy(&msg[22], padded.c_str(), 8);
        
        // Price
        *reinterpret_cast<uint32_t*>(&msg[30]) = htonl(static_cast<uint32_t>(price * 10000));
        
        // Time in force
        *reinterpret_cast<uint32_t*>(&msg[34]) = htonl(99999); // Day
        
        // Firm (4 bytes)
        std::memcpy(&msg[38], "TEST", 4);
        
        // Display
        msg[42] = 'Y';
        
        // Capacity
        msg[43] = 'A';
        
        // Intermarket sweep
        msg[44] = 'N';
        
        // Min quantity
        *reinterpret_cast<uint32_t*>(&msg[45]) = htonl(1);
        
        file.write(reinterpret_cast<char*>(msg.data()), msg.size());
    }
    
    // Helper functions
    std::string format_timestamp(const std::chrono::system_clock::time_point& tp) {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y%m%d-%H:%M:%S");
        
        // Add milliseconds
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            tp.time_since_epoch()) % 1000;
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        
        return ss.str();
    }
    
    void add_fix_checksum(std::stringstream& msg) {
        std::string body = msg.str();
        
        // Add body length
        size_t start = body.find("35=");
        size_t length = body.length() - start;
        
        // Insert body length after version
        size_t pos = body.find("\x01", 8);
        body.insert(pos + 1, "9=" + std::to_string(length) + "\x01");
        
        // Calculate checksum
        uint8_t checksum = 0;
        for (char c : body) {
            checksum += static_cast<uint8_t>(c);
        }
        
        msg.str(body);
        msg << "10=" << std::setfill('0') << std::setw(3) 
            << (checksum % 256) << "\x01";
    }
    
    uint64_t get_nanos_since_midnight(const std::chrono::system_clock::time_point& tp) {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        std::tm* tm = std::gmtime(&time_t);
        
        uint64_t seconds_since_midnight = tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec;
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
            tp.time_since_epoch()) % 1000000000;
        
        return seconds_since_midnight * 1000000000ULL + nanos.count();
    }
    
    void write_itch_timestamp(uint8_t* buf, uint64_t nanos) {
        buf[0] = (nanos >> 40) & 0xFF;
        buf[1] = (nanos >> 32) & 0xFF;
        buf[2] = (nanos >> 24) & 0xFF;
        buf[3] = (nanos >> 16) & 0xFF;
        buf[4] = (nanos >> 8) & 0xFF;
        buf[5] = nanos & 0xFF;
    }
    
    void log_anomaly(std::ofstream& file,
                    const std::chrono::system_clock::time_point& timestamp,
                    const std::string& symbol,
                    const std::string& type,
                    const std::string& description,
                    const std::string& severity) {
        file << format_timestamp(timestamp) << ","
             << symbol << ","
             << type << ","
             << description << ","
             << severity << "\n";
    }
    
    Config config_;
    std::map<std::string, SymbolData> symbol_data_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> uniform_dist_{0.0, 1.0};
    
    // Sequence numbers
    uint64_t fix_sequence_ = 1;
    uint16_t itch_tracking_ = 1;
    uint64_t itch_order_ref_ = 1000000;
    uint64_t itch_match_ = 2000000;
    uint64_t ouch_token_ = 3000000;
};

int main(int argc, char* argv[]) {
    MarketDataGenerator::Config config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--symbols" && i + 1 < argc) {
            config.symbols.clear();
            std::string symbols = argv[++i];
            size_t pos = 0;
            while ((pos = symbols.find(',')) != std::string::npos) {
                config.symbols.push_back(symbols.substr(0, pos));
                symbols.erase(0, pos + 1);
            }
            config.symbols.push_back(symbols);
        } else if (arg == "--duration" && i + 1 < argc) {
            config.duration_seconds = std::stoi(argv[++i]);
        } else if (arg == "--rate" && i + 1 < argc) {
            config.messages_per_second = std::stoi(argv[++i]);
        } else if (arg == "--anomaly-rate" && i + 1 < argc) {
            config.anomaly_rate = std::stod(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
            if (config.output_dir.back() != '/') {
                config.output_dir += '/';
            }
        } else if (arg == "--condition" && i + 1 < argc) {
            std::string cond = argv[++i];
            if (cond == "normal") config.market_condition = MarketDataGenerator::Config::NORMAL;
            else if (cond == "trending") config.market_condition = MarketDataGenerator::Config::TRENDING;
            else if (cond == "volatile") config.market_condition = MarketDataGenerator::Config::VOLATILE;
            else if (cond == "calm") config.market_condition = MarketDataGenerator::Config::CALM;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --symbols SYMBOL1,SYMBOL2,...  Symbols to generate (default: AAPL,MSFT,GOOGL,AMZN,TSLA)\n"
                      << "  --duration SECONDS            Duration in seconds (default: 3600)\n"
                      << "  --rate MESSAGES_PER_SEC       Message rate (default: 10000)\n"
                      << "  --anomaly-rate RATE           Anomaly rate 0.0-1.0 (default: 0.02)\n"
                      << "  --output DIR                  Output directory (default: generated_data/)\n"
                      << "  --condition TYPE              Market condition: normal|trending|volatile|calm\n"
                      << "  --help                        Show this help\n";
            return 0;
        }
    }
    
    MarketDataGenerator generator(config);
    generator.generate();
    
    return 0;
}
