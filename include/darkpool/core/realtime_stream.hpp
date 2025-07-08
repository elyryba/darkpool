#pragma once

#include <memory>
#include <atomic>
#include <chrono>
#include "darkpool/types.hpp"
#include "darkpool/config.hpp"

namespace darkpool {

// Forward declarations
class Detector;
class ProtocolNormalizer;
class RealTimeStream;

// Factory function
std::unique_ptr<RealTimeStream> create_realtime_stream(
    const Config& config,
    const RealTimeStream::Config& stream_config = {}
);

} 
