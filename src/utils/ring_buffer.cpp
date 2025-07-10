#include "darkpool/utils/ring_buffer.hpp"

namespace darkpool::utils {

// Explicit instantiations for common types
template class RingBuffer<MarketMessage>;
template class RingBuffer<Anomaly>;
template class RingBuffer<Order>;
template class RingBuffer<Trade>;
template class RingBuffer<Quote>;

// Specializations for bulk operations can be added here if needed
// The main implementation is in the header as it's templated

} 
