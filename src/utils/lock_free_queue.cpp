#include "darkpool/utils/lock_free_queue.hpp"

namespace darkpool::utils {

// Explicit instantiations for common types
template class LockFreeQueue<MarketMessage>;
template class LockFreeQueue<Anomaly>;
template class LockFreeQueue<Order>;
template class LockFreeQueue<Trade>;
template class LockFreeQueue<Quote>;

template class SPSCQueue<MarketMessage>;
template class SPSCQueue<Anomaly>;
template class SPSCQueue<Order>;
template class SPSCQueue<Trade>;
template class SPSCQueue<Quote>;

} 
