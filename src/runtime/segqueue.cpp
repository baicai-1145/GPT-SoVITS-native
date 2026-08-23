// segqueue.cpp — non-template helpers for the pipeline skeleton.
#include "segqueue.hpp"

namespace gsv {
namespace runtime {
namespace detail {

// QoS label -> calling thread (§4 cluster mapping). Kept out-of-line so the
// header stays template-only and the mapping has a single definition point.
bool applyQos(const std::string& name) {
    qos_class_t q;
    if (name.empty() || name == "default") return true;
    if (name == "user_initiated") q = QOS_CLASS_USER_INITIATED;
    else if (name == "user_interactive") q = QOS_CLASS_USER_INTERACTIVE;
    else if (name == "utility") q = QOS_CLASS_UTILITY;
    else if (name == "background") q = QOS_CLASS_BACKGROUND;
    else return false;
    return pthread_set_qos_class_self_np(q, 0) == 0;
}

}  // namespace detail
}  // namespace detail
}  // namespace runtime
