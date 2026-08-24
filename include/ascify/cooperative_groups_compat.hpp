#ifndef ASCIFY_COOPERATIVE_GROUPS_COMPAT_HPP
#define ASCIFY_COOPERATIVE_GROUPS_COMPAT_HPP

#include <type_traits>

#if !defined(__has_include)
#error "Ascify cooperative groups compatibility requires __has_include support"
#elif __has_include(<simt_api/cooperative_groups.h>) &&                     \
    __has_include(<simt_api/device_functions.h>) &&                        \
    __has_include(<simt_api/device_sync_functions.h>) &&                   \
    __has_include(<simt_api/device_warp_functions.h>)
#define ASCIFY_HAS_LEGACY_COOPERATIVE_GROUPS 1
#elif __has_include(<simt_api/kernel_simt_intf.h>)
#error "Ascify cooperative groups compatibility is unsupported on public CANN 8.5: no native cooperative_groups header"
#else
#error "Ascify cooperative groups compatibility requires the verified legacy CANN cooperative_groups headers"
#endif

#if defined(ASCIFY_HAS_LEGACY_COOPERATIVE_GROUPS)
#include <simt_api/device_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>
#include <simt_api/cooperative_groups.h>

namespace cooperative_groups {

// CUDA exposes free sync(group). The verified legacy CANN header only exposes
// thread_block::sync(), whose implementation reaches the work-item barrier.
// Keep this overload concrete: accepting arbitrary groups would incorrectly
// admit tile/coalesced synchronization backed only by a block memory fence.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void sync(const thread_block& group) {
  group.sync();
}

}  // namespace cooperative_groups
#endif

#endif  // ASCIFY_COOPERATIVE_GROUPS_COMPAT_HPP
