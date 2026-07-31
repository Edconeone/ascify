#include <stdint.h>
#include <type_traits>

#define __aicore__
#define __ubuf__

struct AscifyTestThreadIndex {
  unsigned int x;
};
static AscifyTestThreadIndex threadIdx{};

inline void asc_syncthreads() {}

#include <acl_cub/aclcub.hpp>

template<typename T>
struct TaggedSum {
  using ascify_reduction_tag = aclcub::Sum;
  using ascify_reduction_value_type = T;
  using ascify_reduction_owner_type = TaggedSum;
  T operator()(T lhs, T rhs) const { return lhs + rhs; }
};

template<typename T>
struct UnknownTag {
  using ascify_reduction_tag = T;
  using ascify_reduction_value_type = T;
  using ascify_reduction_owner_type = UnknownTag;
  T operator()(T lhs, T) const { return lhs; }
};

struct BoolAdd {
  using ascify_reduction_tag = aclcub::Sum;
  using ascify_reduction_value_type = bool;
  using ascify_reduction_owner_type = BoolAdd;
  bool operator()(bool lhs, bool rhs) const { return lhs + rhs; }
};

struct MultiplyFromTaggedSum : TaggedSum<float> {
  float operator()(float lhs, float rhs) const { return lhs * rhs; }
};

static_assert(
    std::is_same<aclcub::DispatchTagOf<float, TaggedSum<float>>::type,
                 aclcub::Sum>::value,
    "supported value types must use the semantic hardware tag");
static_assert(
    std::is_same<aclcub::DispatchTagOf<double, TaggedSum<double>>::type,
                 void>::value,
    "unsupported value types must retain the generic reducer");
static_assert(
    std::is_same<aclcub::DispatchTagOf<float, UnknownTag<float>>::type,
                 void>::value,
    "unknown user aliases must not select a hardware primitive");
static_assert(
    std::is_same<aclcub::DispatchTagOf<float, BoolAdd>::type,
                 void>::value,
    "a marker for a different reducer value type must use the generic path");
static_assert(
    std::is_same<aclcub::DispatchTagOf<float, TaggedSum<double>>::type,
                 void>::value,
    "template reducer value type must match BlockReduce value type");
static_assert(
    std::is_same<aclcub::DispatchTagOf<float, MultiplyFromTaggedSum>::type,
                 void>::value,
    "an inherited marker must not tag a derived reducer override");
static_assert(
    std::is_same<aclcub::OpTagOf<MultiplyFromTaggedSum>::type,
                 void>::value,
    "the public classification trait must reject an inherited marker");

int main() {
  aclcub::BlockReduce<double, 32>::TempStorage storage{};
  aclcub::BlockReduce<double, 32> reduction(storage);
  (void)reduction.Sum(1.0);
  return 0;
}
