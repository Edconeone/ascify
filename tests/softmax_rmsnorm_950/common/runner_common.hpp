#ifndef ASCIFY_TESTS_SOFTMAX_RMSNORM_950_RUNNER_COMMON_HPP_
#define ASCIFY_TESTS_SOFTMAX_RMSNORM_950_RUNNER_COMMON_HPP_

#include <acl/acl.h>
#include <simt_api/asc_fp16.h>
#include <simt_api/device_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ascify950 {

inline const char* GetArg(int argc, char** argv, const char* key, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) { return argv[i + 1]; }
  }
  return fallback;
}

inline bool HasFlag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) { return true; }
  }
  return false;
}

inline std::string AclErrorReason(const char* operation, aclError error) {
  std::ostringstream os;
  os << operation << " failed with ACL error " << static_cast<int>(error);
  const char* recent = aclGetRecentErrMsg();
  if (recent != nullptr && recent[0] != '\0') { os << ": " << recent; }
  return os.str();
}

inline void AclCheckOrExit(aclError error, const char* expression, const char* file, int line) {
  if (error == ACL_SUCCESS) { return; }
  std::fprintf(stderr, "%s at %s:%d\n", AclErrorReason(expression, error).c_str(), file, line);
  std::exit(1);
}

#define ASCIFY950_ACL_CHECK(expression) \
  ::ascify950::AclCheckOrExit((expression), #expression, __FILE__, __LINE__)

inline std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (quoted) {
      if (ch == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else if (ch == '"') {
        quoted = false;
      } else {
        field.push_back(ch);
      }
    } else if (ch == '"') {
      quoted = true;
    } else if (ch == ',') {
      fields.push_back(field);
      field.clear();
    } else if (ch != '\r') {
      field.push_back(ch);
    }
  }
  fields.push_back(field);
  return fields;
}

inline std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) { return value; }
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') { escaped.push_back('"'); }
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

inline bool FileHasContent(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return input.good() && input.peek() != std::ifstream::traits_type::eof();
}

inline bool WriteRuntimeGridConfig(const std::string& path, const std::string& run_id,
                                   const std::string& op, const std::string& variant,
                                   const std::string& target_entry,
                                   const std::string& block_threads_policy, int device_id,
                                   std::string* reason) {
  if (path.empty()) { return true; }
  if (run_id.find_first_of("\t\n\r") != std::string::npos
      || op.find_first_of("\t\n\r") != std::string::npos
      || variant.find_first_of("\t\n\r") != std::string::npos
      || target_entry.find_first_of("\t\n\r") != std::string::npos
      || block_threads_policy.find_first_of("\t\n\r") != std::string::npos) {
    if (reason != nullptr) { *reason = "runtime grid config contains a TSV-unsafe value"; }
    return false;
  }
  int64_t vector_core_count = 0;
  const aclError query_error =
      aclrtGetDeviceInfo(static_cast<uint32_t>(device_id), ACL_DEV_ATTR_VECTOR_CORE_NUM,
                         &vector_core_count);
  if (query_error != ACL_SUCCESS || vector_core_count <= 0 || vector_core_count > 1024) {
    if (reason != nullptr) {
      *reason = query_error == ACL_SUCCESS
                    ? "runtime vector-core count is outside the supported grid domain"
                    : AclErrorReason("aclrtGetDeviceInfo(vector core count)", query_error);
    }
    return false;
  }
  static const char* const kHeader =
      "runtime_config_schema_version\trun_id\top\tvariant\tdevice_id\t"
      "target_entry\trequested_grid_cap\tvector_core_count\tresolved_grid_cap\t"
      "grid_policy\tblock_threads_policy";
  const bool has_content = FileHasContent(path);
  if (has_content) {
    std::ifstream input(path);
    std::string actual_header;
    std::getline(input, actual_header);
    if (actual_header != kHeader) {
      if (reason != nullptr) { *reason = "runtime grid config schema mismatch"; }
      return false;
    }
  }
  std::ofstream output(path, std::ios::out | std::ios::app);
  if (!output) {
    if (reason != nullptr) { *reason = "cannot open runtime grid config"; }
    return false;
  }
  if (!has_content) { output << kHeader << '\n'; }
  constexpr int requested_grid_cap = 0;
  const int64_t resolved_grid_cap = vector_core_count * 32;
  output << "1\t" << run_id << '\t' << op << '\t' << variant << '\t' << device_id
         << '\t' << target_entry << '\t' << requested_grid_cap << '\t'
         << vector_core_count << '\t' << resolved_grid_cap
         << "\tauto-aiv-x32\t" << block_threads_policy << '\n';
  output.flush();
  if (!output) {
    if (reason != nullptr) { *reason = "cannot write runtime grid config"; }
    return false;
  }
  return true;
}

inline bool ParseBool(const std::string& value, bool fallback) {
  if (value.empty()) { return fallback; }
  if (value == "1" || value == "true" || value == "yes") { return true; }
  if (value == "0" || value == "false" || value == "no") { return false; }
  return fallback;
}

struct Case {
  std::string case_id;
  std::string op = "softmax";
  std::string dtype = "fp16";
  std::string tier = "full";
  bool run_check = false;
  bool run_bench = true;
  int64_t idx = -1;
  int64_t rows = 0;
  int64_t cols = 0;
  std::string scenario;
  std::string input_pattern = "random";
  std::string cuda_pred_path;
  bool affine = false;
  double eps = 1.0e-5;
  std::string notes;
};

inline int TierRank(const std::string& tier) {
  if (tier == "smoke") { return 0; }
  if (tier == "tune") { return 1; }
  return 2;
}

inline bool TierEnabled(const std::string& case_tier, const std::string& requested_tier) {
  return TierRank(case_tier) <= TierRank(requested_tier);
}

inline std::vector<Case> ReadCases(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    std::fprintf(stderr, "cannot open shape file: %s\n", path.c_str());
    std::exit(1);
  }

  std::string line;
  if (!std::getline(input, line)) {
    std::fprintf(stderr, "empty shape file: %s\n", path.c_str());
    std::exit(1);
  }
  const std::vector<std::string> header = SplitCsvLine(line);
  std::unordered_map<std::string, size_t> column;
  for (size_t i = 0; i < header.size(); ++i) { column[header[i]] = i; }

  auto field = [&](const std::vector<std::string>& values, const char* name,
                   const std::string& fallback = std::string()) -> std::string {
    const auto it = column.find(name);
    if (it == column.end() || it->second >= values.size() || values[it->second].empty()) {
      return fallback;
    }
    return values[it->second];
  };

  std::vector<Case> cases;
  int64_t ordinal = 0;
  while (std::getline(input, line)) {
    if (line.empty()) { continue; }
    const std::vector<std::string> values = SplitCsvLine(line);
    Case item;
    const std::string idx_text = field(values, "idx", std::to_string(ordinal));
    item.idx = std::strtoll(idx_text.c_str(), nullptr, 10);
    item.case_id = field(values, "case_id", idx_text);
    item.op = field(values, "op", "softmax");
    item.dtype = field(values, "dtype", "fp16");
    item.tier = field(values, "tier", "full");
    item.run_check = ParseBool(field(values, "run_check"), false);
    item.run_bench = ParseBool(field(values, "run_bench"), true);
    item.rows = std::strtoll(field(values, "rows", "0").c_str(), nullptr, 10);
    item.cols =
        std::strtoll(field(values, "cols", field(values, "ncol", "0")).c_str(), nullptr, 10);
    item.scenario = field(values, "scenario", "unspecified");
    item.input_pattern = field(values, "input_pattern", "random");
    item.cuda_pred_path =
        field(values, "cuda_pred_path", field(values, "predicted_path", "unknown"));
    item.affine = ParseBool(field(values, "affine"), false);
    item.eps = std::strtod(field(values, "eps", "1e-5").c_str(), nullptr);
    item.notes = field(values, "notes");
    cases.push_back(std::move(item));
    ++ordinal;
  }
  return cases;
}

inline bool CheckedTensorSize(int64_t rows, int64_t cols, size_t element_size, size_t* elements,
                              size_t* bytes, std::string* reason) {
  if (rows <= 0 || cols <= 0) {
    if (reason != nullptr) { *reason = "rows and cols must both be positive"; }
    return false;
  }
  const uint64_t urows = static_cast<uint64_t>(rows);
  const uint64_t ucols = static_cast<uint64_t>(cols);
  if (urows > std::numeric_limits<size_t>::max() / ucols) {
    if (reason != nullptr) { *reason = "rows*cols overflows size_t"; }
    return false;
  }
  const size_t count = static_cast<size_t>(urows * ucols);
  if (count > std::numeric_limits<size_t>::max() / element_size) {
    if (reason != nullptr) { *reason = "tensor byte count overflows size_t"; }
    return false;
  }
  if (elements != nullptr) { *elements = count; }
  if (bytes != nullptr) { *bytes = count * element_size; }
  return true;
}

inline uint64_t SplitMix64Host(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

enum InputPatternCode : int {
  kRandom = 0,
  kConstant = 1,
  kRamp = 2,
  kDominant = 3,
  kWide = 4,
  kFp16Extrema = 5,
};

inline InputPatternCode PatternCode(const std::string& pattern) {
  if (pattern == "constant") { return kConstant; }
  if (pattern == "ramp") { return kRamp; }
  if (pattern == "dominant") { return kDominant; }
  if (pattern == "wide") { return kWide; }
  if (pattern == "fp16_extrema") { return kFp16Extrema; }
  return kRandom;
}

inline float HostInputValue(InputPatternCode pattern, uint64_t linear, int64_t row, int64_t col,
                            int64_t cols, uint64_t seed) {
  if (pattern == kConstant) { return 0.25f; }
  if (pattern == kRamp) {
    return static_cast<float>(static_cast<int>(col % 257) - 128) * (1.0f / 32.0f);
  }
  if (pattern == kDominant) { return col == (row % cols) ? 8.0f : -8.0f; }
  if (pattern == kFp16Extrema) {
    return (col & 1) == 0 ? 65504.0f : -65504.0f;
  }
  const uint64_t hash = SplitMix64Host(linear ^ seed);
  const float unit = static_cast<float>(hash & 0x00ffffffULL) * (1.0f / 16777215.0f);
  return pattern == kWide ? -15.0f + 30.0f * unit : -3.0f + 6.0f * unit;
}

__aicore__ inline uint64_t SplitMix64Device(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void FillHalfKernel(half* output, uint64_t elements, int64_t cols, int pattern,
                               uint64_t seed) {
  const uint64_t start =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + static_cast<uint64_t>(threadIdx.x);
  const uint64_t step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t linear = start; linear < elements; linear += step) {
    const int64_t row = static_cast<int64_t>(linear / static_cast<uint64_t>(cols));
    const int64_t col = static_cast<int64_t>(linear - static_cast<uint64_t>(row) * cols);
    float value;
    if (pattern == kConstant) {
      value = 0.25f;
    } else if (pattern == kRamp) {
      value = static_cast<float>(static_cast<int>(col % 257) - 128) * (1.0f / 32.0f);
    } else if (pattern == kDominant) {
      value = col == (row % cols) ? 8.0f : -8.0f;
    } else if (pattern == kFp16Extrema) {
      value = (col & 1) == 0 ? 65504.0f : -65504.0f;
    } else {
      const uint64_t hash = SplitMix64Device(linear ^ seed);
      const float unit = static_cast<float>(hash & 0x00ffffffULL) * (1.0f / 16777215.0f);
      value = pattern == kWide ? -15.0f + 30.0f * unit : -3.0f + 6.0f * unit;
    }
    output[linear] = static_cast<half>(value);
  }
}

inline aclError FillDeviceHalf(half* output, size_t elements, int64_t cols, InputPatternCode pattern,
                               uint64_t seed, aclrtStream stream) {
  constexpr int block = 256;
  size_t grid = (elements + block - 1) / block;
  grid = std::max<size_t>(1, std::min<size_t>(grid, 8192));
  FillHalfKernel<<<static_cast<int>(grid), block, 0, stream>>>(
      output, static_cast<uint64_t>(elements), cols, static_cast<int>(pattern), seed);
  return aclrtPeekAtLastError(static_cast<aclrtLastErrLevel>(0));
}

inline float HalfBitsToFloat(uint16_t half_bits) {
  const uint32_t sign = static_cast<uint32_t>(half_bits & 0x8000U) << 16;
  uint32_t exponent = (half_bits >> 10) & 0x1fU;
  uint32_t mantissa = half_bits & 0x03ffU;
  uint32_t float_bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      float_bits = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffU;
      float_bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fU) {
    float_bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    float_bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &float_bits, sizeof(value));
  return value;
}

inline uint16_t FloatToHalfBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  const uint32_t absolute = bits & 0x7fffffffU;
  if (absolute >= 0x7f800000U) {
    if ((absolute & 0x007fffffU) == 0) { return static_cast<uint16_t>(sign | 0x7c00U); }
    return static_cast<uint16_t>(sign | 0x7e00U);
  }

  int32_t exponent = static_cast<int32_t>((absolute >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = absolute & 0x007fffffU;
  if (exponent >= 31) { return static_cast<uint16_t>(sign | 0x7c00U); }
  if (exponent <= 0) {
    if (exponent < -10) { return static_cast<uint16_t>(sign); }
    mantissa |= 0x00800000U;
    const int shift = 14 - exponent;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U))) { ++rounded; }
    return static_cast<uint16_t>(sign | rounded);
  }

  uint32_t rounded = mantissa + 0x00000fffU + ((mantissa >> 13) & 1U);
  if ((rounded & 0x00800000U) != 0) {
    rounded = 0;
    ++exponent;
    if (exponent >= 31) { return static_cast<uint16_t>(sign | 0x7c00U); }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10)
                               | (rounded >> 13));
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), bytes_(other.bytes_) {
    other.pointer_ = nullptr;
    other.bytes_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      pointer_ = other.pointer_;
      bytes_ = other.bytes_;
      other.pointer_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }

  aclError Allocate(size_t bytes) {
    Reset();
    bytes_ = bytes;
    const aclError error = aclrtMalloc(&pointer_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (error != ACL_SUCCESS) {
      pointer_ = nullptr;
      bytes_ = 0;
    }
    return error;
  }

  void Reset() {
    if (pointer_ != nullptr) {
      aclrtFree(pointer_);
      pointer_ = nullptr;
      bytes_ = 0;
    }
  }

  template<typename T>
  T* As() {
    return static_cast<T*>(pointer_);
  }

  void* Get() { return pointer_; }
  size_t Bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  size_t bytes_ = 0;
};

class AclSession {
 public:
  AclSession() = default;
  ~AclSession() { Reset(); }
  AclSession(const AclSession&) = delete;
  AclSession& operator=(const AclSession&) = delete;

  aclError Init(int device_id) {
    device_id_ = device_id;
    aclError error = aclInit(nullptr);
    if (error != ACL_SUCCESS) { return error; }
    initialized_ = true;
    error = aclrtSetDevice(device_id_);
    if (error != ACL_SUCCESS) { return error; }
    device_set_ = true;
    error = aclrtCreateContext(&context_, device_id_);
    if (error != ACL_SUCCESS) { return error; }
    error = aclrtCreateStream(&stream_);
    return error;
  }

  aclrtStream Stream() const { return stream_; }

  void Reset() {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
      stream_ = nullptr;
    }
    if (context_ != nullptr) {
      aclrtDestroyContext(context_);
      context_ = nullptr;
    }
    if (device_set_) {
      aclrtResetDevice(device_id_);
      device_set_ = false;
    }
    if (initialized_) {
      aclFinalize();
      initialized_ = false;
    }
  }

 private:
  int device_id_ = -1;
  bool initialized_ = false;
  bool device_set_ = false;
  aclrtContext context_ = nullptr;
  aclrtStream stream_ = nullptr;
};

struct LatencyStats {
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double min_ms = 0.0;
  double p90_ms = 0.0;
};

struct TimingResult {
  bool ok = false;
  aclError error = ACL_SUCCESS;
  std::string reason;
  LatencyStats stats;
};

template<typename Launch>
TimingResult TimeKernel(Launch&& launch, aclrtStream stream, int warmup, int samples,
                        int inner_repeats) {
  TimingResult result;
  if (warmup < 0 || samples <= 0 || inner_repeats <= 0) {
    result.reason = "invalid timing configuration";
    return result;
  }
  for (int i = 0; i < warmup; ++i) {
    const aclError error = launch();
    if (error != ACL_SUCCESS) {
      result.error = error;
      result.reason = AclErrorReason("warmup launch", error);
      return result;
    }
  }
  aclError error = aclrtSynchronizeStream(stream);
  if (error != ACL_SUCCESS) {
    result.error = error;
    result.reason = AclErrorReason("warmup synchronize", error);
    return result;
  }

  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  error = aclrtCreateEventExWithFlag(&start, ACL_EVENT_TIME_LINE);
  if (error != ACL_SUCCESS) {
    result.error = error;
    result.reason = AclErrorReason("aclrtCreateEventExWithFlag(start)", error);
    return result;
  }
  error = aclrtCreateEventExWithFlag(&end, ACL_EVENT_TIME_LINE);
  if (error != ACL_SUCCESS) {
    aclrtDestroyEvent(start);
    result.error = error;
    result.reason = AclErrorReason("aclrtCreateEventExWithFlag(end)", error);
    return result;
  }

  std::vector<float> measurements;
  measurements.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    error = aclrtRecordEvent(start, stream);
    if (error != ACL_SUCCESS) { break; }
    for (int repeat = 0; repeat < inner_repeats; ++repeat) {
      error = launch();
      if (error != ACL_SUCCESS) { break; }
    }
    if (error != ACL_SUCCESS) { break; }
    error = aclrtRecordEvent(end, stream);
    if (error != ACL_SUCCESS) { break; }
    error = aclrtSynchronizeEvent(end);
    if (error != ACL_SUCCESS) { break; }
    float elapsed_ms = 0.0f;
    error = aclrtEventElapsedTime(&elapsed_ms, start, end);
    if (error != ACL_SUCCESS) { break; }
    measurements.push_back(elapsed_ms / static_cast<float>(inner_repeats));
  }
  aclrtDestroyEvent(start);
  aclrtDestroyEvent(end);

  if (error != ACL_SUCCESS) {
    result.error = error;
    result.reason = AclErrorReason("timed launch", error);
    return result;
  }
  if (measurements.size() != static_cast<size_t>(samples)) {
    result.reason = "timing produced an incomplete sample set";
    return result;
  }

  std::sort(measurements.begin(), measurements.end());
  double sum = 0.0;
  for (float value : measurements) { sum += value; }
  result.stats.mean_ms = sum / measurements.size();
  result.stats.median_ms = measurements[measurements.size() / 2];
  result.stats.min_ms = measurements.front();
  const size_t p90_index =
      std::min(measurements.size() - 1,
               static_cast<size_t>(std::ceil(0.9 * measurements.size())) - 1);
  result.stats.p90_ms = measurements[p90_index];
  result.ok = true;
  return result;
}

struct AccuracyRecord {
  std::string run_id;
  std::string op;
  std::string dtype;
  std::string math_mode;
  std::string variant;
  Case test_case;
  int device_id = -1;
  std::string status;
  std::string reason;
  double max_abs_error = 0.0;
  double max_scaled_rel_error = 0.0;
  double max_row_sum_error = 0.0;
  double max_aux_abs_error = 0.0;
  double max_aux_scaled_rel_error = 0.0;
  uint64_t nonfinite_count = 0;
  uint64_t guard_mismatch_count = 0;
  uint64_t canary_mismatch_count = 0;
};

class AccuracyCsvWriter {
 public:
  explicit AccuracyCsvWriter(const std::string& path)
      : output_(path, std::ios::out | std::ios::app) {
    static const char* const kHeader =
        "run_id,op,dtype,math_mode,variant,case_id,idx,tier,rows,cols,scenario,"
        "input_pattern,cuda_pred_path,device_id,status,reason,max_abs_error,"
        "max_scaled_rel_error,max_row_sum_error,max_aux_abs_error,"
        "max_aux_scaled_rel_error,nonfinite_count,guard_mismatch_count,"
        "canary_mismatch_count";
    if (!output_) {
      std::fprintf(stderr, "cannot open accuracy CSV: %s\n", path.c_str());
      std::exit(1);
    }
    if (FileHasContent(path)) {
      std::ifstream input(path);
      std::string actual_header;
      std::getline(input, actual_header);
      if (actual_header != kHeader) {
        std::fprintf(stderr,
                     "accuracy CSV schema mismatch: %s; choose a new output file\n",
                     path.c_str());
        std::exit(1);
      }
    } else {
      output_ << kHeader << '\n';
    }
  }

  void Write(const AccuracyRecord& record) {
    output_ << CsvEscape(record.run_id) << ',' << CsvEscape(record.op) << ','
            << CsvEscape(record.dtype) << ',' << CsvEscape(record.math_mode) << ','
            << CsvEscape(record.variant) << ',' << CsvEscape(record.test_case.case_id) << ','
            << record.test_case.idx << ',' << CsvEscape(record.test_case.tier) << ','
            << record.test_case.rows << ',' << record.test_case.cols << ','
            << CsvEscape(record.test_case.scenario) << ','
            << CsvEscape(record.test_case.input_pattern) << ','
            << CsvEscape(record.test_case.cuda_pred_path) << ',' << record.device_id << ','
            << CsvEscape(record.status) << ',' << CsvEscape(record.reason) << ','
            << std::setprecision(17) << record.max_abs_error << ','
            << record.max_scaled_rel_error << ',' << record.max_row_sum_error << ','
            << record.max_aux_abs_error << ',' << record.max_aux_scaled_rel_error << ','
            << record.nonfinite_count << ',' << record.guard_mismatch_count << ','
            << record.canary_mismatch_count << '\n';
    output_.flush();
  }

 private:
  std::ofstream output_;
};

struct PerfRecord {
  std::string run_id;
  std::string op;
  std::string dtype;
  std::string math_mode;
  std::string variant;
  Case test_case;
  int device_id = -1;
  int warmup = 0;
  int samples = 0;
  int inner_repeats = 1;
  LatencyStats latency;
  uint64_t logical_bytes = 0;
  double gbps = 0.0;
  std::string status;
  std::string reason;
};

class PerfCsvWriter {
 public:
  explicit PerfCsvWriter(const std::string& path)
      : output_(path, std::ios::out | std::ios::app) {
    static const char* const kHeader =
        "run_id,op,dtype,math_mode,variant,case_id,idx,tier,rows,cols,scenario,"
        "input_pattern,cuda_pred_path,device_id,warmup,samples,inner_repeats,"
        "lat_ms_mean,lat_ms_median,lat_ms_min,lat_ms_p90,logical_bytes,gbps,"
        "status,reason";
    if (!output_) {
      std::fprintf(stderr, "cannot open performance CSV: %s\n", path.c_str());
      std::exit(1);
    }
    if (FileHasContent(path)) {
      std::ifstream input(path);
      std::string actual_header;
      std::getline(input, actual_header);
      if (actual_header != kHeader) {
        std::fprintf(stderr,
                     "performance CSV schema mismatch: %s; choose a new output file\n",
                     path.c_str());
        std::exit(1);
      }
    } else {
      output_ << kHeader << '\n';
    }
  }

  void Write(const PerfRecord& record) {
    output_ << CsvEscape(record.run_id) << ',' << CsvEscape(record.op) << ','
            << CsvEscape(record.dtype) << ',' << CsvEscape(record.math_mode) << ','
            << CsvEscape(record.variant) << ',' << CsvEscape(record.test_case.case_id) << ','
            << record.test_case.idx << ',' << CsvEscape(record.test_case.tier) << ','
            << record.test_case.rows << ',' << record.test_case.cols << ','
            << CsvEscape(record.test_case.scenario) << ','
            << CsvEscape(record.test_case.input_pattern) << ','
            << CsvEscape(record.test_case.cuda_pred_path) << ',' << record.device_id << ','
            << record.warmup << ',' << record.samples << ',' << record.inner_repeats << ','
            << std::setprecision(17) << record.latency.mean_ms << ','
            << record.latency.median_ms << ',' << record.latency.min_ms << ','
            << record.latency.p90_ms << ',' << record.logical_bytes << ',' << record.gbps << ','
            << CsvEscape(record.status) << ',' << CsvEscape(record.reason) << '\n';
    output_.flush();
  }

 private:
  std::ofstream output_;
};

}  // namespace ascify950

#endif  // ASCIFY_TESTS_SOFTMAX_RMSNORM_950_RUNNER_COMMON_HPP_
