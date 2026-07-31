#include <cuda.h>
#include <stdint.h>
#include <type_traits>

template<typename T, int N>
using FixturePackType =
    typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;

template<typename T, int N>
union FixturePack {
  static_assert(
      sizeof(FixturePackType<T, N>) == sizeof(T) * N, "");
  FixturePackType<T, N> storage;
  T elem[N];
};

template<typename T, int N>
struct NonOverlayPack {
  static_assert(
      sizeof(FixturePackType<T, N>) == sizeof(T) * N, "");
  FixturePackType<T, N> storage;
  T elem[N];
};

// Positive adapter fixtures deliberately use non-conventional class, method,
// field and parameter names.  Matching their spelling would fail this test.
template<typename Wire, typename Accumulator>
struct PackedInput {
  PackedInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}

  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
#pragma unroll
    for (int item = 0; item < Width; ++item) {
      output[item] = static_cast<Accumulator>(packet.elem[item]);
    }
  }

  const Wire* base_;
  int64_t pitch_;
};

template<typename Accumulator, typename Wire>
struct PackedOutput {
  PackedOutput(Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}

  template<int Width>
  __device__ void transfer(const Accumulator* input, int64_t outer,
                           int64_t inner) {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
#pragma unroll
    for (int item = 0; item < Width; ++item) {
      packet.elem[item] = static_cast<Wire>(input[item]);
    }
    *(reinterpret_cast<FixturePackType<Wire, Width>*>(base_)
      + packed_index) = packet.storage;
  }

  Wire* base_;
  int64_t pitch_;
};

template<typename Accumulator, typename Wire, bool affine>
struct AffinePackedOutput {
  AffinePackedOutput(Wire* base, const Wire* weight, int32_t pitch)
      : base_(base), weight_(weight), pitch_(pitch) {}

  template<int Width>
  __device__ void transfer(const Accumulator* input, int32_t outer,
                           int32_t inner) {
    FixturePack<Wire, Width> packet;
    FixturePack<Wire, Width> weight_packet;
    const int32_t packed_index =
        (outer * pitch_ + inner) / Width;
    const int32_t weight_index = inner / Width;
    if (affine) {
      weight_packet.storage =
          *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
                weight_)
            + weight_index);
    }
#pragma unroll
    for (int item = 0; item < Width; ++item) {
      if (affine) {
        packet.elem[item] =
            static_cast<Wire>(input[item])
            * weight_packet.elem[item];
      } else {
        packet.elem[item] = static_cast<Wire>(input[item]);
      }
    }
    *(reinterpret_cast<FixturePackType<Wire, Width>*>(base_)
      + packed_index) = packet.storage;
  }

  Wire* base_;
  const Wire* weight_;
  int32_t pitch_;
};

// A derived adapter inherits the generated aliases from PackedInput, but its
// hidden transfer has different semantics.  The owner alias must keep it out.
template<typename Wire, typename Accumulator>
struct DerivedShiftedInput : PackedInput<Wire, Accumulator> {
  using PackedInput<Wire, Accumulator>::PackedInput;

  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    output[0] = static_cast<Accumulator>(
        this->base_[outer * this->pitch_ + inner + 1]);
  }
};

// Exact field/constructor/memory shapes are not enough: each pack element
// must be a direct static_cast-copy.
template<typename Wire, typename Accumulator>
struct ScaledElementInput {
  ScaledElementInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    for (int item = 0; item < Width; ++item) {
      output[item] =
          static_cast<Accumulator>(packet.elem[item]) * 2;
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct ClampedElementInput {
  ClampedElementInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    for (int item = 0; item < Width; ++item) {
      Accumulator value =
          static_cast<Accumulator>(packet.elem[item]);
      output[item] = value > 0 ? value : 0;
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct SingleElementInput {
  SingleElementInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    output[0] = static_cast<Accumulator>(packet.elem[0]);
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct PacketMutatedInput {
  PacketMutatedInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    packet.elem[0] = Wire{};
    for (int item = 0; item < Width; ++item) {
      output[item] =
          static_cast<Accumulator>(packet.elem[item]);
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct NonOverlayInput {
  NonOverlayInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    NonOverlayPack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    for (int item = 0; item < Width; ++item) {
      output[item] =
          static_cast<Accumulator>(packet.elem[item]);
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Accumulator, typename Wire, bool affine>
struct UnguardedWeightOutput {
  UnguardedWeightOutput(
      Wire* base, const Wire* weight, int32_t pitch)
      : base_(base), weight_(weight), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(const Accumulator* input, int32_t outer,
                           int32_t inner) {
    FixturePack<Wire, Width> packet;
    FixturePack<Wire, Width> weight_packet;
    const int32_t packed_index =
        (outer * pitch_ + inner) / Width;
    const int32_t weight_index = inner / Width;
    weight_packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              weight_)
          + weight_index);
    if (affine) {
      weight_packet.elem[0] = weight_[0];
    }
    for (int item = 0; item < Width; ++item) {
      if (affine) {
        packet.elem[item] =
            static_cast<Wire>(input[item])
            * weight_packet.elem[item];
      } else {
        packet.elem[item] = static_cast<Wire>(input[item]);
      }
    }
    *(reinterpret_cast<FixturePackType<Wire, Width>*>(base_)
      + packed_index) = packet.storage;
  }
  Wire* base_;
  const Wire* weight_;
  int32_t pitch_;
};

template<typename Wire, typename Accumulator>
struct ScaledStrideInput {
  ScaledStrideInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * (pitch_ + 1) + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    for (int item = 0; item < Width; ++item) {
      output[item] =
          static_cast<Accumulator>(packet.elem[item]);
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct PostPackedShiftInput {
  PostPackedShiftInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index + 1);
    for (int item = 0; item < Width; ++item) {
      output[item] =
          static_cast<Accumulator>(packet.elem[item]);
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct MutatedOffsetInput {
  MutatedOffsetInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    ++packed_index;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    for (int item = 0; item < Width; ++item) {
      output[item] =
          static_cast<Accumulator>(packet.elem[item]);
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct ShiftedElementBaseInput {
  ShiftedElementBaseInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    for (int item = 0; item < Width; ++item) {
      (output + 1)[item] =
          static_cast<Accumulator>(packet.elem[item]);
    }
  }
  const Wire* base_;
  int64_t pitch_;
};

template<typename Wire, typename Accumulator>
struct VolatileInput {
  VolatileInput(const volatile Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    for (int item = 0; item < Width; ++item) {
      output[item] = static_cast<Accumulator>(
          base_[outer * pitch_ + inner + item]);
    }
  }
  const volatile Wire* base_;
  int64_t pitch_;
};

template<typename Accumulator, typename Wire>
struct ReservedMarkerOutput {
  ReservedMarkerOutput(Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}
  template<int Width>
  __device__ void transfer(const Accumulator* input, int64_t outer,
                           int64_t inner) {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    for (int item = 0; item < Width; ++item) {
      packet.elem[item] = static_cast<Wire>(input[item]);
    }
    *(reinterpret_cast<FixturePackType<Wire, Width>*>(base_)
      + packed_index) = packet.storage;
  }
  using ascify_target_adapter_owner_type = ReservedMarkerOutput;
  static constexpr bool ascify_target_store_is_affine = false;
  Wire* ascify_target_weight() const { return base_; }
  Wire* base_;
  int64_t pitch_;
};

// Negative adapter fixture: the extra skip/bias state means the target
// facade must not reinterpret this as a direct row-major input.
template<typename Wire, typename Accumulator>
struct SkipLoadLike {
  SkipLoadLike(const Wire* base, const Wire* skip, Wire bias,
               int64_t pitch)
      : base_(base), skip_(skip), bias_(bias), pitch_(pitch) {}

  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    const int64_t packed_index =
        (outer * pitch_ + inner) / Width;
    for (int item = 0; item < Width; ++item) {
      output[item] = static_cast<Accumulator>(
          base_[packed_index * Width + item]
          + skip_[packed_index * Width + item] + bias_);
    }
  }

  const Wire* base_;
  const Wire* skip_;
  Wire bias_;
  int64_t pitch_;
};

// Negative adapter fixture: two fields and matching types are insufficient
// when the packed offset is not exactly (row * stride + col) / pack.
template<typename Wire, typename Accumulator>
struct ShiftedInput {
  ShiftedInput(const Wire* base, int64_t pitch)
      : base_(base), pitch_(pitch) {}

  template<int Width>
  __device__ void transfer(Accumulator* output, int64_t outer,
                           int64_t inner) const {
    FixturePack<Wire, Width> packet;
    const int64_t packed_index =
        (outer * pitch_ + inner + 1) / Width;
    packet.storage =
        *(reinterpret_cast<const FixturePackType<Wire, Width>*>(
              base_)
          + packed_index);
    for (int item = 0; item < Width; ++item) {
      output[item] = static_cast<Accumulator>(packet.elem[item]);
    }
  }

  const Wire* base_;
  int64_t pitch_;
};

template<typename T>
__device__ T SemanticExp(T value);
template<>
__device__ float SemanticExp(float value) {
  return expf(value);
}
template<>
__device__ double SemanticExp(double value) {
  return exp(value);
}

template<typename T>
__device__ T SemanticDivide(T value, T scale);
template<>
__device__ float SemanticDivide(float value, float scale) {
  return value / scale;
}
template<>
__device__ double SemanticDivide(double value, double scale) {
  return value / scale;
}

template<typename T>
__device__ T SemanticRsqrt(T value);
template<>
__device__ float SemanticRsqrt(float value) {
  return rsqrtf(value);
}
template<>
__device__ double SemanticRsqrt(double value) {
  return rsqrt(value);
}

template<typename T>
__device__ T SemanticMax(T lhs, T rhs);
template<>
__device__ float SemanticMax(float lhs, float rhs) {
  return fmaxf(lhs, rhs);
}
template<>
__device__ double SemanticMax(double lhs, double rhs) {
  return fmax(lhs, rhs);
}

template<typename T>
__device__ T ScalarReduce(T value) {
  return value;
}

// Negative semantic fixtures below intentionally use an identity "reduction"
// and fixed (row, column) adapter coordinates.  They exercise spelling-
// independent algebra recognition, but do not prove full row/column coverage
// and therefore must never receive a direct target recipe.
template<typename T>
__device__ T FakeIdentity(T value) {
  return value;
}

template<typename T>
__device__ T FakeLog(T value) {
  return log(value);
}

template<typename T>
__device__ T FakeSqrt(T value) {
  return sqrt(value);
}

enum class RowTransform {
  NormalizedExponent,
  Logarithmic,
  IdentityExponent,
  SquareRootExponent,
  DisconnectedExponent,
};

template<typename Input, typename Output, typename Compute,
         RowTransform transform>
__global__ void RowProbabilityKernel(Input input, Output output,
                                     int64_t rows, int64_t columns) {
  Compute value = 0;
  input.template transfer<1>(&value, 0, 0);
  Compute thread_maximum = value;
  thread_maximum = ::SemanticMax(thread_maximum, value);
  Compute row_maximum = ::ScalarReduce(thread_maximum);
  Compute row_total = 0;
  Compute unrelated_total = 0;
  if (transform == RowTransform::NormalizedExponent) {
    value = ::SemanticExp(value - row_maximum);
    row_total += value;
  } else if (transform == RowTransform::IdentityExponent) {
    value = FakeIdentity(value - row_maximum);
    row_total += value;
  } else if (transform == RowTransform::SquareRootExponent) {
    value = FakeSqrt(value - row_maximum);
    row_total += value;
  } else if (transform == RowTransform::DisconnectedExponent) {
    value = SemanticExp(value - row_maximum);
    unrelated_total += value;
  } else if (transform == RowTransform::Logarithmic) {
    value -= row_maximum;
    row_total += FakeLog(value);
  }
  if (transform == RowTransform::NormalizedExponent) {
    value = ::SemanticDivide(value, row_total);
  } else if (transform == RowTransform::IdentityExponent) {
    value = ::SemanticDivide(value, row_total);
  } else if (transform == RowTransform::SquareRootExponent) {
    value = ::SemanticDivide(value, row_total);
  } else if (transform == RowTransform::DisconnectedExponent) {
    value = ::SemanticDivide(value, row_total);
  } else if (transform == RowTransform::Logarithmic) {
    value -= FakeLog(row_total);
  }
  output.template transfer<1>(&value, 0, 0);
}

template<typename Input, typename Output, typename Compute,
         RowTransform transform>
cudaError_t RouteProbabilityImpl(cudaStream_t stream, Input input,
                                 Output output, int64_t rows,
                                 int64_t columns) {
  RowProbabilityKernel<Input, Output, Compute, transform>
      <<<1, 32, 0, stream>>>(input, output, rows, columns);
  return cudaPeekAtLastError();
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteProbability(cudaStream_t stream, Input input,
                             Output output, int64_t rows,
                             int64_t columns) {
  if (columns < 1024) {
    return RouteProbabilityImpl<
        Input, Output, Compute,
        RowTransform::NormalizedExponent>(
            stream, input, output, rows, columns);
  } else {
    bool cached = false;
    {
      cudaError_t status = RouteProbabilityImpl<
          Input, Output, Compute,
          RowTransform::NormalizedExponent>(
              stream, input, output, rows, columns);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteProbabilityImpl<
          Input, Output, Compute,
          RowTransform::NormalizedExponent>(
              stream, input, output, rows, columns);
    }
    return cudaSuccess;
  }
}

// Negative dispatcher fixture: the same routing topology selects a
// semantically different enum branch.
template<typename Input, typename Output, typename Compute>
cudaError_t RouteLogarithmic(cudaStream_t stream, Input input,
                             Output output, int64_t rows,
                             int64_t columns) {
  if (columns < 1024) {
    return RouteProbabilityImpl<
        Input, Output, Compute, RowTransform::Logarithmic>(
            stream, input, output, rows, columns);
  } else {
    bool cached = false;
    {
      cudaError_t status = RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::Logarithmic>(
              stream, input, output, rows, columns);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::Logarithmic>(
              stream, input, output, rows, columns);
    }
    return cudaSuccess;
  }
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteIdentity(cudaStream_t stream, Input input,
                          Output output, int64_t rows,
                          int64_t columns) {
  if (columns < 1024) {
    return RouteProbabilityImpl<
        Input, Output, Compute, RowTransform::IdentityExponent>(
            stream, input, output, rows, columns);
  } else {
    bool cached = false;
    {
      cudaError_t status = RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::IdentityExponent>(
              stream, input, output, rows, columns);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::IdentityExponent>(
              stream, input, output, rows, columns);
    }
    return cudaSuccess;
  }
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteSquareRoot(cudaStream_t stream, Input input,
                            Output output, int64_t rows,
                            int64_t columns) {
  if (columns < 1024) {
    return RouteProbabilityImpl<
        Input, Output, Compute, RowTransform::SquareRootExponent>(
            stream, input, output, rows, columns);
  } else {
    bool cached = false;
    {
      cudaError_t status = RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::SquareRootExponent>(
              stream, input, output, rows, columns);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::SquareRootExponent>(
              stream, input, output, rows, columns);
    }
    return cudaSuccess;
  }
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteDisconnected(cudaStream_t stream, Input input,
                              Output output, int64_t rows,
                              int64_t columns) {
  if (columns < 1024) {
    return RouteProbabilityImpl<
        Input, Output, Compute, RowTransform::DisconnectedExponent>(
            stream, input, output, rows, columns);
  } else {
    bool cached = false;
    {
      cudaError_t status = RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::DisconnectedExponent>(
              stream, input, output, rows, columns);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteProbabilityImpl<
          Input, Output, Compute, RowTransform::DisconnectedExponent>(
              stream, input, output, rows, columns);
    }
    return cudaSuccess;
  }
}

// Negative dispatcher fixture: models the single-return double fallback
// overload in the real source.  Referencing the positive enum alone must not
// be enough to receive a target prologue.
template<typename Input, typename Output, typename Compute>
cudaError_t RouteDoubleFallback(cudaStream_t stream, Input input,
                                Output output, int64_t rows,
                                int64_t columns) {
  return RouteProbabilityImpl<
      Input, Output, Compute,
      RowTransform::NormalizedExponent>(
          stream, input, output, rows, columns);
}

template<typename Input, typename Output, typename Compute, int Block>
__global__ void RowScaleKernel(Input input, Output output, int rows,
                               int columns, double epsilon,
                               Compute* inverse_scale) {
  // Fixed coordinates and the absence of a cross-thread sum reduction make
  // this an RMSNorm hard negative even though the scalar algebra is complete.
  Compute value = 0;
  input.template transfer<1>(&value, 0, 0);
  Compute square_sum = value * value;
  Compute square_mean =
      square_sum / static_cast<Compute>(columns);
  Compute scale = ::SemanticRsqrt(
      square_mean + static_cast<Compute>(epsilon));
  inverse_scale[0] = scale;
  value *= scale;
  output.template transfer<1>(&value, 0, 0);
}

template<typename Input, typename Output, typename Compute, int Block>
__global__ void RowFakeScaleKernel(Input input, Output output, int rows,
                                   int columns, double epsilon,
                                   Compute* inverse_scale) {
  Compute value = 0;
  input.template transfer<1>(&value, 0, 0);
  Compute square_sum = value * value;
  Compute square_mean =
      square_sum / static_cast<Compute>(columns);
  Compute scale = FakeIdentity(
      square_mean + static_cast<Compute>(epsilon));
  inverse_scale[0] = scale;
  value *= scale;
  output.template transfer<1>(&value, 0, 0);
}

template<typename Input, typename Output, typename Compute, int Block>
__global__ void RowDisconnectedScaleKernel(
    Input input, Output output, int rows, int columns,
    double epsilon, Compute* inverse_scale) {
  Compute value = 0;
  input.template transfer<1>(&value, 0, 0);
  Compute square_sum = value * value;
  Compute unrelated_sum = 1;
  Compute square_mean =
      unrelated_sum / static_cast<Compute>(columns);
  Compute scale = ::SemanticRsqrt(
      square_mean + static_cast<Compute>(epsilon));
  inverse_scale[0] = scale;
  value *= scale;
  output.template transfer<1>(&value, 0, 0);
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteScaleImpl(cudaStream_t stream, Input input,
                           Output output, int64_t rows,
                           int64_t columns, double epsilon,
                           Compute* inverse_scale) {
  RowScaleKernel<Input, Output, Compute, 128>
      <<<1, 128, 0, stream>>>(input, output, static_cast<int>(rows),
                              static_cast<int>(columns), epsilon,
                              inverse_scale);
  return cudaPeekAtLastError();
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteFakeScaleImpl(
    cudaStream_t stream, Input input, Output output,
    int64_t rows, int64_t columns, double epsilon,
    Compute* inverse_scale) {
  RowFakeScaleKernel<Input, Output, Compute, 128>
      <<<1, 128, 0, stream>>>(input, output, static_cast<int>(rows),
                              static_cast<int>(columns), epsilon,
                              inverse_scale);
  return cudaPeekAtLastError();
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteDisconnectedScaleImpl(
    cudaStream_t stream, Input input, Output output,
    int64_t rows, int64_t columns, double epsilon,
    Compute* inverse_scale) {
  RowDisconnectedScaleKernel<Input, Output, Compute, 128>
      <<<1, 128, 0, stream>>>(input, output, static_cast<int>(rows),
                              static_cast<int>(columns), epsilon,
                              inverse_scale);
  return cudaPeekAtLastError();
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteScale(cudaStream_t stream, Input input,
                       Output output, int64_t rows, int64_t columns,
                       double epsilon, Compute* inverse_scale) {
  if (columns <= 1024) {
    return RouteScaleImpl(stream, input, output, rows, columns,
                          epsilon, inverse_scale);
  } else {
    bool cached = false;
    {
      cudaError_t status =
          RouteScaleImpl(stream, input, output, rows, columns,
                         epsilon, inverse_scale);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteScaleImpl(stream, input, output, rows, columns,
                            epsilon, inverse_scale);
    }
    return cudaSuccess;
  }
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteFakeScale(
    cudaStream_t stream, Input input, Output output,
    int64_t rows, int64_t columns, double epsilon,
    Compute* inverse_scale) {
  if (columns <= 1024) {
    return RouteFakeScaleImpl(
        stream, input, output, rows, columns,
        epsilon, inverse_scale);
  } else {
    bool cached = false;
    {
      cudaError_t status = RouteFakeScaleImpl(
          stream, input, output, rows, columns,
          epsilon, inverse_scale);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteFakeScaleImpl(
          stream, input, output, rows, columns,
          epsilon, inverse_scale);
    }
    return cudaSuccess;
  }
}

template<typename Input, typename Output, typename Compute>
cudaError_t RouteDisconnectedScale(
    cudaStream_t stream, Input input, Output output,
    int64_t rows, int64_t columns, double epsilon,
    Compute* inverse_scale) {
  if (columns <= 1024) {
    return RouteDisconnectedScaleImpl(
        stream, input, output, rows, columns,
        epsilon, inverse_scale);
  } else {
    bool cached = false;
    {
      cudaError_t status = RouteDisconnectedScaleImpl(
          stream, input, output, rows, columns,
          epsilon, inverse_scale);
      if (status != cudaSuccess) { return status; }
      cached = true;
    }
    if (!cached) {
      return RouteDisconnectedScaleImpl(
          stream, input, output, rows, columns,
          epsilon, inverse_scale);
    }
    return cudaSuccess;
  }
}
