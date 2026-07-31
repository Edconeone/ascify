# Softmax / RMSNorm 950PR 转换与调优技术记录

## 测量与算量口径

| 项目 | 固定值 |
|---|---|
| 设备 | Ascend 950PR device 5，56 AIV |
| 编译环境 | CANN 9.1 beta3 用户态包，`dav-c310-vec` |
| 数据类型 | FP16 输入/输出，FP32 归约 |
| 正式采样 | 20 warmup，50 samples，20 inner repeats |
| 正式流程 | prepare 冻结输入、side-by-side 构建并发布 binary bundle；重新选卡后 measure 执行全部门禁及 direct-A → native → direct-B，全程 `SKIP_BUILD=1` |
| 锁 | fd 8 保护 prepare/resume/measure 的 set 状态；fd 9 从选卡后覆盖 measure 重验、门禁、预热和测量，不覆盖 prepare 构建 |
| direct 中心值 | `sqrt(direct_A * direct_B)` |
| 支持域 | 正行列、有限 FP16 输入、`columns <= 32768` |

普通算术吞吐按前向数据流计数：

- Softmax：`3RC - R` 次 FP32 算术；另计 `RC` 次 `exp` 和 `RC - R`
  次比较；
- RMSNorm plain：`3RC + R` 次 FP32 算术；另计 `R` 次 `rsqrt`；
- RMSNorm affine：plain 再加 `RC` 次 FP16 multiply。

表中的 TFLOPS 只使用上述普通算术计数；`exp`/`rsqrt` 单列为 SFU
Gop/s，不折算或加到 TFLOPS。logical GB/s 分别按 Softmax `4RC`、
RMSNorm plain `4RC + 4R`、RMSNorm affine `4RC + 4R + 2C` 字节计算。

## 910C 直接转换

Ascify 构建和转换均在
仓库唯一的 `.work/softmax_rmsnorm_950` 实验根内完成。
四个转换单元统一使用：

```text
ascify-clang <input> \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --cuda-path=<work-root>/cuda \
  --clang-resource-directory=<repo>/ascify_install/include/ascify \
  -o <output> -- -Iinputs -std=c++17
```

`layer_norm.cuh`、`rms_norm.cuh` 和 affine caller adapter 额外
forced-include `cuda_fp16.h`、`cuda_bf16.h`。完整 argv 由
`run_910_conversion_v3.sh` 写入转换 manifest。

输出仍以 `.cuh` 保存，但 CUDA runtime/API、kernel qualifier 和 launch
已转换为 950PR 可编译的 ACL/AscendC 兼容形式；row-wise recipe 再插入以下
结构：

```cpp
#include <ascify/target/dav_c310/rowwise_norm_recipes.hpp>

// 由严格 adapter 证明生成
using ascify_target_direct_load_tag = void;
using ascify_target_adapter_owner_type = DirectLoad;

// Softmax：direct wrapper 入口
const auto result = ::ascify::target::dav_c310::TrySoftmax(...);
if (result.handled) { return result.status; }

// RMSNorm：保留 GetNumBlocks 和失败返回后、原 launch 前
const auto result = ::ascify::target::dav_c310::TryRmsNorm(...);
if (result.handled) { return result.status; }
```

`NotHandled` 继续执行原转换后的 launch 和错误传播，不删除原路径。
RMSNorm affine 的 store 定义在 caller 中，因此必须把
`layer_norm.cuh`、`rms_norm.cuh` 和具体 affine store adapter 一起转换。

| 单元 | 输出 SHA256 | load/store marker | recipe 调用 |
|---|---|---:|---:|
| `softmax.cuh` | `efc941c2f60046a4d26279fc0555449186025b3793264bbf01e9b608a16eee8c` | 1 / 1 | `TrySoftmax` 3 |
| `layer_norm.cuh` | `d7367229578430dac5aa8a6dd91fdd9135eb0fcd7c501f9749c32b17f750a85d` | 1 / 1 | 0 |
| `rms_norm.cuh` | `49fa76c0c732180936690c9f293dc26ea98a4158c429bc6f7e7bff02a67e7618` | 0 / 0 | `TryRmsNorm` 3 |
| affine adapter | `17acd0718ef3be49ee845fbcb6a435b0db7016322c13d6231c62df162078b63c` | 0 / 1 | 0 |

生成头从 910C 证据同步到 950PR 后未做手工修改。

## 反馈前直转与手工 native

反馈前的 `ascify_fast_perf_final_v1` 只使用传统转换和局部 fast-SIMT
rewrite；它没有识别完整 row-wise 算子，也不会自动生成 warp/block 路由、
缓存和 half2 数据路径。手工 native 的固定性能记录为
`native_combined_perf_final_v1`。
该历史手调 run 用于记录优化增益；正式同轮 native control 是同一
`dav_c310::v1` target implementation 的薄入口，用于检验 direct 生成路径能否
完整复现已沉淀实现。

手工 Softmax 使用：

- 16 B `int4` transaction 和 half2 指数/存储；非对齐与尾部走 scalar；
- `asc_reduce_max/add`；warp-per-row 与低行数长列 block-per-row 路由；
- `columns <= 4096` lane-local cache，长列 streaming 或 block UB cache；
- 运行时 AIV 数和工作量推导 grid cap，不写死业务 shape；
- FP32 centered exponent、half2 参数残差修正。

手工 RMSNorm 使用：

- 16 B transaction、每行 FP32 square-sum、`asc_reduce_add`、一次 `rsqrtf`；
- `columns <= 8192` 输入缓存；任意列宽和非对齐使用 tail-safe fallback；
- `rows <= 256 && 4096 <= columns <= 8192` 使用 block-per-row；
- plain/affine block 分别选 512/256 threads；
- affine block 路径限定使用 `__hmulx2`，保持“先归一化到 FP16，再乘 FP16
  weight”的两次舍入语义。

## native 经验进入 Ascify 的实现

没有改写 Ascify 主框架；新增的是一个 opt-in、版本化、fail-close 的
`DavC310TargetRecipe` 和 target library：

1. **adapter 证明**：验证 exact-owner、无继承、FP16 storage/FP32 compute、
   packed row-major 地址、单次直接 load/store；affine 还证明同列 FP16
   weight 和精确乘法语义；
2. **算子数据流证明**：Softmax 证明
   `load → max → subtract → exp → sum → divide → store`；RMSNorm 证明
   `load → square → sum/mean → epsilon → rsqrt → inverse/store`；
3. **effect/CFG 证明**：拒绝额外全局写、atomic、volatile、asm、trap、
   未解析调用、死分支证据、条件 launch 和不完整 row/column coverage；
4. **wrapper 证明与放置**：Softmax 只在三个 direct wrapper 入口插入；
   RMSNorm 保留 dependent `GetNumBlocks` 及其失败边，再在唯一 launch 前插入；
   dispatcher、LogSoftmax 和顶层 RMSNorm wrapper 不命中；
5. **target library**：把手工验证过的 16 B load/store、native reduce、
   cache、warp/block 路由、动态 AIV grid、half2 精化和 fallback 封装在
   `ascify::target::dav_c310::v1`；
6. **运行时契约**：类型、stride、指针、affine weight、epsilon、整数范围或
   `columns <= 32768` 不满足时返回 `NotHandled`。

project-specific 文件名、kernel 和 wrapper 名不是 recipe 触发条件；真实源码
mutation matrix 用重命名用例以及 coverage 缺口、伪 reducer、副作用、错误
CFG、宏插入点、名字冲突和 member wrapper 共 38/38 验证当前源码族。
默认 `portable + precise` 行为不变；只有
`dav-c310-vec + fast` 启用该 recipe。

泛化单位是经 AST、数据流、effect 和 CFG 证明的源码结构族，不是文件名、
wrapper 名或业务 shape 查表。当前 mutation 证明覆盖所列重命名与扰动；语义
等价但超出当前规范化或证明形态的源码会 fail-close。

## 反馈后直转性能复现

正式集合 `ascify_recipe_formal_v3_final11` 在同卡同轮比较反馈后生成头与
side-by-side native control。`反馈前直转` 和 `手调 native` 是较早的固定
run；后者早于新增的跨卡 `+65504/-65504` 极差门禁，因此只记录调优阶段性能，
不充当最终正确性对照。

生成头未经手改；direct 通过 recipe 插入的 marker/`Try*` 调用进入已封装的
`dav_c310::v1` target implementation。正式同轮 native control 是该实现的
薄入口。因此接近 100% 证明的是 Ascify 能直接复现沉淀后的实现及其调度和
运行时契约，不代表从任意 CUDA 源码自动综合出新的底层 kernel。

| 算子 | Rows × Cols | 反馈前直转 TFLOPS | 手调 native TFLOPS | 反馈后直转中心 TFLOPS | 同轮 native TFLOPS | 直转/native | 直转 GB/s | SFU Gop/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 8192 × 128 | 0.056990 | 0.152507 | 0.132667 | 0.133008 | 99.743% | 177.351 | 44.338 |
| Softmax | 8192 × 512 | 0.129204 | 0.395197 | 0.371593 | 0.372530 | 99.749% | 495.781 | 123.945 |
| Softmax | 8192 × 1024 | 0.055630 | 0.428503 | 0.405123 | 0.406587 | 99.640% | 540.340 | 135.085 |
| Softmax | 1024 × 4096 | 0.088006 | 0.284904 | 0.272024 | 0.273024 | 99.634% | 362.728 | 90.682 |
| Softmax | 128 × 8192 | 0.125280 | 0.250189 | 0.241913 | 0.244498 | 98.943% | 322.563 | 80.641 |
| RMSNorm plain | 8192 × 128 | 0.040316 | 0.124532 | 0.123124 | 0.124996 | 98.502% | 165.018 | 0.320 |
| RMSNorm plain | 8192 × 512 | 0.140703 | 0.402231 | 0.400580 | 0.402042 | 99.636% | 534.801 | 0.261 |
| RMSNorm plain | 8192 × 1024 | 0.167103 | 0.517547 | 0.517985 | 0.519120 | 99.781% | 691.096 | 0.169 |
| RMSNorm plain | 1024 × 4096 | 0.172435 | 0.365915 | 0.362087 | 0.366221 | 98.871% | 482.862 | 0.029 |
| RMSNorm plain | 128 × 8192 | 0.199494 | 0.277952 | 0.279709 | 0.281224 | 99.461% | 372.976 | 0.011 |
| RMSNorm affine | 8192 × 128 | 0.052888 | 0.095796 | 0.095645 | 0.095943 | 99.689% | 96.210 | 0.186 |
| RMSNorm affine | 8192 × 512 | 0.136038 | 0.284341 | 0.285155 | 0.285971 | 99.715% | 285.590 | 0.139 |
| RMSNorm affine | 8192 × 1024 | 0.147771 | 0.345007 | 0.345258 | 0.345880 | 99.820% | 345.532 | 0.084 |
| RMSNorm affine | 1024 × 4096 | 0.174431 | 0.309557 | 0.309783 | 0.310763 | 99.685% | 309.990 | 0.019 |
| RMSNorm affine | 128 × 8192 | 0.214632 | 0.320269 | 0.317954 | 0.319324 | 99.571% | 319.225 | 0.010 |

| 组 | 反馈后直转 / 反馈前直转 | 反馈后直转 / 历史手调 | 反馈后直转 / 同轮 native |
|---|---:|---:|---:|
| Softmax | 3.110× | 0.935× | 0.995412× |
| RMSNorm plain | 2.398× | 0.996× | 0.992493× |
| RMSNorm affine | 1.877× | 0.999× | 0.996959× |

Softmax 的历史手调 run 未包含后加的跨卡极差用例。当前 target 和同轮 native
都在 half2 打包前用 FP32 select 把有限负参数限制到 `-65504`；这消除了跨卡
复测中 device 5 曾出现的 nonfinite，但相对历史手调 run 的组几何均值为
0.935。反馈后直转与
采用同一正确性契约的 native control 为 0.995412。

## 正确性与性能门禁

| 路径 | Softmax boundary | RMSNorm boundary | tune oracle |
|---|---:|---:|---:|
| 反馈后直转 | 42/42 | 18/18 | 5/5 + 10/10 |
| native control | 42/42 | 18/18 | 5/5 + 10/10 |

Softmax boundary 覆盖 Warp、Block、Warp-stream、随机/常量、guard/canary、
2-byte 非对齐基址以及有限 FP16 `+65504/-65504` 极差。正式性能验证要求：

- 每个 shape 的 `sqrt(A*B)/native >= 0.90`；
- Softmax、RMSNorm plain、affine 各组几何均值 `>= 0.95`；
- 每个 shape 的 `max(A,B)/min(A,B) <= 1.05`。

实测三组为 `0.995412 / 0.992493 / 0.996959`；最大 A/B spread 为
`1.007300`。正式 direct/native boundary 各 60/60，tune 各 15/15，
A/native/B 性能各 15/15。

prepare 在 fd 8 下校验转换证据、冻结构建输入、构建 direct/native 并原子
发布 binary bundle，不占用设备。构建后丢弃调用者遗留的 `DEVICE`，重新选择
健康、无进程且 AICore/AIVector/NPU/HBM bandwidth 使用率均为 0 的 950PR。
measure 继承
fd 8 和 fd 9，重验 freeze/bundle 后只从 bundle 执行。若选卡前停止，resume
用同一 set/tag 复核已发布产物并跳过构建；已有 phase 证据则拒绝复用。
bundle schema v2 绑定 build-input snapshot 的规范路径和 SHA256；bundle
目录、manifest 与 8 个二进制均只读，并在每个 phase 和最终 bracket 再校验。

每个 phase 的即时 pre/post snapshot 绑定 run ID、hostname、device 和时间，
并严格要求 `Health Status: OK`、`No process in device.` 及四项使用率为 0。
这证明两个端点空闲；project-local fd 9 只能排除遵守同一锁协议的运行，不能
证明非协作外部调度器在整个 phase 区间绝对没有短暂进入。

## 950PR 探针与被拒绝实现

| 探针 | 实测 |
|---|---:|
| 1 block / 56 AIV launch | 1.549 / 1.576 µs |
| 16 B copy | 1210.099 GB/s |
| 1024 threads / 2 × 1024 resident | 1210.250 / 1133.908 GB/s |
| native reduce / shuffle butterfly | 23.491 / 8.640 Gwarp/s，2.719× |
| `expf` / `rsqrtf` | 937.253 / 842.657 Gcall/s |

- Softmax 直接 `h2exp` 在 128 列的 scaled relative error 约
  `2.81e-3`，超过 `2e-3` 门禁；保留 residual refinement；
- half2 saturating-convert 候选在 device 5 的极差用例产生 8 个 nonfinite；
  `fmax` 正确但更慢；保留 FP32 select；
- RMSNorm affine half2 若用于所有 warp shape，会在 32/64 列出现错误；
  仅允许已验证的 block-per-row 路径；
- affine block 128/256/512/1024 threads 的 8192 列试验约为
  164.62/204.04/188.58/160.97 GB/s，选择 256；
- 2048-thread block 非法；`float[256]` 每线程 cache 超出 local-memory
  限制；未进入 target library。

这些结果是已探索配置中的本地最优，不是对未枚举编译器代码形态、时钟状态或
未来 CANN 版本的绝对硬件上限。

## 泛化边界与证据哈希

当前 recipe 只覆盖结构上可证明的 packed row-major FP16 Softmax、RMSNorm
plain 和同列 FP16 affine store；不覆盖 LogSoftmax、mask、bias、dropout、
非连续 stride、未知 caller store 或 `columns > 32768`。system-header
provenance 依赖受信任的 910C 编译参数；运行时 route 名称来自已冻结 runner
配置，不是设备 telemetry。CSV 的 `cuda_pred_path` 是源 CUDA
wrapper/shape 的预测标签，不能作为 target 实际分支选择的证据。

| 产物 | SHA256 |
|---|---|
| Ascify 910C binary | `276fc8426569b8f12ada9be3fd18e52a6733e54654e0521638988a57d9482dde` |
| `DavC310TargetRecipe.cpp` | `1469cfa6c87f04d973aca93a2bb2e98cb403edd9032215e58c3408589d3a3c59` |
| 转换 manifest | `ae2d4e3acefeeab3bd19ad49f785f976ce6e29c68139686759e1a7e120b586cf` |
| mutation matrix | `fcf1b52363565e0e3431c5b41f1a4c85bd084ab3a4dd45edb9a637571dc00997` |
| 正式 set manifest | `3d4b335107b3d75d5727ef1ee262fc328c44d197be07e302662a32bb1b55697a` |
| build-input snapshot | `882044387cd31d9cc6018ffdfed5cef2603c3392c1c8dd8cb559d7d79e48dd38` |
| binary-bundle manifest | `38f2a7cb694464ae3b8b73e454179498f28aa06a0f050198659155dc01942753` |
| A/native/B summary | `8845bc40a7555380842a9aa2f005da6912b62349c03041623c6cba4d28d085d1` |
| formal metrics | `87bd0e99da6ea32921783061e8473c22ebb93a11d2fe84b0b3856a37ffc3d765` |
