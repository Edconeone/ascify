# Ascend950PR SIMT 硬件上限探针

本目录只回答 softmax / RMSNorm 调优前必须量化的五个问题：

1. 单 block 与铺满 56 个 AIV 时，SIMT kernel 的启动底噪是多少；
2. 2 / 4 / 8 / 16 字节 load-store 宽度分别能到多少 GM 带宽；
3. 固定 grid 和 16 字节访存时，32 到 1024 threads/block 的最优点在哪里，以及
   每 AIV 2048 resident threads 的上限形态能否继续提升；
4. `asc_reduce_add` 硬件 warp 归约是否优于 5 级 `asc_shfl_xor`；
5. `expf` 与 `rsqrtf` 的整卡吞吐上限是多少。

它不是算子 benchmark，也不做 A800 对比。这里得到的是 950PR 的本机基线，后续
softmax / RMSNorm 性能结果应先与这些上限比较，才能判断瓶颈来自访存、归约、数学
函数还是 launch 粒度。

与本门禁直接相关的文件：

- `simt_hw_probes.cce`：host+device 硬件探针；
- `rowwise_recipe_traits_compile.cce`：public recipe header 的 CCEC compile-only 契约；
- `rowwise_recipe_contract_probe.cce`：不启动设备 kernel 的 host runtime fallback 契约；
- `parse_results.py`：只依赖 Python 标准库的 CSV 校验与汇总器；
- `README.md`：构建、运行、字段口径和反馈规则。

## 构建接入

源码采用仓库已经验证过的 DPP/SIMT 语法：`__global__` kernel、四尖括号 ACL
stream 启动、`simt_api` warp/math intrinsic。CANN 9.1.0 beta3 下应交给
Bisheng 套件里的 `ccec -x dpp` 驱动；不要对这个混合 host+device 文件使用只面向
ASC kernel 的 `--asc-aicore-lang`。

在统一 `softmax_rmsnorm_950` 工程内，优先使用已经接入的脚本：

```bash
tests/softmax_rmsnorm_950/scripts/build.sh probe
tests/softmax_rmsnorm_950/scripts/run_probes.sh
```

`build.sh probe` 不依赖已转换的 softmax header，二进制只写入
`.work/softmax_rmsnorm_950/bin/`。`run_probes.sh` 通过现有
`select_device.sh` 动态选择健康空闲设备，并在同一个项目锁内先执行
`rowwise_recipe_contract_probe`，再完成硬件探针、17 行
校验和解析；产物统一写入 `.work/softmax_rmsnorm_950/results/<run_id>.{build.log,run.log,csv,summary.md}`。

上层统一构建脚本可以把下列命令作为接入模板，并复用其现有的完整 link library
集合：

```bash
CANN_ROOT=/path/to/user-owned/cann
source "$CANN_ROOT/set_env.sh"

"$CANN_ROOT/tools/bisheng_compiler/bin/ccec" \
  -x dpp --cce-aicore-arch=dav-c310-vec \
  -std=c++17 -O2 -DNDEBUG \
  simt_hw_probes.cce \
  -I"$CANN_ROOT/include" \
  -I"$CANN_ROOT/include/ascendc/host_api" \
  -I"$CANN_ROOT/compiler/ascendc/include/highlevel_api" \
  -I"$CANN_ROOT/compiler/tikcpp/tikcfw" \
  -I"$CANN_ROOT/compiler/tikcpp/tikcfw/impl" \
  -I"$CANN_ROOT/compiler/tikcpp/tikcfw/interface" \
  -I"$CANN_ROOT/compiler/tikcpp/tikcfw/lib" \
  -I"$CANN_ROOT/compiler/tikcpp/tikcfw/lib/matmul" \
  -I"$CANN_ROOT/x86_64-linux/asc/include" \
  -L"$CANN_ROOT/lib64" \
  -lascendcl -lruntime -lregister -lerror_manager \
  -lprofapi -lascendalog -lmmpa -lascend_dump -lc_sec \
  -lstdc++ -lm \
  -o simt_hw_probes
```

选择 `dav-c310-vec` 是为了保证只在 AIV 上运行 SIMT probe；beta3 编译器内部对应
950PR 的 `__NPU_ARCH__=3510` / `__CCE_AICORE__=310`。不要修改系统 CANN、驱动
或全局环境，构建只使用 `CANN_ROOT` 指向的用户态 beta3 包。beta3 不提供
`libascendc_runtime`，因此 link 参数中不得沿用 beta1 的
`-lascendc_runtime`。

## 运行

推荐先把目标物理卡映射成逻辑 device 0，再运行完整探针。stdout 只含 CSV，诊断和
实际探测到的 AIV/warp/thread 配置写到 stderr：

```bash
ASCEND_RT_VISIBLE_DEVICES=<physical_device> \
  ./simt_hw_probes --device 0 > simt_hw_probes.csv

python3 parse_results.py --strict simt_hw_probes.csv
```

第一次验证构建或排错时可用较小工作量：

```bash
ASCEND_RT_VISIBLE_DEVICES=<physical_device> \
  ./simt_hw_probes --device 0 --quick > simt_hw_probes_quick.csv
```

完整默认值为 256 MiB copy、5 个 timing sample、每个 sample 10 次 throughput
kernel、每 lane 256 次归约/数学调用，以及 `32 × AIV 数` 个 throughput blocks。
每行报告各 sample 的中位数与最小值。可用以下选项覆盖：

```text
--bytes-mib N
--warmup N
--iterations N
--samples N
--launch-iterations N
--inner-iterations N
--threads N
--blocks-per-aiv N
```

线程曲线始终覆盖 `32,64,128,256,512,1024,2048`。其中前六点是实际
threads/block；beta3 A5 SIMT 的合法 block 上限为 1024，ACL
`MAX_THREAD_PER_VECTOR_CORE=2048` 表示每 AIV 最大并发线程数，而不是合法的
2048-thread block。因此最后一点使用两组 1024-thread blocks，variant 明确记录为
`copy_16b_2x1024`，并将 grid 扩为两倍，让调度器暴露每 AIV 2048 resident-thread
形态。`--threads` 只控制访存宽度对比、warp reduce 和 math probe 的统一
threads/block，必须不大于 1024。

## CSV 口径

正常 950PR 完整运行固定输出 17 行结果，schema version 为 1：

| probe | 行数 | 主要指标 | 工作量定义 |
|---|---:|---|---|
| `launch_floor` | 2 | `ns_per_launch` | 一个可观察的 32-thread SIMT kernel |
| `copy_bandwidth` | 4 | `gbps` | source read + destination write，即 `2 × bytes` |
| `thread_scaling` | 7 | `gbps` | 32–1024 改 threads/block；2048 用 2×1024 合法 blocks |
| `warp_reduce` | 2 | `gops` | logical warp reductions/s，不是标量 add 数 |
| `math` | 2 | `gops` | 每 lane 的 `expf` 或 `rsqrtf` calls/s |

公共字段：

- `elapsed_ms_median` / `elapsed_ms_min`：一个 batch 的 ACL event 时间；
- `timed_launches`：该 batch 中的 kernel 次数；
- `traffic_bytes`：中位数 batch 对应的读写总字节数；
- `operations`：中位数 batch 中的 logical reduction 或 math call 数；
- `checksum` / `status`：计时后的小规模正确性防护。

带宽用十进制 GB/s。`parse_results.py` 会验证表头、schema、数值有限性，列出完整
thread 曲线并标记最佳点；`--format csv` 可让上层脚本继续机器处理。`--strict`
会在任意行 `status != ok` 时返回非零。

## Target recipe 编译契约

`rowwise_recipe_traits_compile.cce` 是 CCEC compile-only 门禁，`build.sh all`
和 `build.sh probe` 都会将它编译到唯一工作根的
`.work/softmax_rmsnorm_950/probes/`。其 `static_assert` 覆盖：

- FP16 storage / FP32 compute 的 direct load/store；
- compile-time affine `true/false` 与 weight accessor；
- exact-owner 对派生类继承 marker 的拒绝；
- volatile input/output pointer 的拒绝；
- double inverse-RMS 的可编译 `NotHandled` fallback；
- dispatcher ComputeType 必须精确为 float 且与 load/store marker alias 相同；
- double/custom dispatcher ComputeType 的可编译 `NotHandled` fallback；
- mismatch 分支在编译期不得实例化 load/store accessor；
- RMSNorm rows/columns 仅在原 CUDA global kernel 的 `int` 参数可无损表示时 handled；
- caller 预定义 Softmax/RMSNorm 配置宏在 public header 前后值不变；
- `ACLCUB_WARP_SIZE` 在 caller 未定义时不泄漏、预定义时按原值恢复。

`rowwise_recipe_contract_probe.cce` 在 host 侧实际检查精确 float adapter 的 null-stream
precondition：Softmax/RMS 均须返回 `handled=false, status=ACL_SUCCESS`，且不得初始化
设备或启动 kernel；同时用纯 shape contract 检查 RMSNorm 的首个 `INT_MAX+1`
row 必须回退。double/custom dispatcher mismatch 只由上面的 trait
`static_assert` 和 well-formed 调用实例化门禁隔离验证；把它与 null stream 放在同一个
runtime case 会掩盖 compute-type guard 是否真的生效，因此不把该 smoke test 表述为
mismatch runtime 证据。

## 如何反馈给 Ascify

探针数据不直接写死到转换器，而应转成保守、可回退的 target heuristic：

- 16B copy 相对 2B 的提升稳定且显著：对齐和长度可证明时生成 128-bit
  load/store；不满足时保留标量 tail/fallback；
- thread 曲线的峰值：作为 950PR SIMT launch-bound / block-size 候选，不替代
  shape-aware 派发；
- `asc_reduce_add` 显著快于 shuffle：将受支持类型的 warp sum lowering 优先映射
  到硬件 intrinsic，shuffle 保留为兼容回退；
- `expf` 吞吐逼近 softmax 实测需求：说明优化重点应转向减少重复 exp、融合或换用
  经精度验证的 math mode，而不是继续只调访存；
- `rsqrtf` 吞吐逼近 RMSNorm 实测需求：优先减少重复统计/rsqrt 和中间 GM 往返；
- tiny-shape 算子耗时接近 `launch_floor`：优先合并工作、减少 kernel 数，不能把收益
  归因于单个 kernel 内部微调。

只有跨至少两次独立运行仍成立、且 softmax 与 RMSNorm 实测能复现收益的规则，才应
进入 Ascify 的 `dav-c310-vec` target policy。这样增强传统转换工具的生成质量，同时不改
其整体框架。
