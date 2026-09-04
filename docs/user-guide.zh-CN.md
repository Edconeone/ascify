# Ascify 使用手册

本文面向第一次使用 Ascify、能够操作 Linux 命令行的开发者。完成本手册后，你将能够在一台新机器上：

1. 从 GitHub 克隆 Ascify；
2. 使用已有或新构建的 LLVM/Clang 编译 `ascify-clang`；
3. 安装 Ascify 及其配套头文件；
4. 将仓库自带的 CUDA 示例转换为面向 ACL/DPP/Ascend 兼容层的源码；
5. 判断“源码转换成功”是否成立，并知道后续目标编译和设备运行还需要什么。

本手册以 Linux x86_64 或 AArch64 为主。Windows SDK 集成、Ascify 内部开发和算子性能调优不在本文范围内。

## 1. 先理解 Ascify 做什么

Ascify 是基于 Clang 的 CUDA C/C++ **源码转换器**。最基本的工作流是：

```text
CUDA 源码
  -> ascify-clang 解析并改写
  -> 面向 ACL/DPP/Ascend 兼容层的 C/C++/CCE 源码
  -> 使用 CANN/目标工程继续编译、链接和上板验证
```

请区分下面两个结果：

| 结果 | 说明 |
|---|---|
| Ascify 返回 0，并生成非空输出 | 仅说明本次源码解析和转换完成 |
| 生成代码在目标环境编译、链接并通过设备测试 | 才说明对应程序完成了目标侧验证 |

运行 Ascify 做源码转换时不需要 NVIDIA GPU，也不需要昇腾 NPU；但机器上必须有可供 Clang 解析的完整 CUDA Toolkit 目录。编译和运行转换后的目标程序通常需要匹配的 CANN 工具链、Ascify 兼容头、目标运行库和昇腾设备。

## 2. 环境要求

### 2.1 必需依赖

| 依赖 | 用途 | 备注 |
|---|---|---|
| Git | 克隆源码 | 任意近期版本 |
| Bash | 执行 `build.sh` 和 `run.sh` | 脚本使用 Bash 语法 |
| CMake | 配置 Ascify | Ascify 要求 3.16.8 或更高版本 |
| Ninja | 默认构建生成器 | 也可用 `CMAKE_GENERATOR` 指定其他生成器 |
| C/C++ 编译器 | 编译 Ascify | 必须能为当前主机生成程序 |
| LLVM + Clang 开发构建 | 提供 Clang 前端、库和 CMake 配置 | 仅安装 `clang` 可执行文件通常不够 |
| CUDA Toolkit 目录 | 解析 CUDA 头文件和内建设施 | 转换机器不要求有 NVIDIA GPU |

Python 3 不是构建 `ascify-clang` 的硬依赖，但运行仓库测试和读取 JSON 回执时建议安装。

已经验证过的一组参考环境如下。这是可复现参考，不表示只支持这些版本：

| 项目 | 已验证值 |
|---|---|
| 操作系统/架构 | Linux AArch64 |
| LLVM/Clang | 23.0.0git，`llvm-project` commit `caf619642a6dbb216969a9450d33dbac5a8d30df` |
| CUDA 解析目录 | CUDA Toolkit 12.8 完整布局 |
| CMake | 4.3.4 |
| Ninja | 1.13.0 |

### 2.2 安装基础工具

以 Ubuntu/Debian 为例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git ninja-build python3
```

其他发行版请安装同名或等价软件包。先检查命令是否可用：

```bash
git --version
cmake --version
ninja --version
python3 --version
```

如果发行版自带的 CMake 太旧，请按 CMake 官方方式升级。若还需要从源码编译 LLVM，应同时满足该 LLVM 版本自己的 CMake 最低版本要求。

## 3. 准备 LLVM/Clang

### 3.1 使用已有 LLVM 构建树

这是最快、也是当前验证过的方式。Ascify 需要一个已经配置并编译好的 `llvm-project`，其中至少应包含：

```text
llvm-project/
├── clang/
├── llvm/
└── build/
    ├── bin/clang
    ├── bin/clang++
    ├── lib/cmake/clang/ClangConfig.cmake
    └── lib/cmake/llvm/LLVMConfig.cmake
```

设置路径并检查：

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"

test -x "$LLVM_BUILD_DIR/bin/clang"
test -x "$LLVM_BUILD_DIR/bin/clang++"
test -f "$LLVM_BUILD_DIR/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$LLVM_BUILD_DIR/lib/cmake/clang/ClangConfig.cmake"

"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
```

最后一条命令应打印当前主机可用的 target triple。LLVM 只包含 NVPTX/AArch64 等非当前主机后端、或 `LLVM_DEFAULT_TARGET_TRIPLE` 为空时，CMake 的编译器检查可能失败。

### 3.2 没有 LLVM 时从源码构建

下面给出一个最小参考配置。LLVM 本身的编译会占用较多时间、内存和磁盘；请先阅读 [LLVM CMake 构建说明](https://llvm.org/docs/CMake.html)。

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# 可选：切到 Ascify 已验证过的 LLVM 快照。
git checkout caf619642a6dbb216969a9450d33dbac5a8d30df

case "$(uname -m)" in
  x86_64)
    LLVM_HOST_TARGET=X86
    LLVM_HOST_TRIPLE=x86_64-unknown-linux-gnu
    ;;
  aarch64|arm64)
    LLVM_HOST_TARGET=AArch64
    LLVM_HOST_TRIPLE=aarch64-unknown-linux-gnu
    ;;
  *) echo "请为当前架构选择 LLVM target" >&2; exit 1 ;;
esac

cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="${LLVM_HOST_TARGET};NVPTX" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="${LLVM_HOST_TRIPLE}"

cmake --build build --target clang lld --parallel 2
cd ..
```

构建完成后设置：

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
```

如果使用发行版提供的 LLVM 开发包，`LLVM_BUILD_DIR` 应指向同时包含 `bin/clang`、LLVM CMake 配置和 Clang CMake 配置的前缀。不同发行版会拆分这些文件；缺少 `ClangConfig.cmake` 时，请安装对应的 Clang development package，或改用完整源码构建树。

## 4. 准备 CUDA 解析目录

安装适合当前主机架构的 CUDA Toolkit，或使用已有的完整 Toolkit 目录。安装方法参见 [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)。常见路径是 `/usr/local/cuda`：

```bash
export CUDA_PATH=/usr/local/cuda

test -f "$CUDA_PATH/include/cuda.h"
test -f "$CUDA_PATH/include/cuda_runtime.h"
test -d "$CUDA_PATH/nvvm"
```

只有头文件、没有 `nvvm`/`libdevice` 的裁剪目录可能无法让 Clang 完成 CUDA 检测。`CUDA_PATH` 应指向 Toolkit 根目录，而不是它的 `include` 子目录。

## 5. 克隆并编译 Ascify

### 5.1 克隆

```bash
git clone https://github.com/Edconeone/ascify.git
cd ascify

git rev-parse HEAD
git status --short --branch
```

记录 commit 有助于复现转换结果。

### 5.2 编译

确认第 3 节中的变量仍然有效，然后执行：

```bash
export LLVM_PROJECT_PATH=/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
export ASCIFY_BUILD_JOBS=2

./build.sh
```

`build.sh` 默认会：

- 使用 Ninja；
- 生成 Release 构建；
- 将构建目录设为仓库内的 `build/`；
- 将默认安装目录设为仓库内的 `ascify_install/`；
- 优先使用 LLVM 构建树中的 `clang`、`clang++` 和 `lld`。

检查二进制：

```bash
test -x build/ascify-clang
build/ascify-clang --help >/dev/null
```

如果 LLVM 的 stage-1 编译器不能为当前主机生成程序，可显式选择系统编译器：

```bash
export ASCIFY_CC=/usr/bin/clang
export ASCIFY_CXX=/usr/bin/clang++
./build.sh
```

常用覆盖项：

```bash
export BUILD_DIR="$PWD/build-release"
export INSTALL_ROOT="$PWD/ascify-install-release"
export ASCIFY_BUILD_JOBS=4
export CMAKE_GENERATOR=Ninja
./build.sh
```

后续命令要使用同一个 `BUILD_DIR` 和 `INSTALL_ROOT`。

### 5.3 安装

使用默认目录时执行：

```bash
cmake --install build

test -x ascify_install/bin/ascify-clang
test -f ascify_install/include/ascify/include/__clang_cuda_runtime_wrapper.h
test -f ascify_install/include/ascify/ascify_cuda_compat.hpp
```

安装步骤会复制 Ascify 二进制、兼容层头文件、frontend compatibility profile 和当前 Clang resource headers。不要只复制单个 `ascify-clang` 文件到另一台机器；二进制仍需要匹配的 resource headers，并可能依赖 LLVM 的共享库。

## 6. 完成第一次转换

以下命令使用仓库自带的 `examples/vector_add.cu`，选择纯 SIMT 目标策略，并生成机器可读的迁移回执。

```bash
export CUDA_PATH=/usr/local/cuda
export CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/include/ascify"
export ASCIFY_BINARY="$PWD/ascify_install/bin/ascify-clang"

mkdir -p generated

./run.sh examples/vector_add.cu \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  --migration-receipt="$PWD/generated/vector_add.receipt.json" \
  -o "$PWD/generated/vector_add.cu.dpp" \
  -- -std=c++17
```

如果 CUDA Toolkit 不在 `/usr/local/cuda`，把 `CUDA_PATH` 改成实际根目录。`run.sh` 会将 `CUDA_PATH` 和 `CLANG_RESOURCE_DIRECTORY` 转换成 Ascify 参数，并调用 `ASCIFY_BINARY`。

命令末尾的 `--` 用于分隔 Ascify 选项和普通 Clang 选项。上例中的 `-std=c++17` 会传给 Clang，而不会被当作 Ascify 选项。

### 6.1 验证结果

如果转换命令没有报错退出，再检查输出文件：

```bash
test -s generated/vector_add.cu.dpp
test -s generated/vector_add.receipt.json
```

再检查关键改写和回执状态：

```bash
grep -E 'ascify_cuda_compat|acl/acl.h|aclrtMalloc' \
  generated/vector_add.cu.dpp

grep -Eq '"status"[[:space:]]*:[[:space:]]*"succeeded"' \
  generated/vector_add.receipt.json
```

回执中的成功状态只覆盖 `source_conversion`。它不会声称目标编译、链接、设备执行、数值正确性或性能已经通过。

### 6.2 不安装也可以转换

如果只执行了构建而没有执行 `cmake --install`，可直接使用 LLVM 的 resource directory：

```bash
export ASCIFY_BINARY="$PWD/build/ascify-clang"
export CLANG_RESOURCE_DIRECTORY="$("$LLVM_BUILD_DIR/bin/clang" -print-resource-dir)"
export CUDA_PATH=/usr/local/cuda

./run.sh examples/vector_add.cu \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  -o "$PWD/generated/vector_add.cu.dpp" \
  -- -std=c++17
```

安装后的 resource 目录更适合固定和记录一次转换环境，因此正式转换建议先安装。

## 7. 转换自己的源码

### 7.1 单个源文件

```bash
./run.sh /path/to/project/kernel.cu \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  -o "$PWD/generated/kernel.cu.dpp" \
  -- -std=c++17 -I/path/to/project/include -DMY_FEATURE=1
```

传给 Clang 的 include、宏、语言标准和 sysroot 应与原项目的实际编译条件一致。源码在原工程中能由 `nvcc` 编译，不代表在缺少相同 `-I`、`-D` 或生成头文件时也能被 Ascify 解析。

### 7.2 带本地头文件的源码

默认只转换主输入文件。要收拢双引号引用的用户本地头文件，可选择：

- `--local-headers`：只处理主文件直接包含的合格本地头；
- `--local-headers-recursive`：递归处理合格的本地头闭包。

示例：

```bash
./run.sh /path/to/project/kernel.cu \
  --local-headers-recursive \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  -o "$PWD/generated/kernel.cu.dpp" \
  -- -std=c++17 -I/path/to/project/include
```

转换后的头文件会发布到 `generated/kernel.cu.dpp.headers/`。该功能不处理任意系统头，也不会自动放宽 CUDA API 支持范围。完整边界见[本地头文件闭包说明](local-header-closure.md)。

### 7.3 直接调用二进制

不使用 `run.sh` 时，最小形式为：

```bash
"$ASCIFY_BINARY" /path/to/input.cu \
  --cuda-path="$CUDA_PATH" \
  --clang-resource-directory="$CLANG_RESOURCE_DIRECTORY" \
  --target-policy=dav-c310-vec \
  --simt-math=precise \
  --target-recipe=none \
  -o /path/to/output.cu.dpp \
  -- -std=c++17
```

## 8. 选择转换模式

建议第一次先使用 `dav-c310-vec + precise + target-recipe=none`，确认基础解析和转换链路正常，再按需求启用更窄的优化路径。

| 目标 | 关键选项 | 说明 |
|---|---|---|
| 保守默认转换 | `--target-policy=portable --simt-math=precise` | 两项都是默认值 |
| 面向 dav-c310 的纯 SIMT 转换 | `--target-policy=dav-c310-vec --target-recipe=none` | `target-recipe=none` 是默认值 |
| 允许已守护的 fast-SIMT 改写 | 再加 `--simt-math=fast` | 可能改变浮点转换策略，需目标验证 |
| 950PR row-wise SIMD+SIMT Hybrid | 再加 `--target-recipe=dav-3510-rowwise-simd-v1` | 只接受已证明的 Softmax/RMSNorm/LayerNorm 结构 |

Hybrid 模式必须同时选择：

```bash
--target-policy=dav-c310-vec
--simt-math=fast
--target-recipe=dav-3510-rowwise-simd-v1
```

否则 Ascify 会拒绝该组合。Hybrid 生成物还需要单独构建并链接版本化 target support；完整流程见 [SIMD+SIMT 转换指南](rowwise-simd-conversion.md)。

`--frontend-compat=ascify-admitted-v1` 是窄范围、版本化的 parser compatibility profile，不是通用的 CUDA 头文件兼容开关。普通输入应先保留默认的 `--frontend-compat=none`。

## 9. 常用选项速查

| 选项 | 用途 |
|---|---|
| `-o <file>` | 输出一个文件 |
| `-o-dir <dir>` | 将多个输入输出到目录 |
| `-inplace` | 原地改写；默认保留备份 |
| `-no-backup` | 与原地改写配合，禁止备份 |
| `-examine` | 只检查并打印统计，不写转换输出 |
| `-print-stats` | 打印转换统计 |
| `-print-stats-csv` | 输出 CSV 统计 |
| `-o-stats <file>` | 指定统计文件 |
| `-cuda-gpu-arch=sm_XX` | 指定 CUDA 解析架构 |
| `--local-headers` | 转换直接引用的合格本地头 |
| `--local-headers-recursive` | 递归转换合格本地头闭包 |
| `--migration-receipt=<file>` | 生成确定性的 JSON 源码转换回执 |
| `--target-policy=portable\|dav-c310-vec` | 选择目标策略 |
| `--simt-math=precise\|fast` | 选择 SIMT 浮点改写模式 |
| `--target-recipe=none\|dav-3510-rowwise-simd-v1` | 选择是否启用显式 Hybrid recipe |
| `--frontend-compat=none\|ascify-admitted-v1` | 选择前端兼容 profile |

查看当前二进制的完整选项：

```bash
"$ASCIFY_BINARY" --help
```

## 10. 运行项目检查

### 10.1 不依赖真实二进制的 host gate

```bash
sh tests/run_release_checks.sh
```

该检查验证源码改写合同和 host Python 测试，适合在修改源码或文档后运行。

### 10.2 带真实转换器的 gate

```bash
ASCIFY_BINARY="$PWD/build/ascify-clang" \
ASCIFY_CUDA_PATH="$CUDA_PATH" \
ASCIFY_CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/include/ascify" \
sh tests/run_release_checks.sh
```

这会增加真实 CUDA fixture 转换检查，耗时也更长。

## 11. 目标编译和设备运行

Ascify 不为任意 CUDA 项目自动生成完整的 CANN 工程。源码转换之后，通常还需要：

1. 在目标机器安装并初始化匹配的 CANN Toolkit；
2. 将 `ascify_install/include` 加入目标编译 include 路径；
3. 按原工程结构补齐 host/runtime 依赖与 CANN 链接参数；
4. 使用目标芯片对应的编译选项构建；
5. 在隔离设备上做数值正确性、错误路径、稳定性和性能验证。

如果使用 `dav-3510-rowwise-simd-v1`，还必须构建 `runtime/dav_3510/rowwise/` 下的四个版本化共享库，并确保链接名和 SONAME 文件均可用。参见 [SIMD+SIMT 转换指南](rowwise-simd-conversion.md)。

不要把以下任一项单独当作设备运行成功：

- Ascify 返回 0；
- JSON 回执为 `succeeded`；
- 生成文件非空；
- 目标编译或链接单独通过；
- 程序只运行一次但没有独立数值检查。

## 12. 常见问题排查

### `LLVM_PROJECT_PATH` 未设置

现象：`build.sh` 立即退出并提示设置变量。

处理：

```bash
export LLVM_PROJECT_PATH=/absolute/path/to/llvm-project
export LLVM_BUILD_DIR="$LLVM_PROJECT_PATH/build"
```

### `missing build dependency: .../bin/clang`

`LLVM_BUILD_DIR` 指错了目录，或 LLVM/Clang 尚未编译。确认 `bin/clang` 和 `bin/clang++` 都存在且可执行。

### CMake 找不到 LLVM 或 Clang

现象通常包含 `LLVMConfig.cmake` 或 `ClangConfig.cmake` not found。

检查：

```bash
find "$LLVM_BUILD_DIR" -path '*/cmake/llvm/LLVMConfig.cmake' -print
find "$LLVM_BUILD_DIR" -path '*/cmake/clang/ClangConfig.cmake' -print
```

只有 Clang 可执行文件而没有开发库/CMake metadata 时无法构建 Ascify。

### CMake 编译器测试失败

确认 LLVM 包含当前主机后端，并且默认 target triple 有效：

```bash
"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
```

也可以通过 `ASCIFY_CC` 和 `ASCIFY_CXX` 改用能够编译本机程序的系统编译器。

### 找不到 Ninja

安装 `ninja-build`，或指定已有生成器：

```bash
export CMAKE_GENERATOR="Unix Makefiles"
./build.sh
```

### `LLVM/resource config failed`

`CLANG_RESOURCE_DIRECTORY` 必须是 `include/` 的父目录，且其中应有 `include/__clang_cuda_runtime_wrapper.h`：

```bash
test -f "$CLANG_RESOURCE_DIRECTORY/include/__clang_cuda_runtime_wrapper.h"
```

使用默认安装路径时，它应为：

```bash
export CLANG_RESOURCE_DIRECTORY="$PWD/ascify_install/include/ascify"
```

### CUDA 路径为空或 CUDA headers/libdevice 找不到

确认 `CUDA_PATH` 是 Toolkit 根目录：

```bash
test -f "$CUDA_PATH/include/cuda.h"
test -f "$CUDA_PATH/include/cuda_runtime.h"
test -d "$CUDA_PATH/nvvm"
```

如机器安装了多个 CUDA 版本，请显式选择一个完整目录，不要依赖模糊的系统软链接。

### 自有源码找不到项目头文件

将原构建命令中的 include、宏和语言标准放在 `--` 后传给 Clang：

```bash
./run.sh input.cu -o output.cu.dpp -- \
  -std=c++17 -I/path/to/include -DMY_MACRO=1
```

### 生成文件不存在

先检查 Ascify 的退出码和标准错误。语法错误、缺失头文件、非法输出路径、回执路径与输入/输出别名冲突，都会让转换失败或拒绝发布输出。不要只检查目录中是否残留旧文件。

### Hybrid recipe 被拒绝

确认 `dav-c310-vec`、`fast` 和 `dav-3510-rowwise-simd-v1` 三项同时出现。即使组合正确，源码未通过窄 AST proof 时也不会获得 Hybrid dispatch；这属于支持边界，不应通过修改生成文件绕过。

## 13. 记录可复现信息

报告转换问题时，建议同时保存：

```bash
git rev-parse HEAD
cmake --version | head -n 1
ninja --version
"$LLVM_BUILD_DIR/bin/clang" --version
"$LLVM_BUILD_DIR/bin/llvm-config" --host-target
readlink -f "$CUDA_PATH"
"$ASCIFY_BINARY" --help > ascify-help.txt
```

同时保留原始命令、标准输出、标准错误、输入文件哈希、生成文件哈希和 `--migration-receipt` 结果。这样才能区分源码解析失败、转换失败、目标编译失败和设备运行失败。
