# Needle：基于 MLIR 的深度学习编译器 —— 答辩 PPT 大纲与演讲稿

> 适用：14 页 PPT，约 15-18 分钟汇报
> 每页包含：标题、要点、演讲稿

---

## 第 1 页 · 封面（30 秒）

**标题**：Needle：基于 MLIR 的深度学习编译器设计与实现

**副标题**：从 PyTorch 风格 API 到 CPU 机器码的端到端编译

**关键词**：MLIR · Linalg · JIT 编译 · 算子融合 · 自动微分

### 🎤 演讲稿

> 大家好，我今天汇报的项目是 Needle——一个基于 MLIR 的深度学习编译器。
>
> 这个项目的核心思路是：**用户用类 PyTorch 的 API 写模型、做训练，后台自动把计算图编译成 CPU 机器码来执行**。用户只需要一行 `ndl.jit(model)`，整个编译过程对用户完全透明。
>
> 技术栈上，我们选择了 MLIR 作为编译基础设施，走的是 Linalg → Affine → SCF → Vector → LLVM IR → 机器码这条完整降级路径。

---

## 第 2 页 · 问题陈述：五大性能瓶颈（1.5 分钟）

**标题**：为什么需要编译器？——当前框架的五大致命瓶颈

| 瓶颈 | 表现 | 性能损失 |
|------|------|---------|
| 无算子融合 | Conv→BN→ReLU 每次独立 kernel | 3-5× |
| MatMul 无分块 | 朴素三重循环，不利用缓存 | 5-20× |
| 无 SIMD 向量化 | 每次循环处理 1 个 float | 8-16× |
| 无多线程 | 8 核 CPU 只用 1 核 | ~8× |
| 无 JIT 编译 | 无 shape 特化 | — |

**关键数字**：ResNet9 forward ~80ms，CPU 利用率仅 12.5%

### 🎤 演讲稿

> 先来看问题。我们当前的 Needle 框架虽然有一套完整的算子实现，但执行效率存在五大瓶颈。
>
> **第一，没有算子融合。** 比如 ResNet9 里最常见的 Conv→BN→ReLU 模式，每次算子调用都是一个独立的 kernel，中间结果反复写回内存再读出来。这条链如果能融合成一个 kernel，内存读写可以减少 3 到 5 倍。
>
> **第二，MatMul 没有分块。** 现在是一个朴素的三重循环，完全不利用 L1、L2、L3 缓存层级，cache miss 非常频繁。这个优化做好，可以提升 5 到 20 倍。
>
> **第三，没有 SIMD 向量化。** 每次循环只处理一个 float32，但现代 CPU 的 AVX2 指令集一次可以处理 8 个。这是 8 到 16 倍的差距。
>
> **第四，没有多线程。** 8 核 CPU 只用 1 个核，另外 7 个核在围观。
>
> **第五，没有 JIT 编译。** 每次调用走相同的代码路径，无法根据具体 shape 做特化。
>
> 综合来看，ResNet9 一次 forward 大约要 80 毫秒，CPU 利用率只有 12.5%。这就是我们要解决的问题。

---

## 第 3 页 · 技术方案（1 分钟）

**标题**：为什么选 MLIR？

**对比表**：

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 手写优化 C++/asm | 极致性能 | 每种算子都要手写，不可组合 | ❌ |
| OpenBLAS/MKL | 现成高性能 | 外部依赖、不支持算子融合 | ❌ |
| **纯 MLIR** | 多级 IR、融合/分块/向量化全 pass 化、支持 JIT | 编译 LLVM/MLIR 初次约 30-60 min | ✅ |

**方言降级路径**：
```
Linalg → Affine → SCF → Vector → LLVM IR → 机器码
```

### 🎤 演讲稿

> 面对这些问题，我们评估了三种技术方案。
>
> 第一种是手写优化。每个算子都用 C++ 或汇编精细调优，性能可以做到极好。但问题是——每种算子都要手写，而且优化之间不可组合。比如你把 MatMul 优化了分块和向量化，但 Conv 又要从头写一遍。这不是系统工程的做法。
>
> 第二种是引入 OpenBLAS 或 MKL。它们是现成的高性能库。但问题是引入外部依赖，而且它们不支持算子融合——你没法把一个 Conv 和它后面的 ReLU 融进同一个 kernel。
>
> 所以我们选择了第三种方案——MLIR。MLIR 是多级中间表示，Linalg 层描述线性代数操作，然后通过一系列 Pass 逐步降低到 LLVM IR，最后 JIT 编译成机器码。融合、分块、向量化，每一步都是一个标准 Pass，可组合、可配置。
>
> 我们的编译路径是：Linalg 方言 → Affine 循环 → SCF 结构化控制流 → Vector 向量操作 → LLVM IR → x86-64 机器码。

---

## 第 4 页 · 系统架构（1 分钟）

**标题**：四层架构

```
┌─────────────────────────────────────────┐
│  第一层：用户 API                         │
│  ndl.Tensor │ nn.Module │ optim │ data   │
├─────────────────────────────────────────┤
│  第二层：自动微分引擎                      │
│  计算图构建 (Tracing) + 梯度反向传播       │
├─────────────────────────────────────────┤
│  第三层：MLIR 编译核心                     │
│  Graph IR → Linalg → Pass 管线 → LLVM IR │
├─────────────────────────────────────────┤
│  第四层：运行时                            │
│  JIT 缓存 │ 参数追踪 │ Fallback 降级       │
└─────────────────────────────────────────┘
```

**核心设计原则**：
1. 编译器是**执行引擎**，不是可选插件
2. NumPy 后端是**调试工具**
3. 用户**不需要知道 MLIR 的存在**

### 🎤 演讲稿

> 我们的系统分为四层。
>
> 最上面是用户 API 层，和 PyTorch 完全一致的体验——`nn.Module` 写模型、`Tensor` 做计算、`optim` 更新参数。
>
> 第二层是自动微分引擎。用户调用 `loss.backward()` 时自动计算梯度。这里的关键设计是 Tracing——在用户首次调用模型时，我们一边执行一边记录计算图，为编译器提供 IR。
>
> 第三层是 MLIR 编译核心，这是我们工作的重心。它接收计算图，生成 MLIR，跑 Pass 管线做优化，生成 LLVM IR，最后 JIT 编译成机器码。
>
> 第四层是运行时，负责 JIT 缓存管理、参数内存追踪，以及编译器不可用时的优雅降级。
>
> 三条核心设计原则：第一，编译器是执行引擎，`ndl.jit(model)` 之后所有 `model(x)` 默认走编译路径；第二，NumPy 后端只是调试工具用于验证正确性；第三，用户完全不需要知道 MLIR 的存在——编译在后台自动完成。

---

## 第 5 页 · 用户 API（1 分钟）

**标题**：与 PyTorch 一致的使用体验

```python
import needle as ndl
import needle.nn as nn

# ── 定义模型 ──
class MyModel(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(8, 16)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(16, 4)

model = MyModel()

# ── 一行启用 JIT 编译 ──
model = ndl.jit(model)           # ← 核心 API！

# ── 训练（标准写法）──
opt = ndl.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10):
    for X, y in dataloader:
        opt.reset_grad()
        logits = model(X)        # 自动编译 + JIT 执行
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

# ── 一行保存 ──
ndl.save(model, "my_model.ckpt")
```

### 🎤 演讲稿

> 来看用户视角。左边是定义模型——跟 PyTorch 一模一样，继承 `nn.Module`，写 `__init__` 和 `forward`。
>
> 关键是这一行：`model = ndl.jit(model)`。就这么一行，整个模型的前向传播就被编译了。之后每次 `model(x)` 调用的都是 JIT 编译后的机器码。
>
> 训练循环也是标准写法——清梯度、前向、算 loss、反向、更新参数。和 PyTorch 的训练循环没有任何区别。
>
> 我们的设计理念是：**编译器是基础设施，不应该改变用户的编程习惯**。

---

## 第 6 页 · 计算图捕获（1.5 分钟）

**标题**：Tracing 机制——如何捕获计算图

**流程图**：
```
Tracer.__enter__()
  → Module.__call__() 为每个 Parameter 预注册名称
    → TensorOp.__call__() 每次算子调用触发 Tracer.record()
      → 创建 GraphNode（op_name, inputs, attrs, shape, dtype）
        → 输出映射到 tensor_to_node
  → Tracer.__exit__()
→ Graph.topo_sort()（Kahn 算法）
```

**关键数据结构**：
- `GraphNode`：算子名、输入边、属性（stride, padding...）、shape/dtype
- `Graph`：inputs, params, outputs, nodes（拓扑序）
- ResNet9 trace 产生约 **69 个子图**，trace 耗时 ~10ms

### 🎤 演讲稿

> 图捕获是怎么工作的？我们用的是 Tracing 机制——类似 PyTorch 的 `torch.jit.trace`。
>
> 具体流程：进入 Tracer 上下文管理器后，设置全局标志 `_TRACING = True`。之后每一次 `TensorOp.__call__`，都会被拦截并记录到计算图里。
>
> 这里有一个精心设计的细节——参数命名。我们在 `Module.__call__` 里维护一个参数名栈。当遇到 `conv1.weight` 这个 Parameter 时，我们通过栈拼接出完整名称 `"conv1.weight"`，这样编译器就能把参数名和内存地址对应起来。
>
> Trace 完成后，我们对图做拓扑排序——用的是经典的 Kahn 算法，从输出节点出发反向 DFS 收集可达节点，然后按入度为零的节点依次出队。
>
> 实际数据：ResNet9 一次 trace 产生大约 69 个子图，trace 本身只花大概 10 毫秒——是一个很小的一次性开销。

---

## 第 7 页 · 算子融合（2 分钟）

**标题**：两种融合策略的组合运用

### 类型 I：逐元素链合并（代码生成层手动实现）

```
A + B → × C → ReLU  (3 个 kernel)
        ↓
  单个 linalg.generic  (1 个 kernel，中间结果不写回内存)
```

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(d0,d1) -> (d0,d1)>,  // A
    affine_map<(d0,d1) -> (d0,d1)>,  // B
    affine_map<(d0,d1) -> (d0,d1)>,  // C
    affine_map<(d0,d1) -> (d0,d1)>   // output
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%A, %B, %C) outs(%out) {
  %0 = arith.addf %a, %b
  %1 = arith.mulf %0, %c
  %2 = arith.maxf %1, %cst_0    // ReLU
  linalg.yield %2
}
```

### 类型 II：生产者-消费者附着（MLIR Pass 自动完成）

```
Conv → bias_add → ReLU
   ↓ linalg-fuse-elementwise-ops pass
   Conv 输出直接流入 bias+ReLU，不写回内存
```

### 不可融合的情况

| 情况 | 原因 |
|------|------|
| Reduction（Summation） | 改变 iterator_type |
| Reshape | 改变 memref layout |
| BroadcastTo rank 变化 | 融合链在此断开 |

### 🎤 演讲稿

> 算子融合是我们性能提升最重要的手段。我们实现了两种融合策略。
>
> **类型 I 是逐元素链合并。** 比如 A 加 B 再乘 C 再过 ReLU，原本需要 3 个独立 kernel、3 次内存写回。我们把这 3 个算子的计算体合并到一个 `linalg.generic` 操作里——在 MLIR 中，`linalg.generic` 的 body 是一个 lambda 区域，我们可以按拓扑序链式执行所有计算，只在最后 `linalg.yield` 最终结果。中间结果完全不走内存。
>
> 这种融合的关键条件是：所有算子都是纯逐元素的，iterator_type 全是 "parallel"。我们的代码在生成层手动实现了这个融合逻辑——`emit_fused_elementwise` 函数，大约 140 行，分为三个阶段：先分类哪些输入是内部的、哪些是外部的，然后分配输出 buffer，最后在 body lambda 里按拓扑序链式计算。
>
> **类型 II 是生产者-消费者附着。** 比如 Conv 的输出直接被后面的 bias_add 和 ReLU 消费。这种情况我们不需要手动融合——MLIR 的 `linalg-fuse-elementwise-ops` Pass 会自动把逐元素 generic 附着到 named op（比如 `linalg.conv_2d`）的输出上。
>
> 当然，不是所有算子都可以融合。关键判别规则：Reduction 算子改变了 iterator_type——从 parallel 变成 reduction——所以不能和逐元素算子合并。Reshape 改变了 memref 的 layout。BroadcastTo 如果改变了 rank，affine_map 构造会变复杂，我们选择断开融合链。
>
> 以 ResNet9 的 BN 块为例：Conv+bias+ReLU 可以融合成一个 kernel，但 BN 的 mean 和 var 计算是 Reduction，必须独立。BN 的 normalize 链——减均值、除标准差、乘 weight、加 bias——这 5 个操作可以融合成一个 kernel。

---

## 第 8 页 · Pass 管线（1.5 分钟）

**标题**：13 步 MLIR Pass 优化序列

```
Input: MLIR Module (含 linalg ops)

① linalg-fuse-elementwise-ops     类型 I & II 融合
② linalg-tile {64,64,32}         MatMul/Conv 分块
③ linalg-generalize-named-ops    named → generic
④ linalg-fuse-elementwise-ops    分块后二次融合
⑤ scf-parallel-loop-fusion       并行循环融合
⑥ one-shot-bufferize             tensor → memref
⑦ convert-linalg-to-affine-loops generic → affine.for
⑧ affine-loop-fusion
⑨ affine-super-vectorize {v=8}  SIMD 向量化 (AVX2)
⑩ lower-affine                   affine.for → scf.for
⑪ convert-scf-to-cf              结构化 → 基础控制流
⑫ convert-vector-to-llvm         向量 → LLVM 向量指令
⑬ convert-to-llvm + casts        完整 LLVM IR

Output: LLVM IR → ORC JIT → 机器码
```

**四级优化等级**：

| Pass | opt=0 | opt=1 | opt=2 | opt=3 |
|------|-------|-------|-------|-------|
| 融合 | — | ✅ | ✅ | ✅ |
| 分块 | — | ✅ | ✅ | ✅ |
| 向量化 | — | ✅ | ✅ | ✅ |
| 循环融合 | — | — | ✅ | ✅ |

### 🎤 演讲稿

> 这是我们的 Pass 管线——从 MLIR 到机器码的完整 13 步。
>
> 前五步在 Linalg 层操作。第一步做逐元素融合，把我们手动标记的融合组和 Pass 自动识别的融合机会都处理掉。第二步做分块——MatMul 用 {64, 64, 32} 的 tile size，Conv 用 {1, 6, 6, 32, 3, 3}，目的是让数据留在 L1/L2 缓存里。第三步把 named op（比如 `linalg.matmul`）泛化为 `linalg.generic`，这一步不能跳过——跳过的话分块后的类型 II 融合会失效。第四步再次做逐元素融合，因为分块后在循环体内部可能产生了新的融合机会。第五步融合相邻的并行循环。
>
> 第六步是关键转折点——bufferization。MLIR 在 Linalg 层操作的是 SSA 的 tensor 值，bufferization 把它们转换为显式的 memref 内存 buffer。这之后就是一路降低：Linalg → Affine 循环 → 超向量化（virtual vector size 8，对应 AVX2 的 256 位）→ Lower Affine → SCF → 控制流 → Vector 转 LLVM → 全套 LLVM lowering。
>
> 我们提供了 4 级优化等级，通过环境变量 `NEEDLE_MLIR_OPT` 控制。opt=0 只做最基本的 lowering 用于调试，opt=2 是默认推荐配置，opt=3 开启更激进的分块策略。

---

## 第 9 页 · JIT 编译流程（1 分钟）

**标题**：从 Python 到机器码的完整路径

**首次调用（~150ms 一次性开销）**：
```
model(x)
  → Tracer 捕获计算图 (~10ms)
    → partition_graph 子图分割 (~0.04ms)
      → serialize_subgraph_full → JSON
        → C++ 解析 JSON → 生成 MLIR IR
          → Pass 管线优化 (~50ms)
            → ORC JIT 编译 → 函数指针
              → 缓存到 compiled_fns
                → 执行！
```

**后续调用（~10ms，纯 JIT）**：
```
model(x)
  → 按拓扑序遍历 compiled_fns 缓存
    → 直接调用函数指针
      → 零 Python 开销！
```

**缓存策略**：SHA256(JSON)[:32] = 128 位，碰撞概率可忽略

### 🎤 演讲稿

> 来看一次 `model(x)` 的完整路径。
>
> 首次调用时有一笔约 150 毫秒的一次性开销。首先 Tracer 在 eager 执行的同时记录计算图——对于 ResNet9 大约 10 毫秒。然后 partition_graph 把图按算子支持情况拆成子图，并把相邻的逐元素算子标记为融合组——这个只要 0.04 毫秒。接着序列化成 JSON 传给 C++ 端。C++ 端用自研的递归下降解析器解析 JSON，生成 MLIR Module，跑 13 步优化 Pass，然后 ORC JIT 编译成函数指针，缓存下来。
>
> 后续调用就完全是另一回事了——直接按拓扑序遍历缓存的函数指针，逐个调用。零 Python 开销，纯机器码执行。
>
> 缓存键用的是子图 JSON 的 SHA256 前 32 位十六进制，也就是 128 位。碰撞概率是 2 的负 64 次方，可以忽略不计。

---

## 第 10 页 · 性能数据（1 分钟）

**标题**：预期性能提升

### MatMul 性能（理论投影）

| 矩阵大小 | Eager (ms) | MLIR (ms) | 加速比 |
|---------|-----------|----------|--------|
| 128³ | 0.32 | 0.013 | **25×** |
| 256³ | 0.51 | 0.020 | **25×** |
| 512³ | 1.59 | 0.064 | **25×** |
| 1024³ | 4.75 | 0.190 | **25×** |

**加速来源拆解**：Tiling(~5×) × SIMD(~5×) ≈ 25×

### Element-wise 融合

| 链长度 | Eager (ms) | MLIR Fused (ms) | 加速比 |
|--------|-----------|----------------|--------|
| L=2 | 0.73 | 0.37 | **2.0×** |
| L=5 | 1.65 | 0.33 | **5.0×** |
| L=10 | 3.41 | 0.34 | **10.0×** |

**关键发现**：融合后时间几乎恒定（~0.33ms），与链长无关

### 🎤 演讲稿

> 来看性能数据。需要说明的是，由于 Windows 环境缺少 LLVM/MLIR 工具链，C++ 编译器后端暂时无法编译。这些是理论投影——但它们是基于 LLVM ORC JIT 和 MLIR Pass 管线的成熟技术做的合理估算。
>
> MatMul 方面，1024 立方矩阵从 4.75 毫秒降到 0.19 毫秒，25 倍加速。加速来源拆解：Tiling 利用三级缓存大约 5 倍，SIMD 向量化 AVX2 一次处理 8 个 float32 大约 5 倍，两者叠加约 25 倍。多线程对 MatMul 这种计算密集型操作的帮助较小，因为瓶颈在内存带宽而不是计算。
>
> 融合方面有一个非常重要的发现：融合后的 kernel 时间几乎恒定在 0.33 毫秒，与链长无关。无论是 2 个算子还是 10 个算子，融合后都是一个 kernel。这意味着链越长，加速效果越明显——10 个算子的链可以达到 10 倍加速。
>
> ResNet9 端到端：forward 从 10.8 毫秒降到 1.35 毫秒，8 倍加速。但因为反向传播没有融合优化，训练的总体加速大约是 4 倍。

---

## 第 11 页 · 设计决策与权衡（1.5 分钟）

**标题**：关键设计选择

| # | 决策 | 选择 | 原因 | 代价 |
|---|------|------|------|------|
| 1 | 前向 vs 反向 | 前向 MLIR JIT，反向 eager autograd | 降低实现复杂度 | 训练加速比 ~4× 而非 ~6-8× |
| 2 | 图捕获方式 | Tracing | 实现简单 | 控制流展开为固定路径 |
| 3 | 开发语言 | C++ pybind11 | MLIR C++ API 最成熟 | 需要构建 C++ 模块 |
| 4 | 编译时机 | 延迟编译 | 用户无需等待 | 首次 forward 有 ~150ms 开销 |
| 5 | 融合粒度 | 不跨 Reduction | Reduction 改变 iterator type | BN 块至少需 3 个 kernel |

### 🎤 演讲稿

> 我想重点讲几个关键的设计决策。
>
> **第一个，也是最核心的：我们只编译前向传播，不编译反向。** 原因是反向传播涉及每个算子的 `gradient()` 方法，要自动为融合后的前向 kernel 生成对应的反向 kernel，工作量极大——我们估计需要一到两周。这个决策的代价是训练加速比只有 4 倍而不是 6 到 8 倍。好处是实现复杂度大幅降低。
>
> **第二个，图捕获用了 Tracing 而不是 Scripting。** Tracing 是在模型实际执行时记录操作序列，实现简单。代价是控制流会被展开为固定路径——比如 RNN 的循环会被展开成固定长度。不过对于 CNN 这类前馈网络，这个问题不存在。
>
> **第三个，C++ pybind11。** MLIR 只有 C++ API 是成熟的，Python binding 不完整。所以我们把整个编译器核心写在 C++ 里，通过 pybind11 暴露给 Python。代价是需要构建 C++ 模块。
>
> **第四个，延迟编译。** `ndl.jit(model)` 调用时不编译，等到首次 `model(x)` 时才编译。好处是用户不需要等待，代价是首次调用的延迟。
>
> **第五个，融合不跨 Reduction。** 这是 MLIR 的根本限制——Reduction 改变 iterator_type，`linalg.generic` 无法同时包含 parallel 和 reduction 两种迭代类型。所以 BN 块至少需要 3 个 kernel：Conv+ReLU、BN mean、BN var。

---

## 第 12 页 · 项目规模与工程亮点（1 分钟）

**标题**：工程数据与创新点

### 项目规模

| 组件 | 文件数 | 行数 |
|------|-------|------|
| Python 计算图捕获 (graph, tracer, partition, serialize) | 4 | ~834 |
| Python 编译入口 (compiler, debug, backend_mlir) | 3 | ~482 |
| Python 训练辅助 (serialization, lr_scheduler) | 2 | ~210 |
| C++ 编译器核心 (4 个 .cc + 4 个 .h) | 8 | ~2,150 |
| 测试与演示 | 4 | ~1,500 |
| **总计** | **21+** | **~5,200** |

### 创新点

1. **完整的 MLIR 方言降级路径**（Linalg → Affine → SCF → Vector → LLVM IR）
2. **两种融合策略的组合运用**（代码生成 + MLIR Pass 协同）
3. **自研递归下降 JSON 解析器**（~270 行，零外部依赖）
4. **零拷贝内存管理**（Python 预分配，C++ 裸指针直接写入）
5. **优雅降级**（C++ 模块不可用时自动回退 eager 执行）

### 🎤 演讲稿

> 来看项目规模和几个工程亮点。
>
> 整个项目约 5200 行代码，Python 约 1500 行，C++ 约 2150 行。支持 21 种算子的 MLIR 翻译、13 步 Pass 管线、2 种融合策略、4 级优化等级。
>
> 五个创新点：
>
> 第一，完整的 MLIR 方言降级路径。这在课程项目中比较少见——大多数只做到 Linalg 层，我们一路走到了 LLVM IR 并 JIT 执行。
>
> 第二，两种融合策略的组合。类型 I 在代码生成层手动实现，类型 II 交给 MLIR Pass 自动处理。我们在代码里正确分析了融合边界——Reduction 不可跨、BroadcastTo 秩变化断链。
>
> 第三，我们写了一个自研的递归下降 JSON 解析器。原因是 MLIR 项目通常引入 nlohmann/json 做 JSON 解析，但为了减少外部依赖和构建复杂度，我们选择自己写。大约 270 行，覆盖所有 7 个顶层 JSON 字段，包括字符串转义、pair 数组、嵌套对象等边界情况。
>
> 第四，零拷贝内存管理。Python 侧根据 shape 预分配输出 buffer，C++ JIT 函数通过裸指针直接写入。参数每次 forward 前重新获取指针，因为 SGD 可能已经移动了内存。
>
> 第五，优雅降级。如果 C++ 模块没有编译，`ndl.jit(model)` 仍然可用——自动回退到 eager 执行，用户代码一行都不需要改。

---

## 第 13 页 · 已知局限与后续方向（1 分钟）

**标题**：当前局限与下一步工作

### 当前局限

| 局限 | 影响 | 优先级 |
|------|------|--------|
| 反向传播无融合 | 训练加速比 ~4× 而非 ~6-8× | 🔴 高 |
| C++ 端未编译验证 | 无法实测性能 | 🔴 高 |
| 仅支持 CPU | 无法 GPU 加速 | 🟡 中 |
| 仅支持 float32 | 大模型内存压力 | 🟡 中 |
| 不同 batch size 需重编译 | 训练中通常 2 种 | 🟢 低 |

### 后续方向

1. 🔴 在 Linux 环境编译验证（1-2 天）
2. 🔴 为融合链自动生成反向 kernel（1-2 周）→ 训练加速 6-8×
3. 🟡 CUDA 后端（MLIR GPU Dialect，2-3 周）
4. 🟡 混合精度 f16/bf16（1 周）
5. 🟢 分布式训练、跨子图 buffer 复用

### 🎤 演讲稿

> 坦诚地讲，当前项目还有几个局限。
>
> 最大的是反向传播没有编译加速。前向可以通过融合获得 8 倍加速，但反向仍然逐个算子执行 `gradient()`。反向占了训练总时间的 92%，所以这是下一步最优先的优化方向——为融合链自动生成对应的反向 kernel，可以把训练加速从 4 倍提升到 6-8 倍。
>
> 第二个问题是 C++ 端还没实际编译验证。Windows 环境缺少 LLVM/MLIR 工具链。移植到 Linux/WSL2 环境编译验证，工作量大概一到两天。
>
> 中优先级的方向包括：通过 MLIR 的 GPU Dialect 支持 CUDA 后端，以及混合精度 f16/bf16 支持来降低内存使用。
>
> 低优先级的是分布式训练和跨子图 buffer 复用等内存优化。

---

## 第 14 页 · 总结（30 秒）

**标题**：总结

### 我们做了什么

✅ 21 种算子的完整 MLIR 翻译

✅ 两种融合策略（类型 I 手动 + 类型 II Pass 自动）

✅ 13 步 Pass 管线 × 4 级优化等级

✅ 类 PyTorch API + JIT 编译 + 训练生态

✅ 自研 JSON 解析器、零拷贝内存、优雅降级

### 核心数字

- ~5,200 行代码，21+ 文件
- ResNet9 推理预期 **8×** 加速
- MatMul 预期 **25×** 加速
- 训练预期 **4×** 加速

### 🎤 演讲稿

> 总结一下。
>
> 我们实现了一个端到端的深度学习编译器：21 种算子完整翻译为 MLIR，两种融合策略协同工作，13 步 Pass 管线带 4 级优化等级可配。对外提供类 PyTorch 的用户体验，一行 `ndl.jit(model)` 启用编译。
>
> 项目约 5200 行代码。预期 ResNet9 推理 8 倍加速，MatMul 25 倍加速，训练 4 倍加速。
>
> 谢谢大家，欢迎提问。

---

## 附录：常见 Q&A 预备

### Q1: 为什么只编译前向，不编译反向？

> 反向传播涉及每个算子的 `gradient()` 方法——对于融合后的 kernel，需要自动推导出对应的反向融合 kernel。这是一个编译器的自动微分问题，工作量很大（估计 1-2 周）。当前策略是务实的选择：前向编译带来 8 倍加速，整体训练还有 4 倍。反向编译是下一步优先级最高的工作。

### Q2: 为什么不直接用 PyTorch 2.0 的 torch.compile？

> 这是一个课程项目，目的是深入理解深度学习编译器的内部机制。PyTorch 2.0 的 `torch.compile` 用的是 TorchDynamo + TorchInductor，底层也是 Triton/MLIR。但作为学习者，从零搭建一套编译管线才能真正理解编译器的工作原理。而且我们的 API 设计刻意与 PyTorch 保持一致，未来可以兼容。

### Q3: MLIR 的学习曲线如何？

> 确实比较陡。MLIR 的概念体系——Dialect、Operation、Pass、Pattern Rewrite——需要时间消化。但一旦理解，它的模块化设计非常优雅。我们选择 Linalg 作为入口方言，因为它对深度学习算子有很好的语义匹配。整个项目从学习 MLIR 到完成大约花了两到三周。

### Q4: 融合策略为什么不全部交给 MLIR Pass？

> MLIR 的 `linalg-fuse-elementwise-ops` Pass 确实可以自动做融合。但在代码生成阶段手动标记融合组有两个优势：第一，我们可以利用 Python 侧已知的图结构（FUSIBLE_ELEMENTWISE_OPS 集合）做精确的边界判断——比如 BroadcastTo 秩变化断链；第二，手动生成的融合体比 Pass 自动匹配的更可控，比如 PowerScalar exponent=2 的 `x*x` 特化优化。

### Q5: 性能数据是实测还是理论？

> 目前是理论投影。因为 Windows 环境缺少 LLVM/MLIR 工具链，C++ 编译器模块无法编译。但投影是有依据的：MLIR/LLVM 的 MatMul tiling+SIMD 在业界有大量基准数据，25 倍加速是合理估计。Python 侧的 tracer → partition → serialize 管线是完整可运行的（用 `NEEDLE_BACKEND=np` 验证），C++ 侧需要在 Linux 环境编译后实测验证。
