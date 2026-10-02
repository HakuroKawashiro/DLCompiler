# Needle MLIR 编译器项目 —— 完整阅读指南

> **适用对象**：了解 Python、用过 PyTorch、听说过 MLIR/LLVM，但对本项目完全不了解的新手
>
> **目标**：按顺序读完以下文件后，能够理解整个编译管线的工作原理，并能自行修改和扩展
>
> **预计耗时**：6-10 小时（分 3-4 次阅读）

---

## 阅读前的前置知识检查

在开始之前，确认你大致了解以下概念：

| 概念 | 需要了解的程度 |
|------|---------------|
| PyTorch `nn.Module` / `forward()` / `backward()` | 熟练使用 |
| 自动微分（autograd）的基本原理 | 知道前向建图、反向传播梯度 |
| NumPy `ndarray` 的 shape/dtype | 熟练使用 |
| MLIR 是什么（Multi-Level IR） | 听说过，知道 Dialect/Pass 的概念 |
| `linalg.generic` 是什么 | 可以不知道，读到 mlir_gen.cc 时会解释 |
| LLVM ORC JIT | 可以不知道，读到 jit_engine.cc 时会解释 |

---

## 阅读路线总图

```
                    第零层：已有框架（选读）
                    autograd.py, nn_basic.py, ops/*.py
                              │
              ┌───────────────┴───────────────┐
              │                               │
        第一层：图 IR                    了解执行目标
        graph.py                        apps/models.py
              │                               │
        第二层：运行时追踪
        tracer.py
              │
        第三层：子图分割
        partition.py
              │
        第四层：序列化
        serialize.py
              │
        第五层：MLIR 后端胶水
        backend_mlir/__init__.py
              │
        第六层：编译入口
        compiler.py
              │
              ├───────────────────────────────┐
              │                               │
    Python 侧完成 ✅                   第七层：训练辅助
    （可运行验证）                       serialization.py
                                       lr_scheduler.py
                                       debug.py
                                              │
                                    第八层：C++ 编译器核心
                                    needle_compiler.h → .cc
                                    mlir_gen.h → .cc
                                    passes.h → .cc
                                    jit_engine.h → .cc
                                              │
                                    第九层：构建 & 集成
                                    CMakeLists.txt
                                              │
                                    第十层：测试 & 演示
                                    test_mlir_backend.py
                                    mlir_demo.py
                                    bench_mlir.py
```

---

## 第零层：已有框架基础（30 分钟，选读）

> **如果**你不熟悉 Needle 框架本身，先快速过一遍这些文件。
> **如果**你已经写过 hw2/hw3 的作业或者熟悉 autograd 实现，直接跳到第一层。

### 文件 0.1：`python/needle/autograd.py`（498 行）

**为什么先读这个**：整个框架的基础。Tensor、Op、TensorOp 的定义都在这里。

**阅读重点**：
- 第 27-78 行：`Op` 基类 — 所有算子的抽象。重点看 `__call__` 方法，理解算子调用时发生了什么
- 第 81-92 行：`TensorOp.__call__` — **关键！** 第 88 行是编译器追踪钩子的注入点。这就是 Tracing 机制插入框架的地方
- 第 102+ 行：`Value` 类 — 计算图中的节点，持有 `op` 和 `inputs` 的引用
- `Tensor` 类的 `realize_cached_data()`、`data` setter、`backward()` 方法

**读完应能回答**：
- 用户写 `a + b`（两个 Tensor）时，实际调用了哪些方法？
- `TensorOp.__call__` 中的 `_TRACING` 标志是做什么的？

### 文件 0.2：`python/needle/nn/nn_basic.py`（383 行）

**阅读重点**：
- 第 10 行：`Parameter(Tensor)` — 只是一个标记子类
- 第 52-109 行：`Module` 类 — 重点看 **`__call__`** 方法（第 80-109 行）
  - 第 82-107 行是新增的 Tracing 钩子：在 tracing 时预注册参数名
- 其他部分（Linear, ReLU, Flatten 等）可以快速浏览

**读完应能回答**：
- `Module.__call__` 在 tracing 时做了哪些额外工作？
- 参数名 `"conv1.weight"` 是怎么拼出来的？

### 文件 0.3：`python/needle/ops/ops_mathematic.py`（611 行）

**快速浏览**，不需要逐行读。了解有哪些算子、它们的类名是什么。这些类名会直接出现在 `EMIT_FN_TABLE` 和 `SUPPORTED_MLIR_OPS` 里。

**关键对应关系**：
| Python 类名 | 含义 | MLIR 映射 |
|------------|------|----------|
| `EWiseAdd` | 逐元素加法 | `arith.addf` |
| `EWiseMul` | 逐元素乘法 | `arith.mulf` |
| `ReLU` | ReLU 激活 | `arith.maxf(x, 0)` |
| `MatMul` | 矩阵乘法 | `linalg.matmul` |
| `Conv` | 卷积 | `linalg.conv_2d_nhwc_hwcf` |
| `Summation` | 规约求和 | `linalg.generic` (reduction) |

### 文件 0.4：`python/needle/optim.py`（118 行）

**浏览**第 17-43 行的 `clip_grad_norm()` — 这是我们新增的方法。SGD（第 46 行）和 Adam（第 76 行）可以快速过一遍。

---

## 第一层：计算图 IR（20 分钟）

### 文件 1.1：`python/needle/graph.py`（175 行）

**整个编译器管线的基础数据结构。** 这是全新的、专门为编译器设计的文件。

**从头到尾逐行读。**

**阅读重点**：
- 第 11-38 行：`GraphNode` — 一个算子调用的记录。注意它是一个纯数据结构，**不包含计算逻辑**
  - `op_name`：算子名字符串
  - `inputs`：输入节点引用列表
  - `attrs`：算子属性字典（stride, padding, axes...）
  - `shape`/`dtype`：输出张量的形状和类型
  - `id`：全局自增唯一 ID
- 第 41-175 行：`Graph` — 计算图容器
  - `inputs`：外部输入（数据 Tensor，不是参数）
  - `params`：参数字典（`{"conv1.weight": GraphNode}`）
  - `outputs`：输出节点
  - `nodes`：拓扑排序后的全量节点
  - 第 80-141 行：`topo_sort()` — **Kahn 算法**实现。从 outputs 出发反向 DFS 收集可达节点 → 计算入度 → 零入度入队 → 出队输出。这是图编译的经典第一步

**读完应能回答**：
- `GraphNode` 和 `Graph` 各自持有哪些信息？哪些信息编译器需要？
- `topo_sort()` 为什么要从 outputs 做反向 DFS 而不是从 inputs 正向 BFS？
- 如果图中有环（虽然不应该），代码会怎么处理？

---

## 第二层：运行时追踪（25 分钟）

### 文件 2.1：`python/needle/tracer.py`（196 行）

**编译器管线的入口。** Tracer 是连接"用户执行模型"和"编译器计算图"的桥梁。

**从头到尾逐行读。**

**阅读重点**：
- 第 13-51 行：`Tracer` 类的 `__init__`、`__enter__`、`__exit__`
  - `__enter__` 设置全局标志 `_TRACING = True` 和 `_ACTIVE_TRACER = self`
  - `__exit__` 清除这些标志
  - 这是一个**上下文管理器**，用法是 `with Tracer() as tracer: output = model(x)`
- 第 53-92 行：`record()` — **核心方法**
  - 被 `TensorOp.__call__`（autograd.py:88）调用
  - 步骤：解析输入节点 → 提取算子名和属性 → 获取输出 shape/dtype（从 eager 执行结果） → 创建 GraphNode
- 第 94-140 行：`get_or_create_input_node()` — 区分 Parameter 和普通 Tensor
  - Parameter → 创建 `"param"` 占位节点，加入 `graph.params`
  - 普通 Tensor → 创建 `"input"` 占位节点，加入 `graph.inputs`
- 第 142-158 行：`_resolve_param_name()` — 参数命名逻辑
- 第 161-169 行：`pause()`/`resume()` — 用于 Dropout 内部跳过 trace
- 第 188-196 行：`record_mutation()` — 记录 BN running stats 的更新

**关键设计**：Tracer 在记录图的同时，**模型仍在 eager 执行**。这意味着我们利用 eager 执行来获取真实的 shape/dtype/控制流路径，而图只记录结构。

**读完应能回答**：
- 为什么 `record()` 需要从 eager 执行结果获取 shape/dtype，而不是从算子定义推导？
- `get_or_create_input_node` 对 Parameter 和普通 Tensor 的处理有什么区别？为什么这样设计？
- `pause()`/`resume()` 是做什么用的？Dropout 为什么要 pause？

---

## 第三层：子图分割（25 分钟）

### 文件 3.1：`python/needle/partition.py`（247 行）

**编译器管线的第二个关键步骤。** 把完整的计算图拆分为"MLIR 可以编译的子图"和"必须走 eager 的 fallback 节点"。

**从头到尾逐行读。**

**阅读重点**：
- 第 8-31 行：**三组关键常量** — 这些是整个项目最重要的配置
  ```python
  SUPPORTED_MLIR_OPS = {21 种}      # 可以翻译成 MLIR 的算子
  FUSIBLE_ELEMENTWISE_OPS = {14 种}  # 可以参与融合的逐元素算子
  FUSION_BARRIER_OPS = {4 种}       # 融合屏障：遇到这些算子，融合链断开
  ```
  花 5 分钟逐一理解：为什么 Summation 是 barrier？为什么 BroadcastTo 在 FUSIBLE 里但 rank 变化时要断链？
- 第 34-53 行：`Subgraph` 数据结构
  - `input_mapping`：外部输入 node_id 列表
  - `param_mapping`：参数名列表
  - `output_mapping`：哪些节点产生输出
  - `mutation_map`：参数突变映射（BN stats 写回）
  - `fused_groups`：融合组区间 `[(start_idx, end_idx), ...]`
- 第 72-221 行：`partition_graph()` — **核心算法**
  1. 拓扑排序（第 83 行）
  2. 遍历节点（第 103 行起）
  3. 连续 `SUPPORTED_MLIR_OPS` 归入同一子图（第 140-144 行）
  4. Fallback 节点（第 108 行）：不受支持的算子 → 独立子图
  5. 融合组识别（第 169-193 行）：扫描 `FUSIBLE_ELEMENTWISE_OPS` 的连续区间
  6. BroadcastTo rank 变化检测（第 181-190 行）：rank 不同 → 断开融合链
  7. 输出映射（第 196-218 行）

**读完应能回答**：
- 一个子图里可以既有 MatMul 又有 ReLU 吗？如果可以，它们是融合在一起的吗？
- BroadcastTo 什么时候会断开融合链？为什么？
- `mutation_map` 是做什么用的？用 BN 的例子说明。

---

## 第四层：序列化（15 分钟）

### 文件 4.1：`python/needle/serialize.py`（212 行）

**Python → C++ 的桥梁。** 把 Subgraph 对象转成 C++ 可以解析的 JSON。

**从头到尾逐行读。**

**阅读重点**：
- 第 14-49 行：`_node_to_dict()` — 单个节点→JSON
  - 第 32-42 行：输入索引逻辑 — 内部节点用本地索引 `[0, num_nodes)`，外部输入用参数索引 `[num_nodes, num_nodes + num_inputs)`
  - **这是最容易出 bug 的地方**：如果索引算错，C++ 会引用错误的内存
- 第 52-121 行：`serialize_subgraph()` — 轻量版，不含 shape/dtype
- 第 124-212 行：`serialize_subgraph_full()` — **生产版本**
  - 从 `graph.find_node()` 解析外部输入的 shape/dtype（第 162-169 行）
  - 从 `graph.params` 解析参数的 shape/dtype（第 179-193 行）

**JSON Schema**（第 55-79 行的注释给出了完整示例）：
```json
{
  "nodes": [{"op": "...", "inputs": [...], "attrs": {...}, "shape": [...], "dtype": "..."}],
  "inputs": [{"node_id": ..., "shape": [...], "dtype": "..."}],
  "params": [{"name": "...", "shape": [...], "dtype": "..."}],
  "num_outputs": 2,
  "fused_groups": [[0, 3], [5, 7]],
  "output_node_indices": [0, 4],
  "mutation_map": {"bn1.running_mean": 1}
}
```

**读完应能回答**：
- 为什么 `inputs` 数组的索引有两种不同的含义（本地索引 vs 参数索引）？
- 传给 C++ 的 JSON 中，`shape` 和 `dtype` 是怎么填进去的？

---

## 第五层：MLIR 后端胶水（10 分钟）

### 文件 5.1：`python/needle/backend_mlir/__init__.py`（99 行）

**Python 和 C++ 之间的薄封装层。**

**从头到尾读。**

**阅读重点**：
- 第 19-27 行：尝试导入 C++ pybind11 模块 `needle_compiler`
  - 如果失败，优雅降级：设置 `_compiler = None`，打印警告
  - 这就是"编译器不可用时自动 fallback"的实现
- 第 30-99 行：`MLIRDevice` 类
  - `compile_and_run()` — 唯一的对外方法
  - 第 73-89 行：把所有 buffer 转成 `np.ascontiguousarray`（C++ 需要连续内存）
  - 第 92-98 行：调用 C++ 的 `compile_and_run`

**读完应能回答**：
- 如果 C++ 模块没有编译，`MLIRDevice.enabled()` 返回什么？调用 `compile_and_run()` 会发生什么？
- 为什么要 `np.ascontiguousarray`？

---

## 第六层：编译入口（30 分钟）

### 文件 6.1：`python/needle/compiler.py`（333 行）

**用户 API 的实现。** 这是整个 Python 侧的编排器。用户调用 `ndl.jit(model)`，返回的就是 `CompiledModule`。

**这是 Python 侧最重要的文件，逐行精读。**

**阅读重点**：
- 第 30-35 行：`_MLIR_AVAILABLE` 检测 — 决定走 JIT 还是 eager
- 第 38-48 行：`jit()` 函数 — 一行包装
- 第 51-334 行：`CompiledModule` 类 — 核心编排器

**初始化**（第 57-76 行）：
- 保存原始模型引用 `self.original`
- `_name_modules()` — 递归给每个子模块设 `_name`
- `_build_param_registry()` — 建 `name → Parameter` 映射

**`forward()`**（第 113-125 行）— **主入口**：
- 判断是否需要重新编译（首次调用或 train/eval 模式变化）
- 走 JIT 还是 eager

**`_trace_graph()`**（第 127-169 行）— **编译管线的 Python 侧**：
1. `with Tracer() as tracer:` → 执行原始模型 → 捕获计算图
2. 标记输出节点
3. `graph.topo_sort()` — 拓扑排序
4. `partition_graph()` — 子图分割
5. `serialize_subgraph_full()` → `hashlib.sha256()` → 缓存键
6. 如果 MLIR 可用 → `_compile_subgraphs()`

**`_compile_subgraphs()`**（第 171-219 行）— 首次编译：
- 遍历每个子图，分配输入/输出 buffer
- 调用 `mlir_device.compile_and_run()` 触发 C++ 编译+执行
- 结果覆盖 `node.tensor.cached_data`

**`_jit_forward()`**（第 221-269 行）— 后续调用（纯 JIT 执行）：
- 参数刷新（第 231-233 行）
- 遍历 `compiled_fns` → 传递空 JSON（走 JIT 缓存）
- **注意**：这里 `subgraph_json = ""` 表示"已在缓存中，直接执行"

**`_eager_forward()`**（第 271-280 行）— Fallback 路径

**`_write_back_mutations()`**（第 290-298 行）— BN stats 写回

**`train()`/`eval()`**（第 302-314 行）— 模式切换 → 缓存失效

**读完应能回答**：
- 首次 `model(x)` 和后续 `model(x)` 的执行路径有什么不同？
- 为什么 `train()` 和 `eval()` 切换会导致缓存失效？
- `cache_key` 是怎么生成的？为什么子图 JSON 相同就能命中缓存？
- `_write_back_mutations` 为什么需要？它解决了什么问题？

---

## 第七层：训练辅助（15 分钟）

### 文件 7.1：`python/needle/serialization.py`（118 行）

**模型保存/加载。** 逻辑简单：递归遍历 Module 的 `__dict__`，收集 Parameter → pickle 序列化。

**重点看**：`save()`（第 13 行）、`load()`（第 26 行）、`_state_dict()`（第 48-83 行）的递归遍历逻辑。

### 文件 7.2：`python/needle/lr_scheduler.py`（91 行）

**学习率调度器。** `StepLR`（第 49 行）和 `CosineAnnealingLR`（第 70 行）。逻辑独立，不依赖编译器的任何部分。

### 文件 7.3：`python/needle/debug.py`（49 行）

**调试模式开关。** `ndl.debug()` → 热切换 NumPy 后端 → 用于编译器正确性验证。

### 文件 7.4：`python/needle/__init__.py`（19 行）

**导出列表。** 确认 `jit`, `debug`, `is_debug`, `save`, `load`, `lr_scheduler` 都已正确导出。

---

## 中场检查点 🏁

**在继续 C++ 之前，你应该能在纸上画出这个流程图：**

```
用户调用 ndl.jit(model)
  → CompiledModule.__init__()
    → _name_modules() + _build_param_registry()

首次 model(x):
  → CompiledModule.forward()
    → _trace_graph()
      → Tracer.__enter__()
      → self.original(*args)  ← eager 执行
        → TensorOp.__call__() → Tracer.record() → GraphNode 创建
      → Tracer.__exit__()
      → Graph.topo_sort()  ← Kahn 算法
      → partition_graph()  ← SUPPORTED_MLIR_OPS 分组 + 融合组标记
      → serialize_subgraph_full() → JSON 字符串
      → hashlib.sha256(JSON)[:32] → cache_key
    → _compile_subgraphs()
      → MLIRDevice.compile_and_run(JSON) → C++ 端

后续 model(x):
  → CompiledModule.forward()
    → _jit_forward()
      → 遍历 compiled_fns
        → MLIRDevice.compile_and_run(cache_key, "")  ← 空 JSON = 走缓存
          → 函数指针调用 → 机器码执行
      → _write_back_mutations() ← BN stats 写回
```

如果你能解释清楚每一步的数据流动（输入是什么、输出是什么、数据如何传递），就可以进入 C++ 部分。

---

## 第八层：C++ 编译器核心（2-3 小时）

> **这部分需要最多时间。** 建议顺序：先读 header 理解接口，再读实现理解细节。

### 文件 8.1：`src/compiler/needle_compiler.h`（53 行）

**C++ 侧的数据结构定义。** 读 10 分钟。

**重点**：
- `NodeDescriptor`（第 15-21 行）：C++ 侧的 "GraphNode"，对应 Python `_node_to_dict()` 的输出
- `SubgraphDescriptor`（第 25-38 行）：C++ 侧的 "Subgraph"，对应 JSON 的 7 个顶层字段
- 环境变量辅助函数（第 41-50 行）：`get_optimization_level()` 等

**读完应能回答**：
- `NodeDescriptor.input_indices` 里的数字是怎么来的？对应 JSON 的哪个字段？

### 文件 8.2：`src/compiler/needle_compiler.cc`（584 行）

**C++ 侧的入口：JSON 解析 + pybind11。** 读 45 分钟。

**分两部分读**：

#### 第一部分：自研 JSON 解析器（第 98-292 行）

**这是整个项目最能体现工程能力的部分之一。** 一个 ~200 行的递归下降解析器，不依赖任何外部 JSON 库。

- `JsonReader` 类（第 103-290 行）
  - `parse_string()`（第 127-150 行）— 处理转义字符
  - `parse_number()`（第 153-174 行）— 处理负数、小数、科学计数法
  - `parse_int64_array()`（第 177-192 行）— 解析 `[1, 2, 3]`
  - `parse_pair_array()`（第 195-210 行）— 解析 `[[0, 3], [5, 7]]`（融合组）
  - `parse_attrs_object()`（第 214-235 行）— 解析 `{"stride": 2, "padding": 1}`
  - `parse_skip_array()`（第 255-271 行）— 跳过数组值（用于 attrs 中的数组）
- `parse_subgraph_json()`（第 294-425 行）— 使用 JsonReader 解析完整的 7 个顶层字段

**关键**：注意第 321-339 行，每个 JSON node 字段的解析顺序（op → inputs → attrs → shape → dtype），必须与 Python `_node_to_dict()` 的输出完全一致。

#### 第二部分：pybind11 入口（第 428-584 行）

- `compile_and_run()`（第 446-548 行）— **核心函数**
  1. 初始化 CompilerInstance 单例（注册所有 MLIR Dialect）
  2. 如果 JSON 有内容 → 解析 → 生成 MLIR Module → 验证 → 优化 Pass → JIT 编译
  3. 如果 JSON 为空 → 从 JIT 缓存查函数指针
  4. 收集输入/输出裸指针 → 调用 JIT 函数
- pybind11 模块定义（第 557-584 行）

**读完应能回答**：
- `compile_and_run` 收到空 JSON 时怎么处理？与收到非空 JSON 有什么区别？
- JIT 函数的 C 签名是什么？（提示：第 533 行）

### 文件 8.3：`src/compiler/mlir_gen.h`（177 行）

**MLIR 生成层的接口声明。** 读 15 分钟。

**重点**：
- `EmitFn` 类型别名（第 30-35 行）：所有 emit 函数的统一签名
- 每个算子对应的 emit 函数声明（第 52-137 行）
- `emit_fused_elementwise()`（第 147-152 行）：融合体生成
- `EMIT_FN_TABLE`（第 174 行）：op_name → emit 函数的映射表

**读完应能回答**：
- 如果要新增一个 `GELU` 算子支持，需要在哪些地方做修改？

### 文件 8.4：`src/compiler/mlir_gen.cc`（800 行）

**整个项目最大的文件。** 21 种算子的 MLIR IR 生成 + 融合体生成。这是编译器前端的核心。

**建议分 5 个部分读，每部分约 15 分钟**：

#### 第一部分：辅助函数（第 1-83 行）
- `make_memref_type()`, `make_constant_f32()`, `make_identity_map()` 等
- `create_linalg_generic()`（第 64-83 行）— 构造 `linalg.generic` 操作的模板函数
  - 理解 `indexing_maps`（数据访问模式）和 `iterator_types`（"parallel" vs "reduction"）

#### 第二部分：逐元素二元算子（第 89-161 行）
- `emit_elementwise_binary()`（第 90-120 行）— 二元算子的通用模板
- `emit_elementwise_add/mul/div/pow` — 四个具体实现，只有 body lambda 不同

#### 第三部分：逐元素一元算子（第 166-226 行）
- `emit_elementwise_unary()` — 一元算子的通用模板
- `emit_relu/tanh/exp/log/negate` — ReLU 使用 `arith.maxf(x, 0)`

#### 第四部分：标量算子 + 命名算子 + 数据移动（第 228-491 行）
- `emit_scalar_add/mul/div`（第 254-285 行）— 从 `node.attrs["scalar"]` 读取标量值
- `emit_power_scalar()`（第 287-302 行）— **注意 exponent=2 的特化**：`x * x` 替代 `powf(x, 2)`
- `emit_matmul()`（第 308-321 行）：`linalg.matmul`
- `emit_conv()`（第 323-359 行）：`linalg.conv_2d_nhwc_hwcf`，stride/dilation 从 attrs 读取
- `emit_broadcast()`（第 365-412 行）：构造 broadcast affine_map
- `emit_reshape()`（第 414-425 行）：`alloc + copy`
- `emit_summation()`（第 457-492 行）：reduction iterator_type

#### 第五部分：融合体生成 + 主入口（第 494-800 行）🌟

**这是编译器最核心、最复杂的函数。**

`emit_fused_elementwise()`（第 498-640 行）— 三阶段算法：

**Phase 1（第 507-531 行）— 分类输入**：
- 构建 `internal_set`（融合组内的节点索引）
- 遍历所有节点输入，区分内部/外部
- 外部输入去重，建立 `ext_to_body_arg` 映射

**Phase 2（第 533-540 行）— 构建输出**：
- 分配输出 memref，构建 identity 索引映射
- 所有 iterator_type = "parallel"

**Phase 3（第 545-639 行）— 构建融合体**：
- 在 `linalg.generic` 的 body lambda 中
- 按拓扑序遍历融合组内节点
- 对每个节点：解析操作数（内部用 `internal_vals`，外部用 `body_args`）
- 分发到正确的 MLIR 操作（AddFOp/MulFOp/DivFOp/...）
- 结果存入 `internal_vals`，供后续节点使用
- 最终节点结果送入 `linalg.yield`

`EMIT_FN_TABLE`（第 646-666 行）— 算子→函数映射表

`generate_mlir_module()`（第 672-797 行）— **主入口**：
1. 创建 `func.func @compiled_subgraph`
2. 构建函数参数（输入 + 参数 memref）
3. 遍历节点：融合组 → 调用 `emit_fused_elementwise`；单节点 → 查表调用 emit 函数
4. 收集输出节点 → `func.return`

**读完应能回答**：
- 如果要支持 `GELU` 算子（逐元素，可实现为 `x * 0.5 * (1 + tanh(0.797 * x + 0.044 * x^3))`），代码中需要改哪些地方？
- `emit_fused_elementwise` 如何保证融合体内的节点按正确的拓扑序执行？
- 为什么 `BroadcastTo` 在融合体内部是"元素透传"（operands[0]）？

### 文件 8.5：`src/compiler/passes.h`（52 行）

**Pass 管线的接口。** 读 5 分钟。

### 文件 8.6：`src/compiler/passes.cc`（164 行）

**13 步 Pass 管线实现。** 读 20 分钟。

**从头到尾逐行读 `build_optimization_pipeline()`（第 36-127 行）。**

对照报告中 §5 的 Pass 序列表，在代码中找到每一步对应的行：

| 步骤 | 代码行 | 条件 |
|------|--------|------|
| ① linalg-fuse-elementwise-ops | 45 | opt>=1 |
| ② linalg-tile | 49-61 | opt>=1 |
| ③ linalg-generalize-named-ops | 68 | opt>=1 |
| ④ linalg-fuse-elementwise-ops (2nd) | 73 | opt>=1 |
| ⑤ scf-parallel-loop-fusion | 83 | opt>=2 |
| ⑥ one-shot-bufferize | 89 | 总是 |
| ⑦ convert-linalg-to-affine-loops | 93 | opt>=1 |
| ⑧ affine-loop-fusion | 98 | opt>=2 |
| ⑨ affine-super-vectorize | 106 | opt>=1 |
| ⑩ lower-affine | 110 | 总是 |
| ⑪ convert-scf-to-cf | 113 | 总是 |
| ⑫ convert-vector-to-llvm | 116 | 总是 |
| ⑬ 全套 LLVM lowering | 120-124 | 总是 |

**读完应能回答**：
- 为什么 `linalg-generalize-named-ops` 不能跳过？（提示：报告 §5 有解释）
- opt=0 和 opt=2 的区别是什么？opt=0 跳过了哪些优化？
- `one-shot-bufferize` 为什么是"总是"执行？tensor → memref 的转换为什么是必须的？

### 文件 8.7：`src/compiler/jit_engine.h`（81 行）

**JIT 引擎的接口。** 读 10 分钟。

**重点**：
- `compile_and_get_fn()` — MLIR Module → 函数指针
- `get_cached_or_compile()` — 缓存查询 + 编译
- `compute_cache_key()` — SHA256 哈希
- `fn_cache_` — `unordered_map<string, void*>`

### 文件 8.8：`src/compiler/jit_engine.cc`（237 行）

**JIT 引擎的实现。** 读 25 分钟。

**从头到尾读**：

- 第 37-64 行：构造函数 — 初始化 LLVM 原生目标，创建 LLJIT 实例
  - `JITTargetMachineBuilder::detectHost()` 自动检测 CPU 特性（AVX2/AVX-512）
- 第 70-101 行：`mlir_to_llvm()` — **MLIR Module → LLVM IR 的转换**
  - 通过 `PassManager` 运行优化管线
  - `translateModuleToLLVMIR()` 做最终翻译
- 第 107-145 行：`compile_and_get_fn()` — 编译+获取函数指针
- 第 151-174 行：`get_cached_or_compile()` — 缓存逻辑
  - 先查 `fn_cache_`
  - 未命中 → 编译 → 缓存 → 返回
- 第 176-195 行：`compute_cache_key()` — SHA256 前 16 字节 = 128 位
- 第 205-234 行：环境变量读取函数

**读完应能回答**：
- Python 侧传过来的 `cache_key` 和 C++ 侧计算的 `compute_cache_key()` 是什么关系？
- `mlir_to_llvm` 函数中，Pass 管线的优化结果最终是怎么变成 LLVM IR 的？

---

## 第九层：构建系统（10 分钟）

### 文件 9.1：`CMakeLists.txt`（147 行）

**构建配置。** 分三部分：

- 第 33-51 行：CPU 后端（`ndarray_backend_cpu`）— 已有，不需要改
- 第 57-110 行：**MLIR 编译器后端（`needle_compiler`）**— 这是我们新增的
  - 第 71-76 行：编译 4 个 .cc 源文件
  - 第 78-88 行：链接 MLIR 各 Dialect 库 + LLVM ORC JIT
  - 第 91 行：C++17 标准（MLIR 要求）
  - 第 99 行：输出到 `python/needle/backend_mlir/`
- 第 115-146 行：CUDA 后端（可选）

---

## 第十层：测试与演示（30 分钟）

### 文件 10.1：`apps/models.py`（119 行）

**ResNet9 和 LanguageModel 的定义。** 这是编译器要编译的目标模型。

**重点看**：`ConvBN` 类（第 9-20 行）— 一次 forward 包含 `conv → bn → relu`，这正是编译器融合优化的主要目标。

### 文件 10.2：`apps/mlir_demo.py`（595 行）

**完整的使用演示。** 包含了：
- 基础算子编译测试
- Element-wise 融合链性能对比
- ResNet9 端到端训练
- 模型保存/加载
- 调试模式验证

**重点看**第 45-58 行的 `timeit` 工具函数和第 80-120 行的基本测试（如果有时间可以跑一遍）。

### 文件 10.3：`tests/hw4/test_mlir_backend.py`（402 行）

**编译器测试。** 验证 tracer → partition → serialize 管线的正确性。不依赖 C++ 编译。

### 文件 10.4：`benchmarks/bench_mlir.py`（300 行）

**性能基准测试。**

---

## 附录 A：验证你的理解

读完整个项目后，你应该能回答以下问题：

### 基础题

1. 用户写 `ndl.jit(model)` 后，`model(x)` 的完整执行路径是什么？
2. `partition_graph` 中的 `FUSIBLE_ELEMENTWISE_OPS` 和 `FUSION_BARRIER_OPS` 的区别是什么？为什么 Summation 是 barrier？
3. JSON 中 `inputs` 字段的索引值为什么有两种范围（`[0, num_nodes)` 和 `[num_nodes, num_nodes+num_inputs)`）？

### 进阶题

4. 如果要新增一个 `GELU` 算子支持，需要在哪些文件中做修改？列出完整清单。
5. `emit_fused_elementwise` 中的三阶段算法：如果融合组内有 `BroadcastTo` 且 rank 变化，Phase 1 的分类逻辑会怎么处理？
6. 假设 MLIR Pass 管线中删除了 `linalg-generalize-named-ops`（第 ③ 步），会产生什么后果？

### 设计题

7. 当前反向传播没有编译加速。如果要为前向融合链自动生成反向 kernel，需要做什么？哪些文件需要改动？
8. 当前每个子图独立分配 buffer。如何实现跨子图的 buffer 复用？需要改动哪些模块？

---

## 附录 B：关键文件索引（按功能分类）

### Python 侧 — 编译器管线（新代码）

| 文件 | 行数 | 职责 | 阅读优先级 |
|------|------|------|-----------|
| `graph.py` | 175 | 计算图 IR | 🔴 必读 |
| `tracer.py` | 196 | 运行时追踪 | 🔴 必读 |
| `partition.py` | 247 | 子图分割 | 🔴 必读 |
| `serialize.py` | 212 | JSON 序列化 | 🔴 必读 |
| `compiler.py` | 333 | 编译入口 | 🔴 必读 |
| `backend_mlir/__init__.py` | 99 | C++ 胶水 | 🟡 重要 |
| `debug.py` | 49 | 调试模式 | 🟢 浏览 |
| `serialization.py` | 118 | 模型保存/加载 | 🟢 浏览 |
| `lr_scheduler.py` | 91 | 学习率调度 | 🟢 浏览 |

### Python 侧 — 已有框架（修改部分）

| 文件 | 修改内容 | 关键行号 |
|------|---------|---------|
| `autograd.py` | `TensorOp.__call__` 添加 trace 钩子 | 87-90 |
| `nn/nn_basic.py` | `Module.__call__` 添加 trace 钩子 + `Module.to()` | 80-109 |
| `optim.py` | 添加 `clip_grad_norm()` | 17-43 |
| `__init__.py` | 导出 jit/debug/save/load/lr_scheduler | 14-19 |
| `backend_selection.py` | 支持 debug 模式 | 15-48 |

### C++ 侧 — 编译器

| 文件 | 行数 | 职责 | 阅读优先级 |
|------|------|------|-----------|
| `needle_compiler.h` | 53 | 数据结构定义 | 🔴 必读 |
| `needle_compiler.cc` | 584 | JSON 解析 + pybind11 入口 | 🔴 必读 |
| `mlir_gen.h` | 177 | MLIR 生成接口 | 🔴 必读 |
| `mlir_gen.cc` | 800 | 21 种算子 MLIR 翻译 + 融合体 | 🔴 必读 |
| `passes.h` | 52 | Pass 管线接口 | 🟡 重要 |
| `passes.cc` | 164 | 13 步 Pass 管线 | 🟡 重要 |
| `jit_engine.h` | 81 | JIT 引擎接口 | 🟡 重要 |
| `jit_engine.cc` | 237 | JIT 编译 + 缓存 | 🟡 重要 |

### 测试与应用

| 文件 | 行数 | 职责 |
|------|------|------|
| `apps/models.py` | 119 | ResNet9 / LanguageModel |
| `apps/mlir_demo.py` | 595 | 完整演示 |
| `tests/hw4/test_mlir_backend.py` | 402 | 编译器测试 |
| `benchmarks/bench_mlir.py` | 300 | 性能基准 |

---

## 附录 C：动手实验建议

读完代码后，建议做以下练习来巩固理解：

### 实验 1：跑通 Python 侧管线（无需 C++ 编译）

```bash
cd hw4
NEEDLE_BACKEND=np python apps/mlir_demo.py --quick
```

观察输出中的：
- 子图数量和大小
- 融合组标记
- trace 耗时

### 实验 2：新增一个算子

试着新增 `GELU` 算子支持：
1. 在 `SUPPORTED_MLIR_OPS` 和 `FUSIBLE_ELEMENTWISE_OPS` 中添加 `"GELU"`
2. 在 `mlir_gen.cc` 的 `EMIT_FN_TABLE` 和 `emit_fused_elementwise` 的 dispatch 中添加处理
3. 跑测试验证

### 实验 3：修改融合策略

试着把 `FUSION_BARRIER_OPS` 中的 `Reshape` 移除，观察：
- 哪些融合组变大了？
- 生成的 JSON 有什么变化？

---

> **最后提醒**：这个项目的核心价值在于理解"编译器如何把深度学习计算图变成高效的机器码"。阅读时，始终带着这个问题："这一步的输出是什么？下一步的输入是什么？数据在这两个模块之间是如何传递的？"
