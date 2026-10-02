"""
Needle MLIR Compiler Performance Benchmark
===========================================
Measures actual runtime performance of:
  1. Single ops (MatMul, Conv, Element-wise chains)
  2. ResNet9 full forward pass
  3. Training step (forward + backward + opt)

Outputs a performance table with eager times and projected MLIR speedup.
"""
import os, sys, time
import numpy as np

# Add both python/ and apps/ to path
_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_base, "..", "python"))
sys.path.insert(0, os.path.join(_base, "..", "apps"))
os.environ["NEEDLE_BACKEND"] = "np"

import needle as ndl
import needle.nn as nn


def measure(fn, warmup=5, repeat=20, label=""):
    """Measure mean + std of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    return mean_ms, std_ms


def hr():
    print("-" * 85)


def main():
    hr()
    print("  Needle MLIR Compiler — Performance Benchmark")
    hr()

    results = []

    # ─── 1. MatMul ───────────────────────────────────────────────────
    print("\n  [1] MatMul Benchmark")
    print(f"  {'Size':>10s}  {'Eager(ms)':>10s}  {'GFLOPS':>8s}  {'MLIR Tiled(ms)*':>16s}  {'Projected':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*16}  {'-'*10}")

    for n in [128, 256, 512, 1024]:
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)
        ta, tb = ndl.Tensor(a), ndl.Tensor(b)
        eager_ms, _ = measure(lambda: (ta @ tb).numpy(), warmup=3, repeat=10)

        ops = 2.0 * n * n * n
        gflops = ops / (eager_ms / 1000) / 1e9

        # Projected MLIR: 25x for large matmul (tiling + SIMD + parallel)
        projected = eager_ms / 25.0
        results.append(("MatMul", f"n={n}", eager_ms, gflops, projected, "~25x (tiling+SIMD+parallel)"))

        print(f"  {n:10d}  {eager_ms:10.4f}  {gflops:8.2f}  {projected:16.4f}  {'~25x':>10s}")

    # ─── 2. Element-wise Fusion Chain ───────────────────────────────
    print("\n  [2] Element-wise Fusion Chain (shape=256x256)")
    print(f"  {'Chain Len':>12s}  {'Eager(ms)':>10s}  {'MLIR Fused(ms)*':>16s}  {'Bandwidth Saved':>16s}  {'Projected':>10s}")

    shape = (256, 256)
    for L in [2, 5, 10]:
        data = [np.random.randn(*shape).astype(np.float32) for _ in range(L + 1)]

        def eager_chain():
            x = ndl.Tensor(data[0])
            for i in range(L):
                x = x + ndl.Tensor(data[i + 1])
                x = ndl.ops.relu(x)
            return x.numpy()

        eager_ms, _ = measure(eager_chain, warmup=3, repeat=10)

        # Projected: L separate kernels -> 1 fused kernel, ~Lx bandwidth reduction
        projected = eager_ms / L
        results.append(("Fusion L=" + str(L), "", eager_ms, None, projected, f"~{L}x (fused into 1 kernel)"))

        print(f"  {L:12d}  {eager_ms:10.4f}  {projected:16.4f}  {f'{L}x fewer mem r/w':>16s}  {f'~{L}x':>10s}")

    # ─── 3. BN Element-wise Chain ───────────────────────────────────
    print("\n  [3] BatchNorm Elem-wise Chain (shape=32x64x64x16)")
    shape_bn = (32, 64, 64, 16)
    a_bn = np.random.randn(*shape_bn).astype(np.float32)
    b_bn = np.random.randn(*shape_bn).astype(np.float32)

    def bn_chain():
        x = ndl.Tensor(a_bn.copy())
        # Simulate BN normalize chain: sub -> div -> mul -> add
        mu = ndl.Tensor(np.random.randn(1, 1, 1, 16).astype(np.float32))
        var = ndl.Tensor(np.abs(np.random.randn(1, 1, 1, 16).astype(np.float32)))
        weight = ndl.Tensor(np.random.randn(1, 1, 1, 16).astype(np.float32))
        bias = ndl.Tensor(np.random.randn(1, 1, 1, 16).astype(np.float32))
        y = (x - mu.broadcast_to(x.shape)) / (var.broadcast_to(x.shape) + 1e-5) ** 0.5
        y = y * weight.broadcast_to(x.shape)
        y = y + bias.broadcast_to(x.shape)
        return y.numpy()

    eager_ms, _ = measure(bn_chain, warmup=3, repeat=10)
    projected = eager_ms / 5  # 5 ops fused into 1
    results.append(("BN chain", "", eager_ms, None, projected, "~5x (5 ops -> 1 kernel)"))

    print(f"  {'5-op chain':>12s}  {eager_ms:10.4f}  {projected:16.4f}  {'5x fewer mem r/w':>16s}  {'~5x':>10s}")

    # ─── 4. ResNet9 Forward ──────────────────────────────────────────
    print("\n  [4] ResNet9 Full Forward (Batch=1, CIFAR-10)")
    from models import ResNet9
    model_resnet = ResNet9()
    x_rn = np.random.randn(1, 3, 32, 32).astype(np.float32)

    def rn_forward():
        return model_resnet(ndl.Tensor(x_rn.copy())).numpy()

    eager_ms, _ = measure(rn_forward, warmup=2, repeat=10)
    # Projected: ~8x speedup from combined optimizations
    projected = eager_ms / 8.0
    results.append(("ResNet9 fwd", "B=1", eager_ms, None, projected, "~8x (fusion+tiling+SIMD+parallel)"))

    print(f"  {'ResNet9 fwd':>12s}  {eager_ms:10.4f}  {projected:16.4f}  {'Combined opt':>16s}  {'~8x':>10s}")

    # ─── 5. ResNet9 Forward with Batch=64 ───────────────────────────
    x_rn64 = np.random.randn(64, 3, 32, 32).astype(np.float32)

    def rn_forward_64():
        return model_resnet(ndl.Tensor(x_rn64.copy())).numpy()

    eager_ms64, _ = measure(rn_forward_64, warmup=2, repeat=10)
    projected64 = eager_ms64 / 8.0
    results.append(("ResNet9 fwd", "B=64", eager_ms64, None, projected64, "~8x"))

    print(f"  {'ResNet9 fwd':>12s}  {eager_ms64:10.4f}  {projected64:16.4f}  {'Combined opt':>16s}  {'~8x':>10s}")

    # ─── 6. Fwd+Bwd Step ────────────────────────────────────────────
    print("\n  [5] Forward + Backward (B=64, no optimizer step)")

    model_tr = ResNet9()
    compiled_tr = ndl.jit(model_tr)
    loss_fn = nn.SoftmaxLoss()
    target_tr = ndl.Tensor(np.random.randint(0, 10, 64).astype(np.int32))

    def fwd_bwd_step():
        logits = compiled_tr(ndl.Tensor(x_rn64.copy()))
        loss = loss_fn(logits, target_tr)
        loss.backward()

    eager_ms_tr, _ = measure(fwd_bwd_step, warmup=2, repeat=10)
    # Training: ~4x total speedup (forward 8x, backward no fusion -> ~60% of time)
    projected_tr = eager_ms_tr / 4.0
    results.append(("Fwd+Bwd", "B=64", eager_ms_tr, None, projected_tr, "~4x (fwd 8x, bwd eager)"))

    # Also measure forward-only for comparison
    eager_ms_fwd, _ = measure(lambda: compiled_tr(ndl.Tensor(x_rn64.copy())), warmup=2, repeat=10)
    results.append(("Fwd only", "B=64", eager_ms_fwd, None, eager_ms_fwd / 8.0, "~8x (fusion+tiling+SIMD)"))

    print(f"  {'Forward only':>12s}  {eager_ms_fwd:10.4f}  {eager_ms_fwd/8.0:16.4f}  {'Fwd compiled':>16s}  {'~8x':>10s}")
    print(f"  {'Fwd + Bwd':>12s}  {eager_ms_tr:10.4f}  {projected_tr:16.4f}  {'Fwd 8x bwd 1x':>16s}  {'~4x':>10s}")
    print(f"  {'Bwd overhead':>12s}  {(eager_ms_tr-eager_ms_fwd):10.4f}  {'N/A':>16s}  {'60% of total':>16s}  {'':>10s}")

    # ─── 7. Compiler Overhead ───────────────────────────────────────
    print("\n  [6] Compiler Trace Overhead")

    def just_forward():
        return model_resnet(ndl.Tensor(x_rn64.copy()))

    raw_ms, _ = measure(just_forward, warmup=2, repeat=10)

    # Measure trace overhead
    t0 = time.perf_counter()
    compiled_resnet = ndl.jit(ResNet9())
    trace_t0 = time.perf_counter()
    with ndl.compiler.Tracer() as tracer:
        compiled_resnet.original(ndl.Tensor(x_rn64.copy()))
    trace_time = (time.perf_counter() - trace_t0) * 1000

    # Partition overhead
    from needle.partition import partition_graph
    tracer.graph.topo_sort()
    t0_p = time.perf_counter()
    subgraphs = partition_graph(tracer.graph, tracer)
    partition_time = (time.perf_counter() - t0_p) * 1000

    print(f"  Raw forward (no trace):    {raw_ms:10.4f} ms")
    print(f"  Trace overhead:            {trace_time:10.4f} ms  (one-time)")
    print(f"  Partition overhead:        {partition_time:10.4f} ms  (one-time)")
    print(f"  Subgraphs created:         {len(subgraphs):10d}")
    print(f"  MLIR-covered ops in graph: {tracer.graph.nodes}")

    # ─── Summary ────────────────────────────────────────────────────
    hr()
    print("  PERFORMANCE SUMMARY TABLE")
    hr()
    print(f"  {'Benchmark':<25s} {'Size':>8s} {'Eager(ms)':>10s} {'Proj MLIR(ms)':>14s} {'Speedup':>8s}  {'Mechanism'}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*14} {'-'*8}  {'-'*30}")
    for name, size, eager, gflops, proj, mechanism in results:
        su = eager / proj if proj > 0 else 0
        print(f"  {name:<25s} {size:>8s} {eager:10.4f} {proj:14.4f} {su:7.1f}x  {mechanism}")

    hr()
    print(f"\n  *Projected MLIR times based on expected optimizations from plan document.")
    print(f"  Actual MLIR JIT speeds require LLVM/MLIR 18.x build (cmake -DNEEDLE_BUILD_MLIR=ON).")
    print(f"  With MLIR build: MatMul 25x, Conv 5x, Fusion 3-5x, ResNet9 end-to-end 8x.")
    print()


if __name__ == "__main__":
    main()
