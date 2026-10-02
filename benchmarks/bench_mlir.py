"""Performance benchmarks for MLIR compiler vs. existing backend.

Measures:
- MatMul GFLOPS across sizes (MLIR vs C++ naive vs NumPy vs PyTorch CPU)
- Conv throughput
- Element-wise fusion chain speedup
- ResNet9 inference latency
- ResNet9 training epoch time

Usage:
    python benchmarks/bench_mlir.py [--quick] [--csv output.csv]
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import needle as ndl


# ============================================================================
# Benchmark utilities
# ============================================================================

def timeit(fn, warmup=3, repeat=10):
    """Measure average execution time of fn() over `repeat` runs after warmup."""
    # Warmup
    for _ in range(warmup):
        fn()
    # Timed runs
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def gflops_matmul(m, n, p, time_sec):
    """Compute GFLOPS for a (m,n) @ (n,p) matmul."""
    ops = 2.0 * m * n * p  # multiply-add counts as 2 ops
    return ops / time_sec / 1e9


def print_header(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name, mlir_time, eager_time, speedup):
    """Print a benchmark result row."""
    print(f"  {name:<30s}  MLIR: {mlir_time*1000:8.3f}ms  "
          f"Eager: {eager_time*1000:8.3f}ms  "
          f"Speedup: {speedup:5.1f}x")


# ============================================================================
# Benchmarks
# ============================================================================

def bench_matmul_sweep(min_size=128, max_size=2048, step=128):
    """Benchmark MatMul across a range of square matrix sizes."""
    print_header("MatMul GFLOPS Sweep")

    sizes = list(range(min_size, max_size + 1, step))
    results = []

    for n in sizes:
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)

        # Eager (C++ backend)
        ta = ndl.Tensor(a)
        tb = ndl.Tensor(b)

        def eager_fn():
            return (ta @ tb).numpy()

        eager_time, _ = timeit(eager_fn, warmup=2, repeat=5)

        # MLIR (if available)
        mlir_time = None
        try:
            compiled = ndl.jit(_MatMulWrapper())
            tma = ndl.Tensor(a.copy())
            tmb = ndl.Tensor(b.copy())

            def mlir_fn():
                return compiled(tma, tmb).numpy()

            mlir_time, _ = timeit(mlir_fn, warmup=2, repeat=5)
        except (ImportError, RuntimeError):
            pass

        eager_gflops = gflops_matmul(n, n, n, eager_time)
        results.append({
            'size': n,
            'eager_time_ms': eager_time * 1000,
            'eager_gflops': eager_gflops,
            'mlir_time_ms': mlir_time * 1000 if mlir_time else None,
            'mlir_gflops': gflops_matmul(n, n, n, mlir_time) if mlir_time else None,
        })

        status = f"GFLOPS: {eager_gflops:.1f}"
        if mlir_time:
            mlir_gflops = gflops_matmul(n, n, n, mlir_time)
            status += f" (MLIR: {mlir_gflops:.1f}, {eager_gflops/mlir_gflops:.1f}x)"
        print(f"  n={n:4d}  {status}")

    return results


def bench_fusion_chain():
    """Benchmark element-wise fusion chains of different lengths."""
    print_header("Element-wise Fusion Chain Speedup")

    shape = (128, 128, 64)
    lengths = [2, 5, 10, 20]

    for L in lengths:
        # Build a chain of L element-wise operations
        # Eager: L separate kernels, L memory round-trips
        data = [np.random.randn(*shape).astype(np.float32) for _ in range(L + 1)]

        def eager_chain():
            x = ndl.Tensor(data[0])
            for i in range(L):
                x = x + ndl.Tensor(data[i + 1])
                x = ndl.ops.relu(x)
            return x.numpy()

        eager_time, _ = timeit(eager_chain, warmup=2, repeat=5)

        # MLIR: single fused kernel
        try:
            compiled = ndl.jit(_FusionChainModule(L))

            def mlir_chain():
                tensors = [ndl.Tensor(d) for d in data]
                return compiled(*tensors).numpy()

            mlir_time, _ = timeit(mlir_chain, warmup=2, repeat=5)
            speedup = eager_time / mlir_time
            print_result(f"Fusion chain L={L}", mlir_time, eager_time, speedup)
        except (ImportError, RuntimeError):
            print(f"  Fusion chain L={L}: MLIR not available, eager={eager_time*1000:.3f}ms")


def bench_resnet9_inference():
    """Benchmark ResNet9 single-batch forward latency."""
    print_header("ResNet9 Inference Latency")

    from apps.models import ResNet9

    batch_sizes = [1, 32, 64, 128]

    for bs in batch_sizes:
        model = ResNet9()
        x = np.random.randn(bs, 32, 32, 3).astype(np.float32)

        # Eager
        def eager_fn():
            t = ndl.Tensor(x)
            return model(t).numpy()

        eager_time, _ = timeit(eager_fn, warmup=2, repeat=5)

        # MLIR
        try:
            compiled = ndl.jit(model)
            x_copy = x.copy()

            def mlir_fn():
                t = ndl.Tensor(x_copy)
                return compiled(t).numpy()

            mlir_time, _ = timeit(mlir_fn, warmup=2, repeat=5)
            speedup = eager_time / mlir_time
            print_result(f"ResNet9 batch={bs}", mlir_time, eager_time, speedup)
        except (ImportError, RuntimeError):
            print(f"  ResNet9 batch={bs}: MLIR not available, eager={eager_time*1000:.3f}ms")


def bench_resnet9_training_epoch():
    """Benchmark one CIFAR-10 training epoch."""
    print_header("ResNet9 Training — 1 Epoch")

    from apps.models import ResNet9

    try:
        model = ResNet9()
        opt = ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = ndl.nn.SoftmaxLoss()

        # Load a small batch of CIFAR-10
        # For benchmarking, use random data
        x = np.random.randn(128, 32, 32, 3).astype(np.float32)
        y = np.random.randint(0, 10, 128)

        def eager_step():
            opt.reset_grad()
            logits = model(ndl.Tensor(x))
            loss = loss_fn(logits, ndl.Tensor(y))
            loss.backward()
            opt.step()
            return loss.numpy()

        eager_time, _ = timeit(eager_step, warmup=2, repeat=5)

        # MLIR
        try:
            compiled_model = ndl.jit(model)
            opt_mlir = ndl.optim.Adam(compiled_model.parameters(), lr=0.001, weight_decay=0.001)

            def mlir_step():
                opt_mlir.reset_grad()
                logits = compiled_model(ndl.Tensor(x.copy()))
                loss = loss_fn(logits, ndl.Tensor(y))
                loss.backward()
                opt_mlir.step()
                return loss.numpy()

            mlir_time, _ = timeit(mlir_step, warmup=2, repeat=5)
            speedup = eager_time / mlir_time
            print_result(f"Training step (batch=128)", mlir_time, eager_time, speedup)
        except (ImportError, RuntimeError):
            print(f"  Training step: MLIR not available, eager={eager_time*1000:.3f}ms")

    except Exception as e:
        print(f"  Training benchmark failed: {e}")


# ============================================================================
# Wrapper modules for benchmark
# ============================================================================

class _MatMulWrapper(ndl.nn.Module):
    """Simple module wrapping a single MatMul for benchmarking."""
    def forward(self, a, b):
        return a @ b


class _FusionChainModule(ndl.nn.Module):
    """Module wrapping a chain of L add+relu operations."""
    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, *args):
        x = args[0]
        for i in range(self.L):
            x = x + args[i + 1]
            x = ndl.ops.relu(x)
        return x


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Needle MLIR Benchmarks")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmarks (smaller sizes)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Output CSV file path")
    args = parser.parse_args()

    # Check MLIR availability via MLIRDevice.enabled()
    # (ndl.backend_mlir is not exposed via needle/__init__.py)
    try:
        from needle.backend_mlir import MLIRDevice
        mlir_ok = MLIRDevice().enabled()
    except (ImportError, RuntimeError):
        mlir_ok = False

    print("Needle MLIR Backend Benchmarks")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend: {'MLIR' if mlir_ok else 'Eager (MLIR not available)'}")

    if args.quick:
        bench_matmul_sweep(min_size=128, max_size=512, step=128)
    else:
        bench_matmul_sweep(min_size=128, max_size=2048, step=128)

    bench_fusion_chain()
    bench_resnet9_inference()
    bench_resnet9_training_epoch()

    print("\nBenchmarks complete.\n")


if __name__ == "__main__":
    main()
