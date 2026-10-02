#pragma once

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LogicalResult.h>

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace needle {
namespace compiler {

/// Build the full optimization pass pipeline for a given optimization level.
///
/// opt_level 0: Only lower, no optimizations
/// opt_level 1: Tiling + vectorization
/// opt_level 2: Fusion + tiling + vectorization + parallelization (default, recommended)
/// opt_level 3: All of above + aggressive tiling
///
/// Pass pipeline at opt_level=2:
///   1. linalg-fuse-elementwise-ops
///   2. linalg-tile (MatMul, Conv)
///   3. linalg-generalize-named-ops  (recommended, not optional)
///   4. linalg-fuse-elementwise-ops (again, post-tiling)
///   5. scf-parallel-loop-fusion
///   6. one-shot-bufferize (tensor → memref)
///   7. convert-linalg-to-affine-loops
///   8. affine-loop-fusion
///   9. affine-super-vectorize {virtual-vector-size=8}
///   10. lower-affine (affine.for → scf.for)
///   11. convert-scf-to-cf
///   12. convert-vector-to-llvm
///   13. convert-to-llvm + reconcile-unrealized-casts
mlir::LogicalResult build_optimization_pipeline(
    mlir::OpPassManager& pm, int opt_level);

/// Create a tiling pass with specified tile sizes.
/// Defaults: MatMul → {m=64, n=64, k=32}, Conv → {n=1, h=6, w=6, c=32, kh=3, kw=3}
std::unique_ptr<mlir::Pass> create_tiling_pass(
    const std::map<std::string, std::vector<int64_t>>& tile_sizes);

/// Create a vectorization pass (affine-super-vectorize).
/// virtual_vector_size=8 for 256-bit AVX2 (float32 × 8).
std::unique_ptr<mlir::Pass> create_vectorization_pass(int virtual_vector_size = 8);

/// Create a parallelization pass (scf.for → scf.parallel).
std::unique_ptr<mlir::Pass> create_parallelization_pass();

}  // namespace compiler
}  // namespace needle
