#include "passes.h"

#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Vector/Transforms/Passes.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace needle {
namespace compiler {

using namespace mlir;

// ============================================================================
// Pass pipeline builder
// ============================================================================

LogicalResult build_optimization_pipeline(OpPassManager& pm, int opt_level) {
    if (opt_level < 0) opt_level = 0;
    if (opt_level > 3) opt_level = 3;

    // === Level 0+: Always run bufferization and lowering ===

    // ① linalg-fuse-elementwise-ops (if opt_level >= 1)
    //    Fuses adjacent linalg.generic ops. Element-wise chains become single generic.
    if (opt_level >= 1) {
        pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
    }

    // ② linalg-tile (if opt_level >= 1, only MatMul and Conv)
    if (opt_level >= 1) {
        // Tiling options for MatMul
        linalg::LinalgTilingOptions matmul_tile_opts;
        matmul_tile_opts.setTileSizes({64, 64, 32});
        pm.addNestedPass<func::FuncOp>(
            createLinalgTilingPass(matmul_tile_opts));

        // Tiling options for Conv
        linalg::LinalgTilingOptions conv_tile_opts;
        conv_tile_opts.setTileSizes({1, 6, 6, 32, 3, 3});
        pm.addNestedPass<func::FuncOp>(
            createLinalgTilingPass(conv_tile_opts));
    }

    // ③ linalg-generalize-named-ops (if opt_level >= 1, recommended)
    //    Converts linalg.matmul/conv_2d to linalg.generic for subsequent loop-level fusion.
    //    NOTE: Per the plan, this should NOT be marked "optional" — skipping it
    //    causes type II fusion (producer-consumer) to fail after tiling.
    if (opt_level >= 1) {
        pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
    }

    // ④ linalg-fuse-elementwise-ops (again, if opt_level >= 1)
    //    Post-tiling: fuses element-wise ops that appeared inside tiled loop bodies.
    if (opt_level >= 1) {
        pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
    }

    // ⑤ scf-parallel-loop-fusion (if opt_level >= 2)
    // Fuses adjacent scf.parallel loops that result from tiling.
    // Note: effective only when tiling produces scf.parallel (setLoopType)
    // rather than the default scf.for. Even when no parallel loops exist,
    // this pass is a harmless no-op.
    if (opt_level >= 2) {
        pm.addNestedPass<func::FuncOp>(
            mlir::createParallelLoopFusionPass());
    }

    // === Bufferization: tensor → memref ===
    // ⑥ one-shot-bufferize
    pm.addPass(bufferization::createOneShotBufferizePass());

    // === Lowering to loops ===
    // ⑦ convert-linalg-to-affine-loops (if opt_level >= 2)
    if (opt_level >= 1) {
        pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
    }

    // ⑧ affine-loop-fusion (if opt_level >= 2)
    if (opt_level >= 2) {
        pm.addNestedPass<func::FuncOp>(createLoopFusionPass());
    }

    // ⑨ affine-super-vectorize {virtual-vector-size=8} (if opt_level >= 1)
    if (opt_level >= 1) {
        // Virtual vector size 8 = 256-bit (AVX2 float32×8)
        // For AVX-512, use 16
        pm.addNestedPass<func::FuncOp>(createSuperVectorizePass({8}));
    }

    // ⑩ lower-affine → scf
    pm.addPass(createLowerAffinePass());

    // ⑪ convert-scf-to-cf
    pm.addPass(createConvertSCFToCFPass());

    // ⑫ convert-vector-to-llvm
    pm.addPass(createConvertVectorToLLVMPass());

    // ⑬ convert-to-llvm + reconcile-unrealized-casts
    // Full LLVM lowering: arith, math, memref, func → LLVM IR
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createMathToLLVMConversionPass());
    pm.addPass(createMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    return success();
}

// ============================================================================
// Individual pass creators
// ============================================================================

std::unique_ptr<Pass> create_tiling_pass(
    const std::map<std::string, std::vector<int64_t>>& tile_sizes) {

    linalg::LinalgTilingOptions opts;

    // Apply MatMul tile sizes if present
    auto matmul_it = tile_sizes.find("matmul");
    if (matmul_it != tile_sizes.end()) {
        opts.setTileSizes(llvm::ArrayRef<int64_t>(matmul_it->second));
    }

    return createLinalgTilingPass(opts);
}

std::unique_ptr<Pass> create_vectorization_pass(int virtual_vector_size) {
    // Create affine-super-vectorize pass with the given virtual vector size.
    // virtual_vector_size=8 corresponds to AVX2 (256-bit / 32-bit float = 8 elements).
    return createSuperVectorizePass({static_cast<unsigned>(virtual_vector_size)});
}

std::unique_ptr<Pass> create_parallelization_pass() {
    // Convert outer scf.for loops to scf.parallel for multi-threading.
    // This pass annotates parallel loops; the actual parallelization happens
    // during LLVM lowering (mapped to OpenMP or pthread).
    //
    // In MLIR 18+, this may be part of the SCF parallel loop fusion or
    // a standalone pass. The exact API depends on the MLIR version.
    return nullptr;  // Placeholder — depends on MLIR version specifics
}

}  // namespace compiler
}  // namespace needle
