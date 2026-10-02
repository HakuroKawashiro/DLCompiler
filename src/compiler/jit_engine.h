#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace needle {
namespace compiler {

/// JIT engine wrapping LLVM ORC JIT.
///
/// Caches compiled function pointers keyed by (subgraph_hash + func_name).
/// Handles MLIR Module → LLVM IR → machine code compilation.
class JITEngine {
public:
    JITEngine();
    ~JITEngine() = default;

    /// Compile an MLIR Module to machine code and return a function pointer.
    ///
    /// Steps:
    /// 1. Convert MLIR Module to LLVM IR module
    /// 2. Add LLVM IR to the ORC JIT
    /// 3. Look up the function symbol
    /// 4. Cache and return the function pointer
    llvm::Expected<void*> compile_and_get_fn(
        mlir::ModuleOp& mlir_module,
        const std::string& func_name);

    /// Get a cached function pointer, or compile if not cached.
    ///
    /// Cached functions are keyed by (cache_key, func_name).
    /// If the cache key is found, returns the cached pointer immediately
    /// without re-compilation.
    void* get_cached_or_compile(
        const std::string& cache_key,
        mlir::ModuleOp& mlir_module,
        const std::string& func_name);

    /// Compute a cache key from descriptor and function name.
    /// Uses SHA256 hash (first 32 hex chars = 128 bits to avoid collisions).
    static std::string compute_cache_key(
        const std::string& subgraph_json,
        const std::string& func_name);

    /// Clear the in-memory function pointer cache.
    void clear_cache();

    /// Check if the JIT engine is operational.
    bool is_ready() const { return jit_ != nullptr; }

private:
    /// Convert an MLIR Module to an LLVM IR module.
    /// Registers all necessary dialect translations and runs the lowering pipeline.
    llvm::Expected<std::unique_ptr<llvm::Module>> mlir_to_llvm(
        mlir::ModuleOp& mlir_module,
        llvm::LLVMContext& llvm_ctx);

    /// The LLVM ORC JIT instance.
    std::unique_ptr<llvm::orc::LLJIT> jit_;

    /// In-memory function pointer cache: (cache_key + "::" + func_name) -> fn_ptr
    std::unordered_map<std::string, void*> fn_cache_;

    /// LLVM context for this JIT session.
    std::unique_ptr<llvm::LLVMContext> llvm_ctx_;
};

}  // namespace compiler
}  // namespace needle
