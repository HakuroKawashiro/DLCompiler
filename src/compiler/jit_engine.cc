#include "jit_engine.h"
#include "passes.h"

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SHA256.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace needle {
namespace compiler {

// ============================================================================
// Constructor
// ============================================================================

JITEngine::JITEngine() {
    // Initialize LLVM native target
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Configure JIT target machine from current CPU
    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb) {
        llvm::errs() << "Warning: Failed to detect host CPU for JIT: "
                     << llvm::toString(jtmb.takeError()) << "\n";
        return;
    }

    // Create LLJIT instance
    auto jit = llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(*jtmb).create();
    if (!jit) {
        llvm::errs() << "Warning: Failed to create LLJIT: "
                     << llvm::toString(jit.takeError()) << "\n";
        return;
    }

    jit_ = std::move(*jit);
    llvm_ctx_ = std::make_unique<llvm::LLVMContext>();

    // Register MLIR → LLVM IR translations (done lazily in mlir_to_llvm)
    // Note: MLIRContext is owned by CompilerInstance, not JITEngine.
    // The dialect translations are registered when the MLIR module is lowered.
}

// ============================================================================
// MLIR → LLVM IR conversion
// ============================================================================

llvm::Expected<std::unique_ptr<llvm::Module>>
JITEngine::mlir_to_llvm(mlir::ModuleOp& mlir_module, llvm::LLVMContext& llvm_ctx) {
    // Register MLIR → LLVM IR translation dialects
    mlir::MLIRContext* mlir_ctx_ptr = mlir_module.getContext();

    // Run the lowering pass pipeline to get to LLVM IR
    mlir::PassManager pm(mlir_ctx_ptr);

    int opt_level = get_optimization_level();
    if (auto result = build_optimization_pipeline(pm, opt_level); mlir::failed(result)) {
        return llvm::make_error<llvm::StringError>(
            "Failed to build pass pipeline",
            llvm::inconvertibleErrorCode());
    }

    if (mlir::failed(pm.run(mlir_module))) {
        return llvm::make_error<llvm::StringError>(
            "Pass pipeline execution failed",
            llvm::inconvertibleErrorCode());
    }

    // Translate MLIR Module → LLVM IR
    llvm::LLVMContext llvm_context;
    auto llvm_module = mlir::translateModuleToLLVMIR(mlir_module, llvm_context);
    if (!llvm_module) {
        return llvm::make_error<llvm::StringError>(
            "Failed to translate MLIR to LLVM IR",
            llvm::inconvertibleErrorCode());
    }

    return std::move(llvm_module);
}

// ============================================================================
// Compile and get function
// ============================================================================

llvm::Expected<void*> JITEngine::compile_and_get_fn(
    mlir::ModuleOp& mlir_module,
    const std::string& func_name) {

    if (!jit_) {
        return llvm::make_error<llvm::StringError>(
            "JIT engine not initialized",
            llvm::inconvertibleErrorCode());
    }

    // Convert MLIR → LLVM IR
    auto llvm_module_or_err = mlir_to_llvm(mlir_module, *llvm_ctx_);
    if (!llvm_module_or_err) {
        return llvm_module_or_err.takeError();
    }

    auto llvm_module = std::move(*llvm_module_or_err);

    // Add LLVM IR module to JIT
    auto tsm = llvm::orc::ThreadSafeModule(std::move(llvm_module),
                                            std::make_unique<llvm::LLVMContext>());
    if (auto err = jit_->addIRModule(std::move(tsm))) {
        return std::move(err);
    }

    // Look up the compiled function symbol
    auto sym_or_err = jit_->lookup(func_name);
    if (!sym_or_err) {
        return sym_or_err.takeError();
    }

    void* fn_ptr = reinterpret_cast<void*>(sym_or_err->getValue());

    // Cache the function pointer
    std::string cache_full_key = func_name;
    fn_cache_[cache_full_key] = fn_ptr;

    return fn_ptr;
}

// ============================================================================
// Cache logic
// ============================================================================

void* JITEngine::get_cached_or_compile(
    const std::string& cache_key,
    mlir::ModuleOp& mlir_module,
    const std::string& func_name) {

    // Check in-memory cache first
    std::string full_key = cache_key + "::" + func_name;
    auto it = fn_cache_.find(full_key);
    if (it != fn_cache_.end()) {
        return it->second;
    }

    // Not cached — compile
    auto fn_ptr_or_err = compile_and_get_fn(mlir_module, func_name);
    if (!fn_ptr_or_err) {
        llvm::errs() << "JIT compilation error: "
                     << llvm::toString(fn_ptr_or_err.takeError()) << "\n";
        return nullptr;
    }

    void* fn_ptr = *fn_ptr_or_err;
    fn_cache_[full_key] = fn_ptr;
    return fn_ptr;
}

std::string JITEngine::compute_cache_key(
    const std::string& subgraph_json,
    const std::string& func_name) {

    // SHA256 hash of the subgraph JSON
    llvm::SHA256 hasher;
    hasher.update(llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t*>(subgraph_json.data()),
        subgraph_json.size()));

    auto hash = hasher.final();

    // Convert to hex string (first 32 chars = 128 bits to avoid collisions)
    std::stringstream ss;
    ss << llvm::format_hex(hash[0], 2, false);
    for (size_t i = 1; i < 16; ++i) {  // 16 bytes = 32 hex chars = 128 bits
        ss << llvm::format_hex(hash[i], 2, false);
    }
    return ss.str();
}

void JITEngine::clear_cache() {
    fn_cache_.clear();
}

// ============================================================================
// Environment variable helpers
// ============================================================================

int get_optimization_level() {
    const char* env = std::getenv("NEEDLE_MLIR_OPT");
    if (env) {
        int level = std::atoi(env);
        if (level >= 0 && level <= 3) return level;
    }
    return 2;  // Default: full optimization
}

bool get_mlir_dump_enabled() {
    const char* env = std::getenv("NEEDLE_MLIR_DUMP");
    return env && std::string(env) == "1";
}

std::string get_cache_dir() {
    const char* env = std::getenv("NEEDLE_MLIR_CACHE_DIR");
    if (env) return std::string(env);

    // Default: ~/.needle_cache
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");  // Windows
    if (home) return std::string(home) + "/.needle_cache";

    return ".needle_cache";
}

bool get_cache_disabled() {
    const char* env = std::getenv("NEEDLE_MLIR_DISABLE_CACHE");
    return env && std::string(env) == "1";
}

}  // namespace compiler
}  // namespace needle
