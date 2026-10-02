/// pybind11 module entry point for the MLIR compiler backend.
///
/// Exposes compile_and_run() to Python, which:
/// 1. Parses subgraph JSON → SubgraphDescriptor
/// 2. Generates MLIR IR via generate_mlir_module()
/// 3. Runs optimization pass pipeline
/// 4. JIT compiles to machine code via LLVM ORC JIT
/// 5. Executes the JIT function with input buffers, writing results to output buffers
///
/// Memory management:
/// - Input buffers are read-only; the JIT function reads from their memory
/// - Output buffers are pre-allocated by Python (from SubgraphDescriptor shapes);
///   the JIT function writes results directly into them (zero-copy)
/// - No cross-language memory ownership issues

#include "needle_compiler.h"
#include "mlir_gen.h"
#include "passes.h"
#include "jit_engine.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <nlohmann/json.hpp>  // JSON parsing (if available)

#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>

namespace py = pybind11;

namespace needle {
namespace compiler {

// Forward declaration
class CompilerInstance;

/// Singleton compiler instance (one per process).
/// JITEngine is shared across all subgraphs.
static CompilerInstance* g_compiler = nullptr;

/// Compiler instance holding JIT engine and MLIR context.
class CompilerInstance {
public:
    CompilerInstance() {
        // Register all required MLIR dialects
        ctx.getOrLoadDialect<mlir::linalg::LinalgDialect>();
        ctx.getOrLoadDialect<mlir::affine::AffineDialect>();
        ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
        ctx.getOrLoadDialect<mlir::math::MathDialect>();
        ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
        ctx.getOrLoadDialect<mlir::func::FuncDialect>();
        ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
        ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

        // Initialize JIT engine with host CPU detection
        try {
            jit_engine = std::make_unique<JITEngine>();
        } catch (const std::exception& e) {
            // JIT initialization may fail if LLVM target is not available
            // (e.g., on some embedded platforms). The compile_and_run()
            // function will return an error in that case.
        }
    }

    mlir::MLIRContext ctx;
    std::unique_ptr<JITEngine> jit_engine;
};

/// Parse SubgraphDescriptor from JSON string.
///
/// JSON schema (from serialize.py):
/// {
///   "nodes": [{"op": "EWiseAdd", "inputs": [0,1], "attrs": {}, "shape": [...], "dtype": "float32"}, ...],
///   "inputs": [{"node_id": 5, "shape": [...], "dtype": "float32"}, ...],
///   "params": [{"name": "w", "shape": [...], "dtype": "float32"}, ...],
///   "num_outputs": 2,
///   "fused_groups": [[0,3], [5,7]],
///   "output_node_indices": [0, 4],
///   "mutation_map": {"bn1.running_mean": 1}
/// }

// ═════════════════════════════════════════════════════════════════════════════
// Minimal recursive-descent JSON parser — sufficient for machine-generated JSON
// ═════════════════════════════════════════════════════════════════════════════
namespace {

class JsonReader {
public:
    explicit JsonReader(const std::string& json) : json_(json), pos_(0) {}

    // ── whitespace ──────────────────────────────────────────────────────
    void skip_ws() {
        while (pos_ < json_.size()) {
            char c = json_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { pos_++; continue; }
            break;
        }
    }

    char peek() { skip_ws(); check_bounds(); return json_[pos_]; }
    char next() { skip_ws(); check_bounds(); return json_[pos_++]; }

    void expect(char c) {
        char actual = next();
        if (actual != c)
            throw std::runtime_error(
                std::string("JSON parse error: expected '") + c + "' got '" + actual + "'");
    }

    // ── string ──────────────────────────────────────────────────────────
    std::string parse_string() {
        expect('"');
        std::string result;
        while (pos_ < json_.size() && json_[pos_] != '"') {
            if (json_[pos_] == '\\') {
                pos_++;
                check_bounds();
                switch (json_[pos_]) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case '/':  result += '/';  break;
                    case 'n':  result += '\n'; break;
                    case 't':  result += '\t'; break;
                    case 'r':  result += '\r'; break;
                    default:   result += json_[pos_]; break;
                }
            } else {
                result += json_[pos_];
            }
            pos_++;
        }
        expect('"');
        return result;
    }

    // ── number (returns int64_t; truncates float fractional parts) ──────
    int64_t parse_number() {
        skip_ws();
        bool neg = false;
        if (pos_ < json_.size() && json_[pos_] == '-') { neg = true; pos_++; }
        int64_t val = 0;
        while (pos_ < json_.size() && json_[pos_] >= '0' && json_[pos_] <= '9') {
            val = val * 10 + (json_[pos_] - '0');
            pos_++;
        }
        // fractional part
        if (pos_ < json_.size() && json_[pos_] == '.') {
            pos_++;
            while (pos_ < json_.size() && json_[pos_] >= '0' && json_[pos_] <= '9') pos_++;
        }
        // exponent
        if (pos_ < json_.size() && (json_[pos_] == 'e' || json_[pos_] == 'E')) {
            pos_++;
            if (pos_ < json_.size() && (json_[pos_] == '+' || json_[pos_] == '-')) pos_++;
            while (pos_ < json_.size() && json_[pos_] >= '0' && json_[pos_] <= '9') pos_++;
        }
        return neg ? -val : val;
    }

    // ── int64 array ─────────────────────────────────────────────────────
    std::vector<int64_t> parse_int64_array() {
        expect('[');
        std::vector<int64_t> result;
        skip_ws();
        if (pos_ < json_.size() && json_[pos_] != ']') {
            result.push_back(parse_number());
            while (true) {
                skip_ws();
                if (pos_ >= json_.size() || json_[pos_] == ']') break;
                expect(',');
                result.push_back(parse_number());
            }
        }
        expect(']');
        return result;
    }

    // ── pair array [[int,int], ...] ─────────────────────────────────────
    std::vector<std::pair<size_t, size_t>> parse_pair_array() {
        expect('[');
        std::vector<std::pair<size_t, size_t>> result;
        skip_ws();
        if (pos_ < json_.size() && json_[pos_] != ']') {
            result.push_back(parse_one_pair());
            while (true) {
                skip_ws();
                if (pos_ >= json_.size() || json_[pos_] == ']') break;
                expect(',');
                result.push_back(parse_one_pair());
            }
        }
        expect(']');
        return result;
    }

    // ── attrs object {key: number_or_array, ...} ────────────────────────
    // Stores scalar values as-is; array values → -1 sentinel (emit fns use defaults).
    std::map<std::string, int64_t> parse_attrs_object() {
        expect('{');
        std::map<std::string, int64_t> result;
        skip_ws();
        if (pos_ < json_.size() && json_[pos_] != '}') {
            do {
                std::string key = parse_string();
                expect(':');
                skip_ws();
                if (pos_ < json_.size() && json_[pos_] == '[') {
                    // Array value — skip it, store -1 sentinel
                    parse_skip_array();
                    result[key] = -1;
                } else {
                    result[key] = parse_number();
                }
                skip_ws();
            } while (pos_ < json_.size() && json_[pos_] == ',' && (next(), true));
        }
        expect('}');
        return result;
    }

    // ── string→int map e.g. {"bn1.running_mean": 1} ────────────────────
    std::map<std::string, size_t> parse_string_to_sizet_map() {
        expect('{');
        std::map<std::string, size_t> result;
        skip_ws();
        if (pos_ < json_.size() && json_[pos_] != '}') {
            do {
                std::string key = parse_string();
                expect(':');
                result[key] = static_cast<size_t>(parse_number());
                skip_ws();
            } while (pos_ < json_.size() && json_[pos_] == ',' && (next(), true));
        }
        expect('}');
        return result;
    }

    // ── helper: skip a JSON array (used to skip array-valued attrs) ─────
    void parse_skip_array() {
        expect('[');
        int depth = 1;
        while (depth > 0 && pos_ < json_.size()) {
            char c = json_[pos_++];
            if (c == '[') depth++;
            else if (c == ']') depth--;
            else if (c == '"') {
                // Skip string
                while (pos_ < json_.size() && json_[pos_] != '"') {
                    if (json_[pos_] == '\\') pos_++;
                    pos_++;
                }
                if (pos_ < json_.size()) pos_++; // closing quote
            }
        }
    }

private:
    void check_bounds() {
        if (pos_ >= json_.size())
            throw std::runtime_error("JSON parse error: unexpected end of input");
    }

    std::pair<size_t, size_t> parse_one_pair() {
        expect('[');
        size_t a = static_cast<size_t>(parse_number());
        expect(',');
        size_t b = static_cast<size_t>(parse_number());
        expect(']');
        return {a, b};
    }

    const std::string& json_;
    size_t pos_;
};

}  // namespace

static SubgraphDescriptor parse_subgraph_json(const std::string& json_str) {
    SubgraphDescriptor desc;
    JsonReader r(json_str);

    r.expect('{');

    // Parse each top-level field
    while (true) {
        r.skip_ws();
        if (r.peek() == '}') break;

        std::string key = r.parse_string();
        r.expect(':');

        if (key == "nodes") {
            r.expect('[');
            r.skip_ws();
            if (r.peek() != ']') {
                do {
                    NodeDescriptor node;
                    r.expect('{');
                    // Parse fields of each node
                    while (true) {
                        r.skip_ws();
                        if (r.peek() == '}') break;
                        std::string nk = r.parse_string();
                        r.expect(':');
                        if (nk == "op") {
                            node.op_name = r.parse_string();
                        } else if (nk == "inputs") {
                            auto arr = r.parse_int64_array();
                            for (auto v : arr) node.input_indices.push_back(static_cast<size_t>(v));
                        } else if (nk == "attrs") {
                            node.attrs = r.parse_attrs_object();
                        } else if (nk == "shape") {
                            node.output_shape = r.parse_int64_array();
                        } else if (nk == "dtype") {
                            node.dtype = r.parse_string();
                        }
                        r.skip_ws();
                        if (r.peek() == ',') r.next();
                    }
                    r.expect('}');
                    desc.nodes.push_back(std::move(node));
                    r.skip_ws();
                } while (r.peek() == ',' && (r.next(), true));
            }
            r.expect(']');

        } else if (key == "inputs") {
            r.expect('[');
            r.skip_ws();
            if (r.peek() != ']') {
                do {
                    r.expect('{');
                    std::vector<int64_t> shape;
                    std::string dtype = "float32";
                    while (true) {
                        r.skip_ws();
                        if (r.peek() == '}') break;
                        std::string ik = r.parse_string();
                        r.expect(':');
                        if (ik == "shape") {
                            shape = r.parse_int64_array();
                        } else if (ik == "dtype") {
                            dtype = r.parse_string();
                        } else if (ik == "node_id") {
                            r.parse_number(); // consumed, not stored
                        }
                        r.skip_ws();
                        if (r.peek() == ',') r.next();
                    }
                    r.expect('}');
                    desc.input_shapes.push_back(std::move(shape));
                    desc.input_dtypes.push_back(std::move(dtype));
                    r.skip_ws();
                } while (r.peek() == ',' && (r.next(), true));
            }
            r.expect(']');

        } else if (key == "params") {
            r.expect('[');
            r.skip_ws();
            if (r.peek() != ']') {
                do {
                    r.expect('{');
                    std::string name;
                    std::vector<int64_t> shape;
                    while (true) {
                        r.skip_ws();
                        if (r.peek() == '}') break;
                        std::string pk = r.parse_string();
                        r.expect(':');
                        if (pk == "name") {
                            name = r.parse_string();
                        } else if (pk == "shape") {
                            shape = r.parse_int64_array();
                        } else if (pk == "dtype") {
                            r.parse_string(); // consumed, not stored separately
                        }
                        r.skip_ws();
                        if (r.peek() == ',') r.next();
                    }
                    r.expect('}');
                    desc.param_names.push_back(std::move(name));
                    desc.param_shapes.push_back(std::move(shape));
                    r.skip_ws();
                } while (r.peek() == ',' && (r.next(), true));
            }
            r.expect(']');

        } else if (key == "num_outputs") {
            desc.num_outputs = static_cast<size_t>(r.parse_number());

        } else if (key == "fused_groups") {
            desc.fused_groups = r.parse_pair_array();

        } else if (key == "output_node_indices") {
            auto arr = r.parse_int64_array();
            for (auto v : arr) desc.output_node_indices.push_back(static_cast<size_t>(v));

        } else if (key == "mutation_map") {
            desc.mutation_map = r.parse_string_to_sizet_map();

        } // else: ignore unknown keys (forward compatibility)

        r.skip_ws();
        if (r.peek() == ',') r.next();
    }

    r.expect('}');
    return desc;
}
// ============================================================================
// Main Python-facing function
// ============================================================================

/// Compile and run a subgraph via MLIR JIT.
///
/// Python signature:
///   compile_and_run(cache_key: str, subgraph_json: str,
///                   input_buffers: List[np.ndarray],
///                   output_buffers: List[np.ndarray]) -> List[np.ndarray]
///
/// Args:
///   cache_key: SHA256 hash for JIT cache lookup (32 hex chars = 128 bits)
///   subgraph_json: JSON serialization of the subgraph (empty string if cached)
///   input_buffers: Pre-populated numpy arrays for input data (read-only)
///   output_buffers: Pre-allocated numpy arrays for output data (written by JIT)
///
/// Returns:
///   The output_buffers list (same objects, now filled with JIT results)
std::vector<py::array_t<float>> compile_and_run(
    const std::string& cache_key,
    const std::string& subgraph_json,
    const std::vector<py::array_t<float>>& input_buffers,
    const std::vector<py::array_t<float>>& output_buffers) {

    // Initialize compiler on first call
    if (!g_compiler) {
        g_compiler = new CompilerInstance();
    }

    auto& compiler = *g_compiler;

    // Check if JIT engine is available
    if (!compiler.jit_engine || !compiler.jit_engine->is_ready()) {
        throw std::runtime_error(
            "MLIR JIT engine not available. "
            "Check that LLVM/MLIR development libraries are installed "
            "and the needle_compiler module was built successfully.");
    }

    // Try cache lookup first (if cache_key is provided and subgraph_json is empty)
    if (!cache_key.empty() && subgraph_json.empty()) {
        // Cache hit path: subgraph_json is empty when we have a cache hit.
        // In that case we expect the function to already be in fn_cache.
        // The actual execution happens below.
    }

    // If subgraph_json is provided, parse and compile
    std::string func_name = "compiled_subgraph";
    void* fn_ptr = nullptr;

    if (!subgraph_json.empty()) {
        // Parse subgraph descriptor from JSON
        SubgraphDescriptor desc = parse_subgraph_json(subgraph_json);

        // Generate MLIR module
        auto module = generate_mlir_module(desc, compiler.ctx);

        // Verify the MLIR module
        if (mlir::failed(mlir::verify(*module))) {
            throw std::runtime_error("MLIR module verification failed");
        }

        // Dump MLIR IR if debug enabled
        if (get_mlir_dump_enabled()) {
            module->dump();
        }

        // Compile via JIT
        fn_ptr = compiler.jit_engine->get_cached_or_compile(
            cache_key, *module, func_name);

        if (!fn_ptr) {
            throw std::runtime_error("JIT compilation failed for subgraph");
        }
    } else {
        // Cached path: look up the function pointer from the cache
        fn_ptr = compiler.jit_engine->get_cached_or_compile(
            cache_key, /*module=*/mlir::OwningOpRef<mlir::ModuleOp>(), func_name);
        if (!fn_ptr) {
            throw std::runtime_error("JIT cache miss — subgraph not previously compiled");
        }
    }

    // Execute the JIT function
    // The function signature is:
    //   void compiled_subgraph(float** inputs, int64_t num_inputs,
    //                          float** params, int64_t num_params,
    //                          float** outputs, int64_t num_outputs)
    //
    // For simplicity, we pass raw pointers to the numpy array data.
    // The JIT function writes results directly into output_buffers' memory.

    // Collect input data pointers
    std::vector<const float*> input_ptrs;
    for (auto& buf : input_buffers) {
        input_ptrs.push_back(buf.data());
    }

    // Collect output data pointers (JIT will write here)
    std::vector<float*> output_ptrs;
    for (auto& buf : output_buffers) {
        output_ptrs.push_back(const_cast<float*>(buf.data()));
    }

    // Cast function pointer to the expected signature and call it
    using JITFuncType = void(*)(const float**, int64_t, const float**, int64_t,
                                 float**, int64_t);
    auto jit_fn = reinterpret_cast<JITFuncType>(fn_ptr);

    // Combine inputs and params (params are appended after inputs in the function args)
    // The exact calling convention depends on how the MLIR func.func is generated.
    // For now, we call with all buffers.
    jit_fn(
        input_ptrs.data(), static_cast<int64_t>(input_ptrs.size()),
        input_ptrs.data(), 0,  // params (empty for now — params are handled as inputs)
        output_ptrs.data(), static_cast<int64_t>(output_ptrs.size())
    );

    // Return the output buffers (same objects, now filled)
    return output_buffers;
}

}  // namespace compiler
}  // namespace needle

// ============================================================================
// pybind11 module definition
// ============================================================================

PYBIND11_MODULE(needle_compiler, m) {
    m.doc() = "Needle MLIR Compiler Backend — JIT compilation via MLIR + LLVM ORC";

    using namespace needle::compiler;

    m.def("compile_and_run", &compile_and_run,
          py::arg("cache_key"),
          py::arg("subgraph_json"),
          py::arg("input_buffers"),
          py::arg("output_buffers"),
          "Compile a subgraph to MLIR, JIT compile, and execute.\n\n"
          "Args:\n"
          "  cache_key: SHA256 hash for JIT cache lookup\n"
          "  subgraph_json: JSON serialization (empty if cached)\n"
          "  input_buffers: List of numpy arrays for input data\n"
          "  output_buffers: List of numpy arrays for output data (pre-allocated)\n\n"
          "Returns: output_buffers (same objects, now filled with results)");

    // Environment helpers
    m.def("get_optimization_level", &get_optimization_level,
          "Get the MLIR optimization level from NEEDLE_MLIR_OPT env var");
    m.def("get_mlir_dump_enabled", &get_mlir_dump_enabled,
          "Check if MLIR IR dumping is enabled (NEEDLE_MLIR_DUMP=1)");
    m.def("get_cache_dir", &get_cache_dir,
          "Get the JIT cache directory");
    m.def("get_cache_disabled", &get_cache_disabled,
          "Check if JIT caching is disabled (NEEDLE_MLIR_DISABLE_CACHE=1)");
}
