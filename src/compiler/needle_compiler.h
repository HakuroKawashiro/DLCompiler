#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>
#include <cstddef>

namespace needle {
namespace compiler {

/// Represents a single operator node in a subgraph.
/// Mirrors the Python-side GraphNode serialization.
struct NodeDescriptor {
    std::string op_name;                        // "EWiseAdd", "MatMul", etc.
    std::vector<size_t> input_indices;          // Indices into the nodes array
    std::map<std::string, int64_t> attrs;       // Operator attributes (stride, padding, axes, etc.)
    std::vector<int64_t> output_shape;          // Output tensor shape
    std::string dtype;                          // "float32"
};

/// Represents a subgraph that can be compiled to MLIR.
/// Mirrors the Python-side Subgraph serialization (serialize.py).
struct SubgraphDescriptor {
    std::vector<NodeDescriptor> nodes;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int64_t>> param_shapes;
    std::vector<std::string> param_names;
    size_t num_outputs;
    // Element-wise fusion group intervals: (start_idx, end_idx) in nodes array
    std::vector<std::pair<size_t, size_t>> fused_groups;
    // Indices of nodes that produce outputs
    std::vector<size_t> output_node_indices;
    // Map from param name to output index for mutation write-back
    std::map<std::string, size_t> mutation_map;
};

/// Environment variable: optimization level (0-3, default 2)
int get_optimization_level();

/// Environment variable: dump MLIR IR to stderr
bool get_mlir_dump_enabled();

/// Environment variable: cache directory path
std::string get_cache_dir();

/// Environment variable: disable JIT cache
bool get_cache_disabled();

}  // namespace compiler
}  // namespace needle
