#pragma once

#include "needle_compiler.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Location.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <functional>
#include <vector>
#include <map>
#include <string>

namespace needle {
namespace compiler {

/// Type alias for emit functions.
/// Each emit function takes: OpBuilder, Location, input Values, NodeDescriptor, output Type
/// Returns the computed Value.
using EmitFn = std::function<mlir::Value(
    mlir::OpBuilder&,
    mlir::Location,
    mlir::ValueRange,
    const NodeDescriptor&,
    mlir::Type)>;

// ============================================================================
// Main entry point
// ============================================================================

/// Generate a complete MLIR Module from a SubgraphDescriptor.
/// Creates a func.func with proper argument types, emits each node's MLIR ops,
/// and returns the result values.
mlir::OwningOpRef<mlir::ModuleOp>
generate_mlir_module(const SubgraphDescriptor& desc, mlir::MLIRContext& ctx);

// ============================================================================
// Element-wise operator emit functions
// All produce linalg.generic with "parallel" iterator_types
// ============================================================================

mlir::Value emit_elementwise_add(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_elementwise_mul(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_elementwise_div(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_relu(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_tanh(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_exp(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_log(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_negate(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_scalar_add(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_scalar_mul(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_div_scalar(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_power_scalar(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_elementwise_pow(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

// ============================================================================
// Named linear algebra ops (linalg.matmul, linalg.conv_2d)
// ============================================================================

mlir::Value emit_matmul(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_conv(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

// ============================================================================
// Data movement ops
// ============================================================================

mlir::Value emit_broadcast(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_reshape(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

mlir::Value emit_transpose(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

// ============================================================================
// Reduction ops
// ============================================================================

mlir::Value emit_summation(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ValueRange inputs, const NodeDescriptor& node, mlir::Type out_type);

// ============================================================================
// Fusion support
// ============================================================================

/// Emit a fused element-wise chain as a single linalg.generic.
/// The nodes in [start_idx, end_idx) are all FUSIBLE_ELEMENTWISE_OPS.
/// Their bodies are merged into one linalg.generic region.
mlir::Value emit_fused_elementwise(
    mlir::OpBuilder& builder, mlir::Location loc,
    const std::vector<NodeDescriptor>& nodes,
    size_t start_idx, size_t end_idx,
    const std::vector<mlir::Value>& node_values,
    mlir::Type out_type);

// ============================================================================
// Utility functions
// ============================================================================

/// Create a memref type with the given shape and element type.
mlir::MemRefType make_memref_type(
    mlir::MLIRContext& ctx,
    const std::vector<int64_t>& shape,
    mlir::Type element_type);

/// Create a constant float value.
mlir::Value make_constant_f32(mlir::OpBuilder& builder, mlir::Location loc, float value);

/// Create a constant integer value.
mlir::Value make_constant_i64(mlir::OpBuilder& builder, mlir::Location loc, int64_t value);

/// Generate identity AffineMap for a given rank.
mlir::AffineMap make_identity_map(mlir::MLIRContext& ctx, int64_t rank);

/// Dispatch table: op_name -> emit function.
extern std::map<std::string, EmitFn> EMIT_FN_TABLE;

}  // namespace compiler
}  // namespace needle
