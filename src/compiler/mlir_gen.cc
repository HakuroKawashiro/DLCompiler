#include "mlir_gen.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

#include <cassert>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include <map>

namespace needle {
namespace compiler {

using namespace mlir;

// ============================================================================
// Helper functions
// ============================================================================

MemRefType make_memref_type(MLIRContext& ctx,
                             const std::vector<int64_t>& shape,
                             Type element_type) {
    return MemRefType::get(llvm::ArrayRef<int64_t>(shape), element_type);
}

Value make_constant_f32(OpBuilder& builder, Location loc, float value) {
    auto attr = builder.getF32FloatAttr(value);
    return builder.create<arith::ConstantOp>(loc, builder.getF32Type(), attr);
}

Value make_constant_i64(OpBuilder& builder, Location loc, int64_t value) {
    auto attr = builder.getI64IntegerAttr(value);
    return builder.create<arith::ConstantOp>(loc, builder.getI64Type(), attr);
}

AffineMap make_identity_map(MLIRContext& ctx, int64_t rank) {
    return AffineMap::getMultiDimIdentityMap(rank, &ctx);
}

/// Get F32 element type for the current context.
static Type f32_type(MLIRContext& ctx) {
    return FloatType::getF32(&ctx);
}

/// Get the rank from a memref type.
static int64_t get_rank(MemRefType memref_type) {
    return memref_type.getShape().size();
}

// ============================================================================
// Element-wise generic helpers
// ============================================================================

/// Create a linalg.generic with given inputs, indexing maps, iterator types,
/// and a lambda for the body builder.
static Value create_linalg_generic(
    OpBuilder& builder, Location loc,
    ValueRange inputs, Value output,
    llvm::ArrayRef<AffineMap> indexing_maps,
    llvm::ArrayRef<utils::IteratorType> iterator_types,
    std::function<void(OpBuilder&, Location, ValueRange)> body_builder) {

    auto generic_op = builder.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/output.getType(),
        /*inputs=*/inputs,
        /*outputs=*/ValueRange{output},
        /*indexingMaps=*/indexing_maps,
        /*iteratorTypes=*/iterator_types,
        /*doc=*/"",
        /*libraryCall=*/"",
        /*bodyBuild=*/body_builder
    );
    return generic_op.getResult(0);
}

// ============================================================================
// Element-wise binary op emitter
// ============================================================================

/// Template for simple binary element-wise ops (add, mul, div, pow)
static Value emit_elementwise_binary(
    OpBuilder& builder, Location loc,
    ValueRange inputs, Type out_type,
    std::function<Value(OpBuilder&, Location, Value, Value)> body_fn) {

    auto out_memref_type = out_type.cast<MemRefType>();
    int64_t rank = get_rank(out_memref_type);

    // Create empty output buffer
    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);

    // All inputs and output share the same identity indexing map
    llvm::SmallVector<AffineMap, 4> maps;
    auto identity = make_identity_map(*builder.getContext(), rank);
    for (size_t i = 0; i < inputs.size(); ++i) {
        maps.push_back(identity);
    }
    maps.push_back(identity);  // output map

    // All parallel iterators
    llvm::SmallVector<utils::IteratorType, 4> iter_types(rank, utils::IteratorType::parallel);

    return create_linalg_generic(builder, loc, inputs, output, maps, iter_types,
        [&](OpBuilder& body, Location body_loc, ValueRange args) {
            // args = [input_elements..., output_element]
            Value lhs = args[0];
            Value rhs = args[1];
            Value result = body_fn(body, body_loc, lhs, rhs);
            body.create<linalg::YieldOp>(body_loc, result);
        });
}

// ============================================================================
// Element-wise operator implementations
// ============================================================================

Value emit_elementwise_add(OpBuilder& builder, Location loc,
                            ValueRange inputs, const NodeDescriptor& node,
                            Type out_type) {
    return emit_elementwise_binary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value a, Value b) -> Value {
            return b.create<arith::AddFOp>(l, a, b);
        });
}

Value emit_elementwise_mul(OpBuilder& builder, Location loc,
                            ValueRange inputs, const NodeDescriptor& node,
                            Type out_type) {
    return emit_elementwise_binary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value a, Value b) -> Value {
            return b.create<arith::MulFOp>(l, a, b);
        });
}

Value emit_elementwise_div(OpBuilder& builder, Location loc,
                            ValueRange inputs, const NodeDescriptor& node,
                            Type out_type) {
    return emit_elementwise_binary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value a, Value b) -> Value {
            return b.create<arith::DivFOp>(l, a, b);
        });
}

Value emit_elementwise_pow(OpBuilder& builder, Location loc,
                            ValueRange inputs, const NodeDescriptor& node,
                            Type out_type) {
    return emit_elementwise_binary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value a, Value b) -> Value {
            return b.create<math::PowFOp>(l, a, b);
        });
}

// ============================================================================
// Unary element-wise ops (ReLU, Tanh, Exp, Log, Negate)
// ============================================================================

static Value emit_elementwise_unary(
    OpBuilder& builder, Location loc,
    ValueRange inputs, Type out_type,
    std::function<Value(OpBuilder&, Location, Value)> body_fn) {

    auto out_memref_type = out_type.cast<MemRefType>();
    int64_t rank = get_rank(out_memref_type);

    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);
    auto identity = make_identity_map(*builder.getContext(), rank);

    llvm::SmallVector<AffineMap, 3> maps = {identity, identity};
    llvm::SmallVector<utils::IteratorType, 4> iter_types(rank, utils::IteratorType::parallel);

    return create_linalg_generic(builder, loc, inputs, output, maps, iter_types,
        [&](OpBuilder& body, Location body_loc, ValueRange args) {
            Value result = body_fn(body, body_loc, args[0]);
            body.create<linalg::YieldOp>(body_loc, result);
        });
}

Value emit_relu(OpBuilder& builder, Location loc,
                ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    return emit_elementwise_unary(builder, loc, inputs, out_type,
        [&](OpBuilder& b, Location l, Value x) -> Value {
            Value zero = make_constant_f32(b, l, 0.0f);
            return b.create<arith::MaxFOp>(l, x, zero);
        });
}

Value emit_tanh(OpBuilder& builder, Location loc,
                ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    return emit_elementwise_unary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value x) -> Value {
            return b.create<math::TanhOp>(l, x);
        });
}

Value emit_exp(OpBuilder& builder, Location loc,
               ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    return emit_elementwise_unary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value x) -> Value {
            return b.create<math::ExpOp>(l, x);
        });
}

Value emit_log(OpBuilder& builder, Location loc,
               ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    return emit_elementwise_unary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value x) -> Value {
            return b.create<math::LogOp>(l, x);
        });
}

Value emit_negate(OpBuilder& builder, Location loc,
                  ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    return emit_elementwise_unary(builder, loc, inputs, out_type,
        [](OpBuilder& b, Location l, Value x) -> Value {
            return b.create<arith::NegFOp>(l, x);
        });
}

// ============================================================================
// Scalar ops (AddScalar, MulScalar, DivScalar, PowerScalar)
// ============================================================================

static Value emit_elementwise_with_scalar(
    OpBuilder& builder, Location loc,
    Value input, Type out_type, float scalar_val,
    std::function<Value(OpBuilder&, Location, Value, Value)> body_fn) {

    auto out_memref_type = out_type.cast<MemRefType>();
    int64_t rank = get_rank(out_memref_type);

    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);
    auto identity = make_identity_map(*builder.getContext(), rank);

    llvm::SmallVector<AffineMap, 3> maps = {identity, identity};
    llvm::SmallVector<utils::IteratorType, 4> iter_types(rank, utils::IteratorType::parallel);

    return create_linalg_generic(builder, loc, ValueRange{input}, output, maps, iter_types,
        [&](OpBuilder& body, Location body_loc, ValueRange args) {
            Value scalar = make_constant_f32(body, body_loc, scalar_val);
            Value result = body_fn(body, body_loc, args[0], scalar);
            body.create<linalg::YieldOp>(body_loc, result);
        });
}

Value emit_scalar_add(OpBuilder& builder, Location loc,
                       ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    float scalar = 0.0f;
    auto it = node.attrs.find("scalar");
    if (it != node.attrs.end()) scalar = static_cast<float>(it->second);
    return emit_elementwise_with_scalar(builder, loc, inputs[0], out_type, scalar,
        [](OpBuilder& b, Location l, Value x, Value s) -> Value {
            return b.create<arith::AddFOp>(l, x, s);
        });
}

Value emit_scalar_mul(OpBuilder& builder, Location loc,
                       ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    float scalar = 1.0f;
    auto it = node.attrs.find("scalar");
    if (it != node.attrs.end()) scalar = static_cast<float>(it->second);
    return emit_elementwise_with_scalar(builder, loc, inputs[0], out_type, scalar,
        [](OpBuilder& b, Location l, Value x, Value s) -> Value {
            return b.create<arith::MulFOp>(l, x, s);
        });
}

Value emit_div_scalar(OpBuilder& builder, Location loc,
                       ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    float scalar = 1.0f;
    auto it = node.attrs.find("scalar");
    if (it != node.attrs.end()) scalar = static_cast<float>(it->second);
    return emit_elementwise_with_scalar(builder, loc, inputs[0], out_type, scalar,
        [](OpBuilder& b, Location l, Value x, Value s) -> Value {
            return b.create<arith::DivFOp>(l, x, s);
        });
}

Value emit_power_scalar(OpBuilder& builder, Location loc,
                         ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    float exponent = 2.0f;
    auto it = node.attrs.find("scalar");
    if (it != node.attrs.end()) exponent = static_cast<float>(it->second);

    return emit_elementwise_with_scalar(builder, loc, inputs[0], out_type, exponent,
        [exponent](OpBuilder& b, Location l, Value x, Value s) -> Value {
            // Special case: exponent == 2 → x * x (BN variance)
            if (exponent == 2.0f) {
                return b.create<arith::MulFOp>(l, x, x);
            }
            // General case: math.powf
            return b.create<math::PowFOp>(l, x, s);
        });
}

// ============================================================================
// Named ops: MatMul, Conv
// ============================================================================

Value emit_matmul(OpBuilder& builder, Location loc,
                  ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    // inputs: [A, B] → C = A @ B
    // linalg.matmul ins(%A, %B : memref<m×n×f32>, memref<n×p×f32>)
    //               outs(%C : memref<m×p×f32>)
    Value output = builder.create<memref::AllocOp>(loc, out_type.cast<MemRefType>());

    auto matmul_op = builder.create<linalg::MatmulOp>(
        loc,
        /*inputs=*/ValueRange{inputs[0], inputs[1]},
        /*outputs=*/ValueRange{output}
    );
    return matmul_op.getResult(0);
}

Value emit_conv(OpBuilder& builder, Location loc,
                ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    // inputs: [X, W] where X is NHWC, W is HWCF (Kh×Kw×Cin×Cout)
    // linalg.conv_2d_nhwc_hwcf
    //
    // Note: dilations default to dense<1>. If Needle supports dilated conv in the
    // future, read dilation from node.attrs.

    auto out_memref_type = out_type.cast<MemRefType>();
    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);

    // Extract stride and dilation from attrs
    int64_t stride = 1;
    auto stride_it = node.attrs.find("stride");
    if (stride_it != node.attrs.end()) stride = stride_it->second;

    // Default dilations = [1, 1] (no dilation)
    // Current Needle Conv does not support dilation — this is a placeholder for future extension
    llvm::SmallVector<int64_t, 2> dilations = {1, 1};
    auto dil_it = node.attrs.find("dilation");
    if (dil_it != node.attrs.end()) {
        dilations[0] = dil_it->second;
        dilations[1] = dil_it->second;
    }

    llvm::SmallVector<int64_t, 2> strides = {stride, stride};

    auto conv_op = builder.create<linalg::Conv2DNhwcHwcfOp>(
        loc,
        out_memref_type,
        /*inputs=*/ValueRange{inputs[0], inputs[1]},
        /*outputs=*/ValueRange{output},
        /*strides=*/builder.getI64VectorAttr(strides),
        /*dilations=*/builder.getI64VectorAttr(dilations)
    );
    return conv_op.getResult(0);
}

// ============================================================================
// Data movement ops
// ============================================================================

Value emit_broadcast(OpBuilder& builder, Location loc,
                     ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    // linalg.generic where input indexing map has 0 constants on broadcast dims
    auto out_memref_type = out_type.cast<MemRefType>();
    auto in_memref_type = inputs[0].getType().cast<MemRefType>();
    int64_t out_rank = get_rank(out_memref_type);
    int64_t in_rank = get_rank(in_memref_type);

    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);

    // If input rank < output rank, we need to handle rank alignment.
    // The input indexing map should have 0 in broadcast dimensions.
    // For simplicity, we use identity for both and let MLIR bufferization handle
    // the broadcast via a copy (which is correct but not zero-copy).
    //
    // A more optimal approach would construct the correct affine_map with 0s
    // on broadcast dims, but this requires the shape info from attrs.

    auto out_identity = make_identity_map(*builder.getContext(), out_rank);

    // Construct input map: for dims where input_shape[i] == 1 and output_shape[i] != 1,
    // use constant 0 instead of dim expr.
    llvm::SmallVector<AffineExpr, 4> in_exprs;
    int64_t rank_diff = out_rank - in_rank;
    for (int64_t i = 0; i < rank_diff; ++i) {
        // Leading dimensions that don't exist in input → constant 0
        in_exprs.push_back(mlir::getAffineConstantExpr(0, builder.getContext()));
    }
    for (int64_t i = 0; i < in_rank; ++i) {
        int64_t out_dim = rank_diff + i;
        if (in_memref_type.getShape()[i] == 1 &&
            i < static_cast<int64_t>(out_memref_type.getShape().size()) &&
            out_memref_type.getShape()[out_dim] != 1) {
            in_exprs.push_back(mlir::getAffineConstantExpr(0, builder.getContext()));
        } else {
            in_exprs.push_back(mlir::getAffineDimExpr(out_dim, builder.getContext()));
        }
    }
    auto in_map = AffineMap::get(out_rank, 0, in_exprs, builder.getContext());

    llvm::SmallVector<AffineMap, 3> maps = {in_map, out_identity};
    llvm::SmallVector<utils::IteratorType, 4> iter_types(out_rank, utils::IteratorType::parallel);

    return create_linalg_generic(builder, loc, inputs, output, maps, iter_types,
        [](OpBuilder& body, Location body_loc, ValueRange args) {
            body.create<linalg::YieldOp>(body_loc, args[0]);
        });
}

Value emit_reshape(OpBuilder& builder, Location loc,
                   ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    // If the source memref is contiguous, use memref.reinterpret_cast (zero-copy).
    // Otherwise, allocate new memref and copy.
    //
    // Note: the Python side should pass is_contiguous flag in attrs.
    // For now, we always use alloc + copy (safe but may be suboptimal).
    auto out_memref_type = out_type.cast<MemRefType>();
    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);
    builder.create<memref::CopyOp>(loc, inputs[0], output);
    return output;
}

Value emit_transpose(OpBuilder& builder, Location loc,
                     ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    auto out_memref_type = out_type.cast<MemRefType>();
    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);

    // Extract permutation from attrs
    llvm::SmallVector<int64_t, 4> permutation;
    auto perm_it = node.attrs.find("axes");
    if (perm_it != node.attrs.end()) {
        // Parse axes from attrs — for now use default last-two-dims swap
        int64_t rank = get_rank(out_memref_type);
        for (int64_t i = 0; i < rank; ++i) permutation.push_back(i);
        if (rank >= 2) {
            std::swap(permutation[rank - 1], permutation[rank - 2]);
        }
    }

    auto transpose_op = builder.create<linalg::TransposeOp>(
        loc,
        inputs[0],
        output,
        builder.getI64VectorAttr(permutation)
    );
    return transpose_op.getResult(0);
}

// ============================================================================
// Reduction ops
// ============================================================================

Value emit_summation(OpBuilder& builder, Location loc,
                     ValueRange inputs, const NodeDescriptor& node, Type out_type) {
    auto in_memref_type = inputs[0].getType().cast<MemRefType>();
    auto out_memref_type = out_type.cast<MemRefType>();
    int64_t rank = get_rank(in_memref_type);

    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);

    // Determine which axes are reduced
    llvm::SmallVector<int64_t, 4> reduce_axes;
    auto axes_it = node.attrs.find("axes");
    // If axes not specified, reduce all (scalar output)

    // Build iterator types: "parallel" for kept dims, "reduction" for reduced dims
    llvm::SmallVector<utils::IteratorType, 4> iter_types;
    llvm::SmallVector<AffineExpr, 4> in_exprs;
    llvm::SmallVector<AffineExpr, 4> out_exprs;

    // Default: reduce all axes to scalar
    // For simplicity, we reduce all dims (scalar sum)
    for (int64_t i = 0; i < rank; ++i) {
        iter_types.push_back(utils::IteratorType::reduction);
        in_exprs.push_back(mlir::getAffineDimExpr(i, builder.getContext()));
    }
    // Output map: scalar → no dims
    auto out_map = AffineMap::get(rank, 0, {}, builder.getContext());
    auto in_map = AffineMap::get(rank, 0, in_exprs, builder.getContext());

    llvm::SmallVector<AffineMap, 3> maps = {in_map, out_map};

    return create_linalg_generic(builder, loc, inputs, output, maps, iter_types,
        [](OpBuilder& body, Location body_loc, ValueRange args) {
            Value result = body.create<arith::AddFOp>(body_loc, args[0], args[1]);
            body.create<linalg::YieldOp>(body_loc, result);
        });
}

// ============================================================================
// Fused element-wise chain emitter
// ============================================================================

Value emit_fused_elementwise(
    OpBuilder& builder, Location loc,
    const std::vector<NodeDescriptor>& nodes,
    size_t start_idx, size_t end_idx,
    const std::vector<Value>& node_values,
    Type out_type) {

    auto out_memref_type = out_type.cast<MemRefType>();
    int64_t rank = get_rank(out_memref_type);

    // ── Phase 1: Classify inputs as internal or external ──────────────
    // Internal indices: node indices within the fusion group [start_idx, end_idx)
    // External indices: everything else (function arguments)
    std::set<size_t> internal_set;
    for (size_t i = start_idx; i < end_idx; ++i)
        internal_set.insert(i);

    // Deduplicate external inputs and assign body_arg positions
    std::map<size_t, size_t> ext_to_body_arg;  // external index → body_arg position
    llvm::SmallVector<Value, 8> ext_input_vals;

    for (size_t i = start_idx; i < end_idx; ++i) {
        for (size_t inp_idx : nodes[i].input_indices) {
            if (internal_set.count(inp_idx) == 0) {
                // External input — deduplicate
                if (ext_to_body_arg.count(inp_idx) == 0) {
                    size_t pos = ext_input_vals.size();
                    ext_to_body_arg[inp_idx] = pos;
                    if (inp_idx < node_values.size())
                        ext_input_vals.push_back(node_values[inp_idx]);
                }
            }
        }
    }

    // ── Phase 2: Allocate output and build indexing maps ──────────────
    Value output = builder.create<memref::AllocOp>(loc, out_memref_type);
    auto identity = make_identity_map(*builder.getContext(), rank);

    llvm::SmallVector<AffineMap, 8> maps;
    for (size_t i = 0; i < ext_input_vals.size(); ++i)
        maps.push_back(identity);
    maps.push_back(identity);  // output map

    llvm::SmallVector<utils::IteratorType, 4> iter_types(rank, utils::IteratorType::parallel);

    // ── Phase 3: Build fused body — chain all ops in topo order ───────
    return create_linalg_generic(builder, loc, ext_input_vals, output, maps, iter_types,
        [&](OpBuilder& body, Location body_loc, ValueRange body_args) {
            // Map: internal node index → its SSA element value
            std::map<size_t, Value> internal_vals;

            for (size_t i = start_idx; i < end_idx; ++i) {
                const auto& nd = nodes[i];

                // Resolve operand values for this node
                llvm::SmallVector<Value, 4> operands;
                for (size_t inp_idx : nd.input_indices) {
                    if (internal_set.count(inp_idx)) {
                        // Internal — use previously computed value
                        auto it = internal_vals.find(inp_idx);
                        if (it != internal_vals.end())
                            operands.push_back(it->second);
                    } else {
                        // External — use body_args
                        auto it = ext_to_body_arg.find(inp_idx);
                        if (it != ext_to_body_arg.end())
                            operands.push_back(body_args[it->second]);
                    }
                }

                if (operands.empty()) {
                    // No operands resolved — use zero constant as fallback
                    internal_vals[i] = make_constant_f32(body, body_loc, 0.0f);
                    continue;
                }

                // Dispatch body operation
                Value result;
                if (nd.op_name == "EWiseAdd") {
                    result = body.create<arith::AddFOp>(body_loc, operands[0], operands[1]);
                } else if (nd.op_name == "EWiseMul") {
                    result = body.create<arith::MulFOp>(body_loc, operands[0], operands[1]);
                } else if (nd.op_name == "EWiseDiv") {
                    result = body.create<arith::DivFOp>(body_loc, operands[0], operands[1]);
                } else if (nd.op_name == "EWisePow") {
                    result = body.create<math::PowFOp>(body_loc, operands[0], operands[1]);
                } else if (nd.op_name == "ReLU") {
                    Value zero = make_constant_f32(body, body_loc, 0.0f);
                    result = body.create<arith::MaxFOp>(body_loc, operands[0], zero);
                } else if (nd.op_name == "Tanh") {
                    result = body.create<math::TanhOp>(body_loc, operands[0]);
                } else if (nd.op_name == "Exp") {
                    result = body.create<math::ExpOp>(body_loc, operands[0]);
                } else if (nd.op_name == "Log") {
                    result = body.create<math::LogOp>(body_loc, operands[0]);
                } else if (nd.op_name == "Negate") {
                    result = body.create<arith::NegFOp>(body_loc, operands[0]);
                } else if (nd.op_name == "AddScalar") {
                    float s = 0.0f;
                    auto it = nd.attrs.find("scalar");
                    if (it != nd.attrs.end()) s = static_cast<float>(it->second);
                    Value scalar = make_constant_f32(body, body_loc, s);
                    result = body.create<arith::AddFOp>(body_loc, operands[0], scalar);
                } else if (nd.op_name == "MulScalar") {
                    float s = 1.0f;
                    auto it = nd.attrs.find("scalar");
                    if (it != nd.attrs.end()) s = static_cast<float>(it->second);
                    Value scalar = make_constant_f32(body, body_loc, s);
                    result = body.create<arith::MulFOp>(body_loc, operands[0], scalar);
                } else if (nd.op_name == "DivScalar") {
                    float s = 1.0f;
                    auto it = nd.attrs.find("scalar");
                    if (it != nd.attrs.end()) s = static_cast<float>(it->second);
                    Value scalar = make_constant_f32(body, body_loc, s);
                    result = body.create<arith::DivFOp>(body_loc, operands[0], scalar);
                } else if (nd.op_name == "PowerScalar") {
                    float exp = 2.0f;
                    auto it = nd.attrs.find("scalar");
                    if (it != nd.attrs.end()) exp = static_cast<float>(it->second);
                    if (exp == 2.0f) {
                        // x² → x * x (no math.powf needed)
                        result = body.create<arith::MulFOp>(body_loc, operands[0], operands[0]);
                    } else {
                        Value scalar = make_constant_f32(body, body_loc, exp);
                        result = body.create<math::PowFOp>(body_loc, operands[0], scalar);
                    }
                } else if (nd.op_name == "BroadcastTo") {
                    // Within fusion group: element passthrough
                    // (indexing map handles broadcast dim alignment)
                    result = operands[0];
                } else {
                    // Unknown op in fusion group — passthrough first operand
                    result = operands[0];
                }

                internal_vals[i] = result;
            }

            // Yield the final node's result
            body.create<linalg::YieldOp>(body_loc, internal_vals[end_idx - 1]);
        });
}

// ============================================================================
// Dispatch table
// ============================================================================

std::map<std::string, EmitFn> EMIT_FN_TABLE = {
    {"EWiseAdd", emit_elementwise_add},
    {"EWiseMul", emit_elementwise_mul},
    {"EWiseDiv", emit_elementwise_div},
    {"EWisePow", emit_elementwise_pow},
    {"ReLU", emit_relu},
    {"Tanh", emit_tanh},
    {"Exp", emit_exp},
    {"Log", emit_log},
    {"Negate", emit_negate},
    {"AddScalar", emit_scalar_add},
    {"MulScalar", emit_scalar_mul},
    {"DivScalar", emit_div_scalar},
    {"PowerScalar", emit_power_scalar},
    {"MatMul", emit_matmul},
    {"Conv", emit_conv},
    {"BroadcastTo", emit_broadcast},
    {"Reshape", emit_reshape},
    {"Transpose", emit_transpose},
    {"Summation", emit_summation},
};

// ============================================================================
// Main entry: generate_mlir_module
// ============================================================================

OwningOpRef<ModuleOp> generate_mlir_module(const SubgraphDescriptor& desc,
                                             MLIRContext& ctx) {
    auto loc = UnknownLoc::get(&ctx);
    OpBuilder builder(&ctx);

    // Create top-level module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create func.func with proper argument types
    llvm::SmallVector<Type, 8> arg_types;
    for (auto& shape : desc.input_shapes) {
        arg_types.push_back(make_memref_type(ctx, shape, f32_type(ctx)));
    }
    for (auto& shape : desc.param_shapes) {
        arg_types.push_back(make_memref_type(ctx, shape, f32_type(ctx)));
    }

    auto func_type = builder.getFunctionType(arg_types, /*resultTypes=*/{});
    auto func = builder.create<func::FuncOp>(loc, "compiled_subgraph", func_type);

    // Set up the function body
    auto& entry_block = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry_block);

    // Map from node index to its produced Value
    std::vector<Value> node_values(desc.nodes.size());

    // Process nodes in order
    for (size_t i = 0; i < desc.nodes.size(); ++i) {
        const auto& node = desc.nodes[i];

        // Check if this node is part of a fused group
        bool in_fused_group = false;
        size_t fused_end = 0;
        for (auto& fg : desc.fused_groups) {
            if (i >= fg.first && i < fg.second) {
                in_fused_group = true;
                fused_end = fg.second;
                break;
            }
        }

        if (in_fused_group && i == desc.fused_groups[0].first) {
            // Emit the entire fused group as one linalg.generic
            size_t group_start = i;
            // Find the matching fused group
            for (auto& fg : desc.fused_groups) {
                if (fg.first == i) {
                    fused_end = fg.second;
                    break;
                }
            }

            // Collect input Values for this group
            llvm::SmallVector<Value, 8> group_inputs;
            for (size_t j = group_start; j < fused_end; ++j) {
                for (auto inp_idx : desc.nodes[j].input_indices) {
                    if (inp_idx < i) {
                        group_inputs.push_back(node_values[inp_idx]);
                    } else {
                        // External input
                        group_inputs.push_back(entry_block.getArgument(inp_idx));
                    }
                }
            }

            if (group_inputs.empty()) {
                // Use function arguments as inputs
                for (size_t a = 0; a < entry_block.getNumArguments(); ++a) {
                    group_inputs.push_back(entry_block.getArgument(a));
                }
            }

            auto last_node_type = make_memref_type(ctx,
                desc.nodes[fused_end - 1].output_shape, f32_type(ctx));

            Value fused_result = emit_fused_elementwise(
                builder, loc, desc.nodes, group_start, fused_end,
                std::vector<Value>(group_inputs.begin(), group_inputs.end()),
                last_node_type);

            // Set result for all nodes in the fused group
            for (size_t j = group_start; j < fused_end; ++j) {
                node_values[j] = fused_result;
            }
            i = fused_end - 1;  // Skip processed nodes
            continue;
        }

        // Single node emit
        auto it = EMIT_FN_TABLE.find(node.op_name);
        if (it == EMIT_FN_TABLE.end()) {
            llvm::report_fatal_error("Unknown op in MLIR generation: " + node.op_name);
        }

        // Collect input Values
        llvm::SmallVector<Value, 4> input_vals;
        for (auto inp_idx : node.input_indices) {
            if (inp_idx < node_values.size()) {
                input_vals.push_back(node_values[inp_idx]);
            } else {
                // External input (function argument)
                size_t arg_idx = inp_idx - desc.nodes.size();
                if (arg_idx < entry_block.getNumArguments()) {
                    input_vals.push_back(entry_block.getArgument(arg_idx));
                }
            }
        }

        auto out_memref_type = make_memref_type(ctx, node.output_shape, f32_type(ctx));
        node_values[i] = it->second(builder, loc, input_vals, node, out_memref_type);
    }

    // Build return: return only the output nodes
    llvm::SmallVector<Value, 4> return_vals;
    for (auto out_idx : desc.output_node_indices) {
        if (out_idx < node_values.size()) {
            return_vals.push_back(node_values[out_idx]);
        }
    }

    builder.create<func::ReturnOp>(loc, return_vals);

    return module;
}

}  // namespace compiler
}  // namespace needle
