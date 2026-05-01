#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace cuda_tile {
namespace cpu {

#define GEN_PASS_DEF_VECTORIZELINALGPASS
#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

using namespace mlir;

namespace {

struct VectorizeLinalgOpPattern : public RewritePattern {
  VectorizeLinalgOpPattern(MLIRContext *context, bool vectorizeNDExtract,
                           bool flatten1DDepthwiseConv)
      : RewritePattern{MatchAnyOpTypeTag(), /*benefit=*/1, context},
        vectorizeNDExtract{vectorizeNDExtract},
        flatten1DDepthwiseConv{flatten1DDepthwiseConv} {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::hasVectorizationImpl(op)) {
      return failure();
    }

    FailureOr<linalg::VectorizationResult> vectorized =
        linalg::vectorize(rewriter, op, /*inputVectorSizes=*/{},
                          /*inputScalableVecDims=*/{}, vectorizeNDExtract,
                          flatten1DDepthwiseConv);
    if (failed(vectorized)) {
      return failure();
    }

    rewriter.replaceOp(op, vectorized->replacements);
    return success();
  }

private:
  bool vectorizeNDExtract;
  bool flatten1DDepthwiseConv;
};

struct VectorizeLinalgPass
    : public cuda_tile::cpu::impl::VectorizeLinalgPassBase<
          VectorizeLinalgPass> {
  using VectorizeLinalgPassBase::VectorizeLinalgPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeLinalgOpPattern>(&getContext(), vectorizeNDExtract,
                                           flatten1DDepthwiseConv);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace
