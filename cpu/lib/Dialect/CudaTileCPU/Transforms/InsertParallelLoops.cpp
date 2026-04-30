#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace cuda_tile {
namespace cpu {

#define GEN_PASS_DEF_INSERTPARALLELLOOPSPASS
#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

using namespace mlir;
using namespace cuda_tile;
using namespace llvm;

namespace {

static Value idxToTensorI32(PatternRewriter &rewriter, Location loc,
                            Value idxVal) {
  auto tensorTy = RankedTensorType::get({}, rewriter.getI32Type());
  Value castOp =
      arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(), idxVal);
  return tensor::FromElementsOp::create(rewriter, loc, tensorTy, castOp);
}

struct GetBlockIdPattern : public OpRewritePattern<cpu::GetBlockIdOp> {
  using OpRewritePattern::OpRewritePattern;

  GetBlockIdPattern(MLIRContext *context, Value ivX, Value ivY, Value ivZ)
      : OpRewritePattern(context), ivX(ivX), ivY(ivY), ivZ(ivZ) {}

  LogicalResult matchAndRewrite(cpu::GetBlockIdOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {
                               idxToTensorI32(rewriter, op.getLoc(), ivX),
                               idxToTensorI32(rewriter, op.getLoc(), ivY),
                               idxToTensorI32(rewriter, op.getLoc(), ivZ),
                           });
    return success();
  }

private:
  Value ivX, ivY, ivZ;
};

struct GetNumBlocksPattern : public OpRewritePattern<cpu::GetNumBlocksOp> {
  using OpRewritePattern::OpRewritePattern;

  GetNumBlocksPattern(MLIRContext *context, OpFoldResult bX, OpFoldResult bY,
                      OpFoldResult bZ)
      : OpRewritePattern(context), boundX(bX), boundY(bY), boundZ(bZ) {}

  LogicalResult matchAndRewrite(cpu::GetNumBlocksOp op,
                                PatternRewriter &rewriter) const override {
    auto boundVal = [&](OpFoldResult ofr) -> Value {
      // the upper bounds of the loop come from the function arguments
      // so this will always get the value
      return getValueOrCreateConstantIndexOp(rewriter, op.getLoc(), ofr);
    };

    rewriter.replaceOp(
        op, {idxToTensorI32(rewriter, op.getLoc(), boundVal(boundX)),
             idxToTensorI32(rewriter, op.getLoc(), boundVal(boundY)),
             idxToTensorI32(rewriter, op.getLoc(), boundVal(boundZ))});
    return success();
  }

private:
  OpFoldResult boundX, boundY, boundZ;
};

struct InsertParallelLoopsPass
    : public cuda_tile::cpu::impl::InsertParallelLoopsPassBase<
          InsertParallelLoopsPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();
    OpBuilder builder{context};

    // TODO: just run this pass on func op instead of module op
    for (auto funcOp : mod.getOps<func::FuncOp>()) {
      Location loc = funcOp.getLoc();

      // add gridX, gridY, gridZ into func args
      auto oldFuncType = funcOp.getFunctionType();
      SmallVector<Type> newArgTypes{oldFuncType.getInputs()};
      newArgTypes.emplace_back(builder.getIndexType());
      newArgTypes.emplace_back(builder.getIndexType());
      newArgTypes.emplace_back(builder.getIndexType());

      funcOp.setType(
          FunctionType::get(context, newArgTypes, oldFuncType.getResults()));

      // add the new func args to the function block args
      Block &entryBlock = funcOp.getBody().front();
      auto gridX = entryBlock.addArgument(builder.getIndexType(), loc);
      auto gridY = entryBlock.addArgument(builder.getIndexType(), loc);
      auto gridZ = entryBlock.addArgument(builder.getIndexType(), loc);

      builder.setInsertionPointToStart(&entryBlock);
      auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
      auto one = arith::ConstantIndexOp::create(builder, loc, 1);

      // create scf.parallel loop over grid
      auto parallelOp =
          scf::ParallelOp::create(builder, loc, {zero, zero, zero},
                                  {gridX, gridY, gridZ}, {one, one, one});

      // move ops of func into the parallel loop
      Block *parallelBody = parallelOp.getBody();
      auto &parallelOps = parallelBody->getOperations();
      auto &funcOps = entryBlock.getOperations();

      parallelOps.splice(std::prev(parallelOps.end()), funcOps,
                         std::next(parallelOp->getIterator()),
                         std::prev(funcOps.end()));

      auto ivs = parallelOp.getLoopInductionVars().value();
      auto bounds = parallelOp.getLoopUpperBounds().value();

      RewritePatternSet patterns{context};
      patterns.add<GetBlockIdPattern>(context, ivs[0], ivs[1], ivs[2]);
      patterns.add<GetNumBlocksPattern>(context, bounds[0], bounds[1],
                                        bounds[2]);

      if (failed(applyPatternsGreedily(parallelOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir {
namespace cuda_tile {
namespace cpu {

std::unique_ptr<::mlir::Pass> createInsertParallelLoops() {
  return std::make_unique<InsertParallelLoopsPass>();
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir
