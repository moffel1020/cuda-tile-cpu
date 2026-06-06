#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace cuda_tile {
namespace cpu {

#define GEN_PASS_DEF_TILEANDFUSEINTOSTOREPASS
#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

using namespace mlir;
using namespace cuda_tile;
using namespace llvm;

namespace {

static SmallVector<OpFoldResult>
getTileSizesForStore(OpBuilder &b, TilingInterface storeOp,
                     ArrayRef<int64_t> requestedSizes,
                     ArrayRef<int64_t> opStaticSizes) {
  const int64_t defaultTileSize = 1;

  SmallVector<Range> domain = storeOp.getIterationDomain(b);
  SmallVector<OpFoldResult> tileSizeOfrs;
  tileSizeOfrs.reserve(domain.size());

  for (size_t i = 0; i < domain.size(); i++) {
    auto reqStart = static_cast<int64_t>(domain.size()) -
                    static_cast<int64_t>(requestedSizes.size());
    if (static_cast<int64_t>(i) >= reqStart) {
      tileSizeOfrs.push_back(b.getIndexAttr(
          std::min(requestedSizes[i - reqStart], opStaticSizes[i])));
    } else {
      tileSizeOfrs.push_back(
          b.getIndexAttr(std::min(defaultTileSize, opStaticSizes[i])));
    }
  }

  return tileSizeOfrs;
}

static void
appendGeneratedSlices(SmallVectorImpl<tensor::ExtractSliceOp> &worklist,
                      ArrayRef<Operation *> generatedSlices) {
  for (Operation *slice : generatedSlices) {
    if (auto extractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(slice)) {
      worklist.push_back(extractSlice);
    }
  }
}

static void fuseProducersGreedily(IRRewriter &rewriter,
                                  SmallVector<LoopLikeOpInterface> &loops,
                                  ArrayRef<Operation *> initialSlices) {
  SmallVector<tensor::ExtractSliceOp> worklist;
  appendGeneratedSlices(worklist, initialSlices);

  while (!worklist.empty()) {
    tensor::ExtractSliceOp candidateSlice = worklist.pop_back_val();
    if (!candidateSlice || candidateSlice.use_empty()) {
      continue;
    }

    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSlice, loops);
    if (!fusedResult) {
      continue;
    }

    appendGeneratedSlices(worklist, fusedResult->generatedSlices);
    if (candidateSlice.use_empty()) {
      rewriter.eraseOp(candidateSlice);
    }
  }
}

static std::optional<ArrayRef<int64_t>>
getStaticShapeOfTileableOp(Operation *op) {
  auto storePtr = dyn_cast<cpu::StorePtrTileOp>(op);
  if (storePtr && storePtr.getValue().getType().hasStaticShape()) {
    return storePtr.getValue().getType().getShape();
  }

  auto storeTile = dyn_cast<cpu::StoreMemRefTileOp>(op);
  if (storeTile && storeTile.getValue().getType().hasStaticShape()) {
    return storeTile.getValue().getType().getShape();
  }

  return std::nullopt;
}

struct TileAndFuseIntoStorePass
    : public cuda_tile::cpu::impl::TileAndFuseIntoStorePassBase<
          TileAndFuseIntoStorePass> {
  using TileAndFuseIntoStorePassBase::TileAndFuseIntoStorePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();
    IRRewriter rewriter(context);

    SmallVector<Operation *> sinkOps;
    SmallVector<ArrayRef<int64_t>> sinkStaticShapes;
    mod.walk([&](Operation *op) {
      if (auto shape = getStaticShapeOfTileableOp(op)) {
        sinkOps.push_back(op);
        sinkStaticShapes.push_back(*shape);
      }
    });

    for (auto [op, staticShape] : llvm::zip(sinkOps, sinkStaticShapes)) {
      if (!op->getBlock()) {
        continue;
      }

      auto tileableOp = dyn_cast<TilingInterface>(op);
      if (!tileableOp) {
        op->emitOpError("op does not implement TilingInterface");
        signalPassFailure();
        return;
      }

      rewriter.setInsertionPoint(op);
      SmallVector<OpFoldResult> tileSizeOfrs =
          getTileSizesForStore(rewriter, tileableOp, tileSizes, staticShape);

      scf::SCFTilingOptions tilingOptions;
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp)
          .setTileSizes(tileSizeOfrs);

      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, tileableOp, tilingOptions);
      if (failed(tilingResult)) {
        op->emitOpError("failed to tile op");
        signalPassFailure();
        return;
      }

      rewriter.eraseOp(op);
      fuseProducersGreedily(rewriter, tilingResult->loops,
                            tilingResult->generatedSlices);
    }
  }
};

} // namespace
