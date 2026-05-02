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
getTileSizesForStore(OpBuilder &builder, TilingInterface storeOp,
                     ArrayRef<int64_t> requestedTileSizes) {
  SmallVector<Range> domain = storeOp.getIterationDomain(builder);
  SmallVector<OpFoldResult> tileSizeOfrs;
  tileSizeOfrs.reserve(domain.size());

  for (auto [index, range] : llvm::enumerate(domain)) {
    int64_t tileSize = 32;
    if (index < requestedTileSizes.size()) {
      tileSize = requestedTileSizes[index];
    }
    tileSizeOfrs.push_back(builder.getIndexAttr(tileSize));
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

struct TileAndFuseIntoStorePass
    : public cuda_tile::cpu::impl::TileAndFuseIntoStorePassBase<
          TileAndFuseIntoStorePass> {
  using TileAndFuseIntoStorePassBase::TileAndFuseIntoStorePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();
    IRRewriter rewriter(context);

    SmallVector<cpu::StorePtrTileOp> storeOps;
    mod.walk([&](cpu::StorePtrTileOp storeOp) { storeOps.push_back(storeOp); });

    for (cpu::StorePtrTileOp storeOp : storeOps) {
      auto tileableStore = dyn_cast<TilingInterface>(storeOp.getOperation());
      if (!tileableStore) {
        storeOp.emitOpError(
            "expected store_ptr_tile to implement TilingInterface");
        signalPassFailure();
        return;
      }

      rewriter.setInsertionPoint(storeOp);
      SmallVector<OpFoldResult> tileSizeOfrs =
          getTileSizesForStore(rewriter, tileableStore, tileSizes);

      scf::SCFTilingOptions tilingOptions;
      tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp)
          .setTileSizes(tileSizeOfrs);

      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, tileableStore, tilingOptions);
      if (failed(tilingResult)) {
        storeOp.emitOpError("failed to tile store_ptr_tile");
        signalPassFailure();
        return;
      }

      rewriter.eraseOp(storeOp);
      fuseProducersGreedily(rewriter, tilingResult->loops,
                            tilingResult->generatedSlices);
    }
  }
};

} // namespace
