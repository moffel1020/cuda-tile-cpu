#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <deque>

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

static void appendGeneratedSlices(std::deque<tensor::ExtractSliceOp> &worklist,
                                  ArrayRef<Operation *> generatedSlices) {
  for (Operation *slice : generatedSlices) {
    if (auto extractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(slice)) {
      worklist.push_back(extractSlice);
    }
  }
}

static bool hasNonDestinationUse(tensor::ExtractSliceOp slice) {
  return llvm::any_of(slice->getUses(), [](OpOperand &use) {
    auto destinationStyleOp =
        dyn_cast<DestinationStyleOpInterface>(use.getOwner());
    return !destinationStyleOp || !destinationStyleOp.isDpsInit(&use);
  });
}

struct FusedSlice {
  Value source;
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  Value tiledValue;
};

static Value findFusedSlice(const DominanceInfo &dominanceInfo,
                            ArrayRef<FusedSlice> fusedSlices,
                            tensor::ExtractSliceOp candidate) {
  for (const FusedSlice &fused : fusedSlices) {
    if (fused.source == candidate.getSource() &&
        dominanceInfo.dominates(fused.tiledValue, candidate) &&
        isEqualConstantIntOrValueArray(fused.offsets,
                                       candidate.getMixedOffsets()) &&
        isEqualConstantIntOrValueArray(fused.sizes,
                                       candidate.getMixedSizes()) &&
        isEqualConstantIntOrValueArray(fused.strides,
                                       candidate.getMixedStrides())) {
      return fused.tiledValue;
    }
  }
  return {};
}

static tensor::ExtractSliceOp
popFirstInProgramOrder(std::deque<tensor::ExtractSliceOp> &worklist,
                       const DominanceInfo &dominanceInfo) {
  auto first = worklist.begin();
  for (auto it = std::next(worklist.begin()); it != worklist.end(); ++it) {
    if (dominanceInfo.properlyDominates(it->getOperation(),
                                        first->getOperation())) {
      first = it;
    }
  }

  tensor::ExtractSliceOp result = *first;
  worklist.erase(first);
  return result;
}

static void fuseProducersGreedily(IRRewriter &rewriter,
                                  SmallVector<LoopLikeOpInterface> &loops,
                                  ArrayRef<Operation *> initialSlices) {
  std::deque<tensor::ExtractSliceOp> worklist;
  SmallVector<FusedSlice> fusedSlices;
  DominanceInfo dominanceInfo;
  appendGeneratedSlices(worklist, initialSlices);

  while (!worklist.empty()) {
    tensor::ExtractSliceOp candidateSlice =
        popFirstInProgramOrder(worklist, dominanceInfo);
    if (!candidateSlice || candidateSlice.use_empty()) {
      continue;
    }

    // Tiling a destination-style op creates slices for both its inputs and
    // inits. Elementwise-to-linalg commonly aliases the first input and the
    // init, so recursively fusing both paths duplicates the producer graph at
    // every step. Keep destination-only slices as the tiled op's init and fuse
    // through data inputs only.
    if (!hasNonDestinationUse(candidateSlice)) {
      continue;
    }

    if (Value tiledValue =
            findFusedSlice(dominanceInfo, fusedSlices, candidateSlice)) {
      rewriter.replaceAllUsesWith(candidateSlice, tiledValue);
      rewriter.eraseOp(candidateSlice);
      continue;
    }

    FusedSlice fusedSlice{candidateSlice.getSource(),
                          candidateSlice.getMixedOffsets(),
                          candidateSlice.getMixedSizes(),
                          candidateSlice.getMixedStrides(),
                          {}};

    std::optional<scf::SCFFuseProducerOfSliceResult> fusedResult =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSlice, loops);
    if (!fusedResult) {
      continue;
    }

    fusedSlice.tiledValue = fusedResult->tiledAndFusedProducer;
    fusedSlices.push_back(std::move(fusedSlice));

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

  auto scatterTile = dyn_cast<cpu::ScatterTileOp>(op);
  if (scatterTile && scatterTile.getValue().getType().hasStaticShape()) {
    return scatterTile.getValue().getType().getShape();
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
