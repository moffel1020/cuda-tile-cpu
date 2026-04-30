#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/TilingInterfaceImpl.h"

#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace {

using namespace mlir;
using namespace cuda_tile;

struct LoadPtrTileOpTilingInterface
    : public TilingInterface::ExternalModel<LoadPtrTileOpTilingInterface,
                                            cpu::LoadPtrTileOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto loadOp = cast<cpu::LoadPtrTileOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        loadOp.getType().getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto loadOp = cast<cpu::LoadPtrTileOp>(op);
    auto ty = loadOp.getType();
    auto shape = ty.getShape();

    SmallVector<Range> ranges = llvm::map_to_vector(shape, [&](auto s) {
      OpFoldResult zero = b.getIndexAttr(0);
      OpFoldResult one = b.getIndexAttr(1);
      return Range{
          /*offset=*/zero,
          /*size=*/b.getIndexAttr(s),
          /*stride=*/one,
      };
    });
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto loadOp = cast<cpu::LoadPtrTileOp>(op);
    auto resTy = loadOp.getType();
    auto src = loadOp.getSource();
    Location loc = loadOp.getLoc();

    SmallVector<OpFoldResult> strides(resTy.getRank(), b.getIndexAttr(1));
    auto slice =
        tensor::ExtractSliceOp::create(b, loc, src, offsets, sizes, strides);

    auto slicedShape = slice.getType().getShape();
    auto slicedResTy =
        RankedTensorType::get(slicedShape, resTy.getElementType());

    auto tiledLoad = cpu::LoadPtrTileOp::create(b, loc, slicedResTy, slice);
    return TilingResult{{tiledLoad.getOperation()},
                        {tiledLoad.getResult()},
                        {slice.getOperation()}};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets = SmallVector<OpFoldResult>(offsets);
    resultSizes = SmallVector<OpFoldResult>(sizes);
    return success();
  };

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    if (resultNumber != 0)
      return failure();
    return getTiledImplementation(op, b, offsets, sizes);
  }
};

} // namespace

namespace mlir {
namespace cuda_tile {
namespace cpu {

void registerTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CudaTileCPUDialect *dialect) {
    LoadPtrTileOp::attachInterface<LoadPtrTileOpTilingInterface>(*ctx);
  });
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir
