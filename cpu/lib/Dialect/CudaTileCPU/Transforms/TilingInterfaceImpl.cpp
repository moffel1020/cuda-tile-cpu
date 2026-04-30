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

struct StorePtrTileOpTilingInterface
    : public TilingInterface::ExternalModel<StorePtrTileOpTilingInterface,
                                            cpu::StorePtrTileOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto storeOp = cast<cpu::StorePtrTileOp>(op);
    auto valueTy = storeOp.getValue().getType();
    SmallVector<utils::IteratorType> iteratorTypes(
        valueTy.getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto storeOp = cast<cpu::StorePtrTileOp>(op);
    auto valueTy = storeOp.getValue().getType();
    auto shape = valueTy.getShape();

    SmallVector<Range> ranges = llvm::map_to_vector(shape, [&](int64_t s) {
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
    auto storeOp = cast<cpu::StorePtrTileOp>(op);
    auto valueTy = storeOp.getValue().getType();
    Location loc = storeOp.getLoc();

    if (offsets.size() != static_cast<size_t>(valueTy.getRank()) ||
        sizes.size() != static_cast<size_t>(valueTy.getRank()))
      return failure();

    SmallVector<OpFoldResult> strides(valueTy.getRank(), b.getIndexAttr(1));
    auto destSlice = tensor::ExtractSliceOp::create(
        b, loc, storeOp.getDestination(), offsets, sizes, strides);
    auto valueSlice = tensor::ExtractSliceOp::create(
        b, loc, storeOp.getValue(), offsets, sizes, strides);

    auto tiledStore =
        cpu::StorePtrTileOp::create(b, loc, destSlice, valueSlice);
    return TilingResult{{tiledStore.getOperation()},
                        {},
                        {destSlice.getOperation(), valueSlice.getOperation()}};
  }
};

} // namespace

namespace mlir {
namespace cuda_tile {
namespace cpu {

void registerTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CudaTileCPUDialect *dialect) {
    LoadPtrTileOp::attachInterface<LoadPtrTileOpTilingInterface>(*ctx);
    StorePtrTileOp::attachInterface<StorePtrTileOpTilingInterface>(*ctx);
  });
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir
