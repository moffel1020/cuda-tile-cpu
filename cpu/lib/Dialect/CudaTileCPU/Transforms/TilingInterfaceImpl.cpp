#include "cuda_tile_cpu/Dialect/CudaTileCPU/Transforms/TilingInterfaceImpl.h"

#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace {

using namespace mlir;
using namespace cuda_tile;

static SmallVector<Range>
getStaticShapeIterationDomain(OpBuilder &b, ArrayRef<int64_t> shape) {
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

static SmallVector<int64_t>
getShapeFromTileSizes(ArrayRef<OpFoldResult> sizes) {
  return llvm::map_to_vector(sizes, [](OpFoldResult size) {
    std::optional<int64_t> staticSize = getConstantIntValue(size);
    return staticSize.value_or(ShapedType::kDynamic);
  });
}

static SmallVector<Value>
addTileOffsetsToBaseOffsets(OpBuilder &b, Location loc, ValueRange baseOffsets,
                            ArrayRef<OpFoldResult> tileOffsets) {
  SmallVector<Value> combinedOffsets;
  combinedOffsets.reserve(baseOffsets.size());

  for (auto [baseOffset, tileOffset] :
       llvm::zip_equal(baseOffsets, tileOffsets)) {
    if (std::optional<int64_t> staticOffset = getConstantIntValue(tileOffset);
        staticOffset && *staticOffset == 0) {
      combinedOffsets.push_back(baseOffset);
      continue;
    }

    Value tileOffsetValue = getValueOrCreateConstantIndexOp(b, loc, tileOffset);
    combinedOffsets.push_back(
        arith::AddIOp::create(b, loc, baseOffset, tileOffsetValue));
  }

  return combinedOffsets;
}

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

    return getStaticShapeIterationDomain(b, shape);
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto loadOp = cast<cpu::LoadPtrTileOp>(op);
    auto resTy = loadOp.getType();
    Location loc = loadOp.getLoc();
    SmallVector<OpFoldResult> strides(resTy.getRank(), b.getIndexAttr(1));

    auto src = loadOp.getSource();
    auto sourceSlice =
        tensor::ExtractSliceOp::create(b, loc, src, offsets, sizes, strides);

    auto mask = loadOp.getMask();
    auto maskSlice = mask ? tensor::ExtractSliceOp::create(
                                b, loc, mask, offsets, sizes, strides)
                          : nullptr;

    auto padding = loadOp.getPaddingValue();
    auto paddingSlice = padding ? tensor::ExtractSliceOp::create(
                                      b, loc, padding, offsets, sizes, strides)
                                : nullptr;

    auto slicedShape = sourceSlice.getType().getShape();
    auto slicedResTy =
        RankedTensorType::get(slicedShape, resTy.getElementType());

    auto tiledLoad = cpu::LoadPtrTileOp::create(
        b, loc, slicedResTy, sourceSlice, maskSlice, paddingSlice);

    auto tilingResult = TilingResult{{tiledLoad.getOperation()},
                                     {tiledLoad.getResult()},
                                     {sourceSlice.getOperation()}};
    if (mask != nullptr) {
      tilingResult.generatedSlices.push_back(maskSlice.getOperation());
    }

    if (padding != nullptr) {
      tilingResult.generatedSlices.push_back(paddingSlice.getOperation());
    }

    return tilingResult;
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
    if (resultNumber != 0) {
      return failure();
    }
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

    return getStaticShapeIterationDomain(b, shape);
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto storeOp = cast<cpu::StorePtrTileOp>(op);
    auto valueTy = storeOp.getValue().getType();
    Location loc = storeOp.getLoc();

    if (offsets.size() != valueTy.getRank() ||
        sizes.size() != valueTy.getRank()) {
      return failure();
    }

    SmallVector<OpFoldResult> strides(valueTy.getRank(), b.getIndexAttr(1));
    auto destSlice = tensor::ExtractSliceOp::create(
        b, loc, storeOp.getDestination(), offsets, sizes, strides);
    auto valueSlice = tensor::ExtractSliceOp::create(b, loc, storeOp.getValue(),
                                                     offsets, sizes, strides);

    auto mask = storeOp.getMask();
    auto maskSlice = mask ? tensor::ExtractSliceOp::create(
                                b, loc, mask, offsets, sizes, strides)
                          : nullptr;

    auto tiledStore =
        cpu::StorePtrTileOp::create(b, loc, destSlice, valueSlice, maskSlice);

    if (maskSlice != nullptr) {
      return TilingResult{{tiledStore.getOperation()},
                          {},
                          {destSlice.getOperation(), valueSlice.getOperation(),
                           maskSlice.getOperation()}};
    } else {
      return TilingResult{
          {tiledStore.getOperation()},
          {},
          {destSlice.getOperation(), valueSlice.getOperation()}};
    }
  }
};

struct LoadMemRefTileOpTilingInterface
    : public TilingInterface::ExternalModel<LoadMemRefTileOpTilingInterface,
                                            cpu::LoadMemRefTileOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto loadOp = cast<cpu::LoadMemRefTileOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        loadOp.getType().getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto loadOp = cast<cpu::LoadMemRefTileOp>(op);
    return getStaticShapeIterationDomain(b, loadOp.getType().getShape());
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto loadOp = cast<cpu::LoadMemRefTileOp>(op);
    auto resultTy = loadOp.getType();
    Location loc = loadOp.getLoc();

    if (offsets.size() != static_cast<size_t>(resultTy.getRank()) ||
        sizes.size() != static_cast<size_t>(resultTy.getRank()) ||
        loadOp.getOffsets().size() != offsets.size()) {
      return failure();
    }

    SmallVector<Value> tiledOffsets =
        addTileOffsetsToBaseOffsets(b, loc, loadOp.getOffsets(), offsets);
    auto tiledResultTy = RankedTensorType::get(getShapeFromTileSizes(sizes),
                                               resultTy.getElementType());
    auto tiledLoad = cpu::LoadMemRefTileOp::create(
        b, loc, tiledResultTy, loadOp.getSource(), tiledOffsets,
        loadOp.getPaddingValueAttr());

    return TilingResult{
        {tiledLoad.getOperation()}, {tiledLoad.getResult()}, {}};
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
    if (resultNumber != 0) {
      return failure();
    }
    return getTiledImplementation(op, b, offsets, sizes);
  }
};

struct StoreMemRefTileOpTilingInterface
    : public TilingInterface::ExternalModel<StoreMemRefTileOpTilingInterface,
                                            cpu::StoreMemRefTileOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto storeOp = cast<cpu::StoreMemRefTileOp>(op);
    auto valueTy = storeOp.getValue().getType();
    SmallVector<utils::IteratorType> iteratorTypes(
        valueTy.getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto storeOp = cast<cpu::StoreMemRefTileOp>(op);
    return getStaticShapeIterationDomain(
        b, storeOp.getValue().getType().getShape());
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto storeOp = cast<cpu::StoreMemRefTileOp>(op);
    auto valueTy = storeOp.getValue().getType();
    Location loc = storeOp.getLoc();

    if (offsets.size() != static_cast<size_t>(valueTy.getRank()) ||
        sizes.size() != static_cast<size_t>(valueTy.getRank()) ||
        storeOp.getOffsets().size() != offsets.size()) {
      return failure();
    }

    SmallVector<OpFoldResult> strides(valueTy.getRank(), b.getIndexAttr(1));
    auto valueSlice = tensor::ExtractSliceOp::create(b, loc, storeOp.getValue(),
                                                     offsets, sizes, strides);
    SmallVector<Value> tiledOffsets =
        addTileOffsetsToBaseOffsets(b, loc, storeOp.getOffsets(), offsets);
    auto tiledStore = cpu::StoreMemRefTileOp::create(
        b, loc, valueSlice, storeOp.getDestination(), tiledOffsets);

    return TilingResult{
        {tiledStore.getOperation()}, {}, {valueSlice.getOperation()}};
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
    LoadMemRefTileOp::attachInterface<LoadMemRefTileOpTilingInterface>(*ctx);
    StoreMemRefTileOp::attachInterface<StoreMemRefTileOpTilingInterface>(*ctx);
  });
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir
