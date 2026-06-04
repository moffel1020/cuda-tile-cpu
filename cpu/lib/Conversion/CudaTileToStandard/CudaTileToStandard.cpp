#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {
namespace cuda_tile {
namespace cpu {

#define GEN_PASS_DEF_CONVERTCUDATILETOSTANDARD
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

namespace {

using namespace mlir;
using namespace mlir::cuda_tile;
using namespace llvm;

class CudaTileTypeConverter : public mlir::TypeConverter {
public:
  CudaTileTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion(
        [&](cuda_tile::TileType type) -> Type { return convertTile(type); });

    addConversion([&](cuda_tile::TensorViewType type) -> Type {
      return convertTensorViewToMemRef(type);
    });

    addConversion([&](cuda_tile::PartitionViewType type) -> Type {
      // note: this loses information (padding, tile size), make sure to save
      // that elsewhere if it is still needed
      return convertTensorViewToMemRef(type.getTensorView());
    });
  }

private:
  Type convertTile(cuda_tile::TileType type) const {
    auto shape = type.getShape();
    Type elemTy = type.getElementType();
    if (auto ptrType = dyn_cast<cuda_tile::PointerType>(elemTy)) {
      elemTy = convertCudaTilePtr(ptrType);
    }
    if (shape.empty()) {
      // convert to scalar
      return elemTy;
    }
    return RankedTensorType::get(shape, elemTy);
  }

  Type convertTensorViewToMemRef(cuda_tile::TensorViewType type) const {
    auto shape = type.getShape();
    Type elemTy = type.getElementType();
    auto layout =
        StridedLayoutAttr::get(type.getContext(), /*offset=*/0,
                               llvm::ArrayRef<int64_t>(type.getStrides()));
    return MemRefType::get(shape, elemTy, layout);
  }

  Type convertCudaTilePtr(cuda_tile::PointerType ptrType) const {
    return IntegerType::get(ptrType.getContext(), 64);
  }
};

static std::optional<arith::RoundingMode>
convertRoundingMode(cuda_tile::RoundingMode mode) {
  using CTR = cuda_tile::RoundingMode;
  using R = arith::RoundingMode;

  switch (mode) {
  case CTR::NEAREST_EVEN:
    return R::to_nearest_even;
  case CTR::ZERO:
    return R::toward_zero;
  case CTR::NEGATIVE_INF:
    return R::downward;
  case CTR::POSITIVE_INF:
    return R::upward;
  case CTR::APPROX:
  case CTR::FULL:
  case CTR::NEAREST_INT_TO_ZERO:
  default:
    // just using default rounding mode here. TODO: give error/warning?
    return std::nullopt;
  }
}

static arith::IntegerOverflowFlags
convertOverflowFlags(cuda_tile::IntegerOverflow of) {
  using OF = arith::IntegerOverflowFlags;
  switch (of) {
  case IntegerOverflow::NONE:
    return OF::none;
  case IntegerOverflow::NSW:
    return OF::nsw;
  case IntegerOverflow::NUW:
    return OF::nuw;
  case IntegerOverflow::NW:
    return OF::nsw | OF::nuw;
  default:
    llvm_unreachable("invalid overflow flag");
  }
}

static bool isScalarType(Type type) { return !isa<ShapedType>(type); }

static bool isScalarValue(Value value) {
  return value && isScalarType(value.getType());
}

static Value createTensorSplat(OpBuilder &builder, Location loc, Value scalar,
                               RankedTensorType resultType) {
  auto empty = tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType());
  return linalg::FillOp::create(builder, loc, ValueRange{scalar},
                                ValueRange{empty})
      .getResult(0);
}

static Value extractScalarTensor(Location loc, Value value,
                                 OpBuilder &rewriter) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (tensorType.getRank() == 0) {
      return tensor::ExtractOp::create(rewriter, loc, value, ValueRange{});
    }
  }
  return value;
}

static Value extractScalarTensorAsIndex(Location loc, Value value,
                                        OpBuilder &rewriter) {
  value = extractScalarTensor(loc, value, rewriter);
  if (isa<IndexType>(value.getType())) {
    return value;
  }
  return arith::IndexCastUIOp::create(rewriter, loc, rewriter.getIndexType(),
                                      value);
}

struct EntryPattern : public OpConversionPattern<cuda_tile::EntryOp> {
  using OpConversionPattern<cuda_tile::EntryOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::EntryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // arg types are converted by marking signature dynamically legal
    auto func = func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                     op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.replaceOp(op, func);

    return success();
  }
};

struct ReturnPattern : public OpConversionPattern<cuda_tile::ReturnOp> {
  using OpConversionPattern<cuda_tile::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.getNumOperands() > 0) {
      return rewriter.notifyMatchFailure(op,
                                         "return with operands not supported");
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

struct ConstantPattern : public OpConversionPattern<cuda_tile::ConstantOp> {
  using OpConversionPattern<cuda_tile::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto oldType = op.getType();
    auto oldAttr = dyn_cast<DenseElementsAttr>(op.getValueAttr());
    if (!oldAttr) {
      return failure();
    }

    if (oldType.getShape().empty()) {
      auto scalarAttr = cast<TypedAttr>(oldAttr.getValues<Attribute>()[0]);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, oldType.getElementType(), scalarAttr);
      return success();
    }

    auto tensorType =
        RankedTensorType::get(oldType.getShape(), oldType.getElementType());
    auto newAttr = oldAttr.reshape(tensorType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, tensorType, newAttr);

    return success();
  }
};

struct IotaPattern : public OpConversionPattern<cuda_tile::IotaOp> {
  using OpConversionPattern<cuda_tile::IotaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opTy = op.getResult().getType();
    if (opTy.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "1d shape expected for iota op");
    }

    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    AffineMap idMap = AffineMap::getMultiDimIdentityMap(1, getContext());

    auto newOp = linalg::GenericOp::create(
        rewriter, op.getLoc(),
        /*resultTensorTypes=*/empty.getType(),
        /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{empty},
        /*indexingMaps=*/ArrayRef<AffineMap>{idMap},
        /*iteratorTypes=*/utils::IteratorType::parallel,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value idx = linalg::IndexOp::create(b, loc, 0);
          Value idxCasted =
              arith::IndexCastOp::create(b, loc, opTy.getElementType(), idx);
          linalg::YieldOp::create(b, loc, idxCasted);
        });

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct OffsetPattern : public OpConversionPattern<cuda_tile::OffsetOp> {
  using OpConversionPattern<cuda_tile::OffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrTy = op.getPtr().getType().getElementType();
    auto pointeeTy = cast<PointerType>(ptrTy).getPointeeType();
    auto bitWidth = pointeeTy.getIntOrFloatBitWidth();
    // the spec says multiply by bitwidth here but that seems wrong
    // byte count is more logical
    auto offWidth = op.getOffset().getType().getElementTypeBitWidth();
    auto widthConst =
        arith::ConstantIntOp::create(rewriter, op.getLoc(), bitWidth / 8, 64);

    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         rewriter.getI64Type());

    auto flags =
        arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw;

    auto createOffsetValue = [&](OpBuilder &b, Location loc, Value ptrElem,
                                 Value offElem) -> Value {
      if (offWidth != 64) {
        offElem = arith::ExtUIOp::create(b, loc, rewriter.getI64Type(), offElem,
                                         true);
      }
      auto mulOp = arith::MulIOp::create(b, loc, offElem, widthConst, flags);
      return arith::AddIOp::create(b, loc, ptrElem, mulOp, flags);
    };

    auto resultType = getTypeConverter()->convertType(op.getType());
    if (isScalarType(resultType)) {
      rewriter.replaceOp(op, createOffsetValue(rewriter, op.getLoc(),
                                               adaptor.getPtr(),
                                               adaptor.getOffset()));
      return success();
    }

    auto newOp = linalg::MapOp::create(
        rewriter, op.getLoc(), {adaptor.getPtr(), adaptor.getOffset()}, empty,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto ptrElem = args[0];
          auto offElem = args[1];
          linalg::YieldOp::create(b, loc,
                                  createOffsetValue(b, loc, ptrElem, offElem));
        });

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct BroadcastPattern : public OpConversionPattern<cuda_tile::BroadcastOp> {
  using OpConversionPattern<cuda_tile::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto opTy = op.getType();
    auto rank = opTy.getRank();
    auto newTy = cast<RankedTensorType>(getTypeConverter()->convertType(opTy));

    if (isScalarValue(adaptor.getSource())) {
      rewriter.replaceOp(op, createTensorSplat(rewriter, op.getLoc(),
                                               adaptor.getSource(), newTy));
      return success();
    }

    AffineMap inputMap =
        getBroadcastInputMap(op.getSource().getType().getShape(),
                             opTy.getRank(), rewriter.getContext());
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(
        opTy.getRank(), rewriter.getContext());

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto empty = tensor::EmptyOp::create(
        rewriter, op.getLoc(), newTy.getShape(), newTy.getElementType());

    auto newOp = linalg::GenericOp::create(
        rewriter, op.getLoc(), empty.getType(), adaptor.getOperands(), {empty},
        {inputMap, outputMap}, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          linalg::YieldOp::create(b, loc, args[0]);
        });

    rewriter.replaceOp(op, newOp);
    return success();
  }

private:
  static AffineMap getBroadcastInputMap(ArrayRef<int64_t> input,
                                        int64_t outputRank, MLIRContext *ctx) {
    SmallVector<AffineExpr> exprs;
    for (auto [i, dimSize] : llvm::enumerate(input)) {
      if (dimSize == 1) {
        exprs.push_back(getAffineConstantExpr(0, ctx));
      } else {
        exprs.push_back(getAffineDimExpr(i, ctx));
      }
    }

    return AffineMap::get(outputRank, 0, exprs, ctx);
  }
};

struct ReshapePattern : public OpConversionPattern<cuda_tile::ReshapeOp> {
  using OpConversionPattern<cuda_tile::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    Value source = adaptor.getSource();

    if (isScalarType(resultType)) {
      rewriter.replaceOp(op,
                         extractScalarTensor(op.getLoc(), source, rewriter));
      return success();
    }

    auto tensorType = cast<RankedTensorType>(resultType);
    if (isScalarValue(source)) {
      auto elemOp = tensor::FromElementsOp::create(rewriter, op.getLoc(),
                                                   resultType, source);
      rewriter.replaceOp(op, elemOp);
      return success();
    }

    auto shapeType =
        RankedTensorType::get({tensorType.getRank()}, rewriter.getIndexType());
    SmallVector<APInt> values =
        llvm::map_to_vector(tensorType.getShape(), [](int64_t val) {
          return APInt{64, (uint64_t)val};
        });

    auto attr = DenseElementsAttr::get(shapeType, values);
    auto shape = arith::ConstantOp::create(rewriter, op.getLoc(), attr);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, tensorType, source,
                                                   shape);
    return success();
  }
};

struct CatPattern : public OpConversionPattern<cuda_tile::CatOp> {
  using OpConversionPattern<cuda_tile::CatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, op.getDim(),
                                                  adaptor.getOperands());
    return success();
  }
};

struct ExtractPattern : public OpConversionPattern<cuda_tile::ExtractOp> {
  using OpConversionPattern<cuda_tile::ExtractOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (isScalarType(convertedResultType)) {
      SmallVector<Value> indices =
          llvm::map_to_vector(adaptor.getIndices(), [&](auto idx) {
            return extractScalarTensorAsIndex(op.getLoc(), idx, rewriter);
          });
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, adaptor.getSource(),
                                                     indices);
      return success();
    }

    auto resType = cast<TensorType>(convertedResultType);
    auto resShape = resType.getShape();
    SmallVector<int64_t> staticStrides(resType.getRank(), 1);
    SmallVector<int64_t> staticOffsets(resType.getRank(), ShapedType::kDynamic);

    SmallVector<Value> offsets;
    for (auto [size, idx] : llvm::zip(resShape, adaptor.getIndices())) {
      auto shapeSizeOp = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                        /*value=*/size);
      auto castedUnpackedIdx =
          extractScalarTensorAsIndex(op.getLoc(), idx, rewriter);

      offsets.emplace_back(arith::MulIOp::create(
          rewriter, op.getLoc(), shapeSizeOp, castedUnpackedIdx));
    }

    auto newOp = tensor::ExtractSliceOp::create(
        rewriter, op.getLoc(), resType, adaptor.getSource(),
        /*offsets=*/offsets, /*sizes=*/ValueRange{},
        /*strides=*/ValueRange{},
        /*static_offsets=*/staticOffsets,
        /*static_sizes=*/resShape,
        /*static_strides=*/staticStrides);

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct PermutePattern : public OpConversionPattern<cuda_tile::PermuteOp> {
  using OpConversionPattern<cuda_tile::PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::PermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    SmallVector<int64_t> permuteMap = llvm::map_to_vector(
        op.getPermutation(), [](auto v) { return static_cast<int64_t>(v); });

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(op, adaptor.getSource(),
                                                     empty, permuteMap);
    return success();
  }
};

struct SelectPattern : public OpConversionPattern<cuda_tile::SelectOp> {
  using OpConversionPattern<cuda_tile::SelectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, adaptor.getCond(), adaptor.getValIfTrue(), adaptor.getValIfFalse());
    return success();
  }
};

struct ReducePattern : public OpConversionPattern<cuda_tile::ReduceOp> {
  using OpConversionPattern<cuda_tile::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                convertedResultTypes))) {
      return failure();
    }

    // create init types, inputs and outputs of linalg op should be tensors
    SmallVector<Type> linalgResultTypes;
    SmallVector<RankedTensorType> initTypes;
    for (Type t : convertedResultTypes) {
      RankedTensorType initType;
      if (isScalarType(t)) {
        initType = RankedTensorType::get({}, t);
      } else {
        initType = cast<RankedTensorType>(t);
      }
      initTypes.push_back(initType);
      linalgResultTypes.push_back(initType);
    }

    // create identity constant values
    SmallVector<Value> inits;
    auto identities = op.getIdentities();
    inits.reserve(identities.size());
    for (auto [id, type] : llvm::zip(identities, initTypes)) {
      Value identity =
          arith::ConstantOp::create(rewriter, op.getLoc(), cast<TypedAttr>(id));
      inits.push_back(createTensorSplat(rewriter, op.getLoc(), identity, type));
    }

    // create reduce op and convert arg types
    auto reduceOp = linalg::ReduceOp::create(
        rewriter, op.getLoc(), linalgResultTypes, adaptor.getOperands(), inits,
        ArrayRef<int64_t>{static_cast<int64_t>(op.getDim())});

    Region &combiner = reduceOp.getCombiner();
    Block *combinerBlock = rewriter.createBlock(&combiner);

    SmallVector<Type> blockArgTypes;
    for (Value input : adaptor.getOperands()) {
      auto inputType = cast<RankedTensorType>(input.getType());
      blockArgTypes.push_back(inputType.getElementType());
    }
    for (RankedTensorType initType : initTypes) {
      blockArgTypes.push_back(initType.getElementType());
    }
    for (Type type : blockArgTypes) {
      combinerBlock->addArgument(type, op.getLoc());
    }

    Block &oldBlock = op.getBody().front();
    IRMapping mapping;
    auto numOperands = op.getNumOperands();
    for (auto [index, oldArg] : llvm::enumerate(oldBlock.getArguments())) {
      // in cuda tile: input and accums are interleaved
      // in linalg: first all inputs, then all accums
      auto newIndex = index % 2 == 0 ? index / 2 : numOperands + (index / 2);
      mapping.map(oldArg, combinerBlock->getArgument(newIndex));
    }

    rewriter.setInsertionPointToEnd(combinerBlock);
    for (Operation &nested : oldBlock.without_terminator()) {
      rewriter.clone(nested, mapping);
    }

    rewriter.clone(*oldBlock.getTerminator(), mapping);

    rewriter.setInsertionPointAfter(reduceOp);

    SmallVector<Value> replacements;
    for (auto [result, convertedType] :
         llvm::zip(reduceOp.getResults(), convertedResultTypes)) {
      if (isScalarType(convertedType)) {
        replacements.push_back(tensor::ExtractOp::create(rewriter, op.getLoc(),
                                                         result, ValueRange{}));
      } else {
        replacements.push_back(result);
      }
    }

    rewriter.replaceOp(op, replacements);
    return success();
  }
};

template <typename T, typename U>
struct ConvertElementwise : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = U::create(rewriter, op.getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

template <typename T, typename SignedOp, typename UnsignedMapOp>
struct MaxIMinIPattern : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto signedness = op.getSignedness();
    switch (signedness) {
    case Signedness::Unsigned:
      rewriter.replaceOpWithNewOp<UnsignedMapOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
      return success();
    case Signedness::Signed:
      rewriter.replaceOpWithNewOp<SignedOp>(op, adaptor.getLhs(),
                                            adaptor.getRhs());
      return success();
    default:
      llvm_unreachable("only unsigned and signed are valid");
    }
    return success();
  }
};

template <typename T, typename PropOp, typename NoPropOp>
struct MaxFMinFPattern : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;
  // ignore ftz
  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getPropagateNan()) {
      rewriter.replaceOpWithNewOp<PropOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<NoPropOp>(op, adaptor.getLhs(),
                                            adaptor.getRhs());
    }

    return success();
  }
};

template <typename T, typename SignedOp, typename UnsignedOp>
struct SignedUnsignedPattern : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto signedness = op.getSignedness();
    switch (signedness) {
    case Signedness::Unsigned:
      rewriter.replaceOpWithNewOp<UnsignedOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs());
      return success();
    case Signedness::Signed:
      rewriter.replaceOpWithNewOp<SignedOp>(op, adaptor.getLhs(),
                                            adaptor.getRhs());
      return success();
    default:
      llvm_unreachable("only unsigned and signed are valid");
    }
    return success();
  }
};

struct CmpIPattern : public OpConversionPattern<cuda_tile::CmpIOp> {
  using OpConversionPattern<cuda_tile::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto sign = op.getSignedness();
    auto pred = op.getComparisonPredicate();
    auto cmpPred = [&]() -> FailureOr<arith::CmpIPredicate> {
      using A = arith::CmpIPredicate;
      using S = cuda_tile::Signedness;

      switch (pred) {
      case ComparisonPredicate::EQUAL:
        return A::eq;
      case ComparisonPredicate::NOT_EQUAL:
        return A::ne;
      case ComparisonPredicate::LESS_THAN:
        return sign == S::Signed ? A::slt : A::ult;
      case ComparisonPredicate::LESS_THAN_OR_EQUAL:
        return sign == S::Signed ? A::sle : A::ule;
      case ComparisonPredicate::GREATER_THAN:
        return sign == S::Signed ? A::sgt : A::ugt;
      case ComparisonPredicate::GREATER_THAN_OR_EQUAL:
        return sign == S::Signed ? A::sge : A::uge;
      default:
        return failure();
      }
    }();

    if (failed(cmpPred)) {
      return rewriter.notifyMatchFailure(op, "unknown comparison predicate");
    }

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, *cmpPred, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct DivIPattern : public OpConversionPattern<cuda_tile::DivIOp> {
  using OpConversionPattern<cuda_tile::DivIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sign = op.getSignedness();
    auto newOp = [&]() -> Operation * {
      using S = cuda_tile::Signedness;
      using RM = cuda_tile::RoundingMode;

      switch (op.getRounding()) {
      case RM::NEGATIVE_INF:
        return arith::FloorDivSIOp::create(rewriter, op.getLoc(),
                                           adaptor.getLhs(), adaptor.getRhs());
      case RM::POSITIVE_INF:
        return sign == S::Signed
                   ? arith::CeilDivSIOp::create(rewriter, op.getLoc(),
                                                adaptor.getLhs(),
                                                adaptor.getRhs())
                   : arith::CeilDivUIOp::create(rewriter, op.getLoc(),
                                                adaptor.getLhs(),
                                                adaptor.getRhs());
      case RM::ZERO:
        return sign == S::Signed
                   ? arith::DivSIOp::create(rewriter, op.getLoc(),
                                            adaptor.getLhs(), adaptor.getRhs())
                   : arith::DivUIOp::create(rewriter, op.getLoc(),
                                            adaptor.getLhs(), adaptor.getRhs());
      default:
        return nullptr;
      }
    }();

    if (newOp == nullptr) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "only rounding modes zero, negative_inf and "
                       "positive_inf are allowed "
                       "on DivI");
    }

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct NegIPattern : public OpConversionPattern<cuda_tile::NegIOp> {
  using OpConversionPattern<cuda_tile::NegIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(cuda_tile::NegIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) {

    Type resultType = getTypeConverter()->convertType(op.getType());
    Type elemTy = getElementTypeOrSelf(resultType);
    Value c0 = arith::ConstantIntOp::create(rewriter, op.getLoc(), elemTy, 0);
    if (!isScalarType(resultType)) {
      c0 = createTensorSplat(rewriter, op.getLoc(), c0,
                             cast<RankedTensorType>(resultType));
    }
    rewriter.replaceOpWithNewOp<arith::SubIOp>(
        op, c0, adaptor.getSource(), convertOverflowFlags(op.getOverflow()));
    return success();
  }
};

struct MulHiIPattern : public OpConversionPattern<cuda_tile::MulhiIOp> {
  using OpConversionPattern<cuda_tile::MulhiIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::MulhiIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO optimization: if lower halve is also calculated with a regular mul
    // op, remove it and just use the lower result of this op

    // according to the spec, this op is only defined for unsigned integers
    auto mulOp = arith::MulUIExtendedOp::create(rewriter, op.getLoc(),
                                                adaptor.getX(), adaptor.getY());
    rewriter.replaceOp(op, mulOp.getHigh());
    return success();
  }
};

template <typename T, typename U>
static Value createConversionOp(T op, typename T::Adaptor adaptor,
                                const TypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  Type resultType = typeConverter->convertType(op.getType());
  return U::create(rewriter, op.getLoc(), resultType, adaptor.getOperands()[0]);
}

struct BitcastPattern : public OpConversionPattern<cuda_tile::BitcastOp> {
  using OpConversionPattern<cuda_tile::BitcastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = createConversionOp<cuda_tile::BitcastOp, arith::BitcastOp>(
        op, adaptor, getTypeConverter(), rewriter);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ExtiPattern : public OpConversionPattern<cuda_tile::ExtIOp> {
  using OpConversionPattern<cuda_tile::ExtIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ExtIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = [&] {
      if (op.getSignedness() == Signedness::Signed) {
        return createConversionOp<cuda_tile::ExtIOp, arith::ExtSIOp>(
            op, adaptor, getTypeConverter(), rewriter);
      } else {
        return createConversionOp<cuda_tile::ExtIOp, arith::ExtUIOp>(
            op, adaptor, getTypeConverter(), rewriter);
      }
    }();

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct FToIPattern : public OpConversionPattern<cuda_tile::FToIOp> {
  using OpConversionPattern<cuda_tile::FToIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FToIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getRoundingMode() != cuda_tile::RoundingMode::NEAREST_INT_TO_ZERO) {
      return rewriter.notifyMatchFailure(
          op, "as of cuda tile 13.2, only nearest_int_to_zero is a supported "
              "rounding mode for this op");
    }

    auto newOp = [&] {
      if (op.getSignedness() == Signedness::Signed) {
        return createConversionOp<cuda_tile::FToIOp, arith::FPToSIOp>(
            op, adaptor, getTypeConverter(), rewriter);
      } else {
        return createConversionOp<cuda_tile::FToIOp, arith::FPToUIOp>(
            op, adaptor, getTypeConverter(), rewriter);
      }
    }();

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct FToFPattern : public OpConversionPattern<cuda_tile::FToFOp> {
  using OpConversionPattern<cuda_tile::FToFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FToFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getRoundingMode() != cuda_tile::RoundingMode::NEAREST_INT_TO_ZERO) {
      return rewriter.notifyMatchFailure(
          op, "as of cuda tile 13.2, only nearest_int_to_zero is a supported "
              "rounding mode for this op");
    }

    auto fromType = op.getFrom().getType().getElementType();
    auto fromWidth = fromType.getIntOrFloatBitWidth();

    auto toType = op.getType().getElementType();
    auto toWidth = toType.getIntOrFloatBitWidth();

    if (fromWidth == toWidth) {
      // TODO: update llvm version and use arith.convertf here for bf16 to f16
      return rewriter.notifyMatchFailure(
          op, "unsupported float to float conversion");
    }

    auto roundingMode = convertRoundingMode(op.getRoundingMode());

    Type resultType = getTypeConverter()->convertType(op.getType());
    auto newOp = [&] {
      if (fromWidth > toWidth) {
        return arith::ExtFOp::create(rewriter, op.getLoc(), resultType,
                                     adaptor.getFrom())
            .getOperation();
      } else /* if (fromWidth < toWidth) */ {
        return (!roundingMode
                    ? arith::TruncFOp::create(rewriter, op.getLoc(), resultType,
                                              adaptor.getFrom())
                    : arith::TruncFOp::create(
                          rewriter, op.getLoc(), resultType, adaptor.getFrom(),
                          arith::RoundingModeAttr::get(getContext(),
                                                       roundingMode.value()),
                          {}))
            .getOperation();
      }
    }();

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct IToFPattern : public OpConversionPattern<cuda_tile::IToFOp> {
  using OpConversionPattern<cuda_tile::IToFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IToFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());
    // TODO: rounding mode? these arith ops don't support it

    auto sign = op.getSignedness();
    auto newOp = [&] {
      if (sign == Signedness::Signed) {
        return createConversionOp<cuda_tile::IToFOp, arith::SIToFPOp>(
            op, adaptor, getTypeConverter(), rewriter);
      } else {
        return createConversionOp<cuda_tile::IToFOp, arith::UIToFPOp>(
            op, adaptor, getTypeConverter(), rewriter);
      }
    }();

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TruncIPattern : public OpConversionPattern<cuda_tile::TruncIOp> {
  using OpConversionPattern<cuda_tile::TruncIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(
        op, resultType, adaptor.getFrom(),
        convertOverflowFlags(op.getOverflow()));

    return success();
  }
};

struct CmpFPattern : public OpConversionPattern<cuda_tile::CmpFOp> {
  using OpConversionPattern<cuda_tile::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto pred = op.getComparisonPredicate();
    auto ord = op.getComparisonOrdering();
    auto arithPred = [&]() -> FailureOr<arith::CmpFPredicate> {
      using A = arith::CmpFPredicate;
      using CTA = cuda_tile::ComparisonPredicate;
      using O = cuda_tile::ComparisonOrdering;

      switch (pred) {
      case CTA::LESS_THAN:
        return ord == O::ORDERED ? A::OLT : A::ULT;
      case CTA::LESS_THAN_OR_EQUAL:
        return ord == O::ORDERED ? A::OLE : A::ULE;
      case CTA::GREATER_THAN:
        return ord == O::ORDERED ? A::OGT : A::UGT;
      case CTA::GREATER_THAN_OR_EQUAL:
        return ord == O::ORDERED ? A::OGE : A::UGE;
      case CTA::EQUAL:
        return ord == O::ORDERED ? A::OEQ : A::UEQ;
      case CTA::NOT_EQUAL:
        return ord == O::ORDERED ? A::ONE : A::UNE;
      default:
        return failure();
      }
    }();

    if (failed(arithPred)) {
      return rewriter.notifyMatchFailure(op, "unknwon comparison predicate");
    }

    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, *arithPred, adaptor.getLhs(),
                                               adaptor.getRhs());

    return success();
  }
};

struct MmaIPattern : public OpConversionPattern<cuda_tile::MmaIOp> {
  using OpConversionPattern<cuda_tile::MmaIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::MmaIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // lhs and rhs must be i8 and acc must be i32 according to spec

    if (isScalarValue(adaptor.getAcc()) || isScalarValue(adaptor.getLhs()) ||
        isScalarValue(adaptor.getRhs())) {
      return rewriter.notifyMatchFailure(
          op, "converted matmul ops should be tensors");
    }

    auto extendIn = [&](auto val, cuda_tile::Signedness sign) -> Operation * {
      auto valTy = cast<TensorType>(val.getType());
      auto empty = tensor::EmptyOp::create(
          rewriter, op.getLoc(), valTy.getShape(), rewriter.getI32Type());

      if (sign == Signedness::Signed) {
        return linalg::MapOp::create(
            rewriter, op.getLoc(), val, empty,
            [](OpBuilder &b, Location loc, ValueRange args) {
              Value extOp =
                  arith::ExtSIOp::create(b, loc, b.getI32Type(), args[0]);
              linalg::YieldOp::create(b, loc, extOp);
            });
      } else {
        return linalg::MapOp::create(
            rewriter, op.getLoc(), val, empty,
            [](OpBuilder &b, Location loc, ValueRange args) {
              Value extOp =
                  arith::ExtUIOp::create(b, loc, b.getI32Type(), args[0]);
              linalg::YieldOp::create(b, loc, extOp);
            });
      }
    };

    Value left =
        extendIn(adaptor.getLhs(), op.getSignednessLhs())->getResult(0);
    Value right =
        extendIn(adaptor.getRhs(), op.getSignednessRhs())->getResult(0);

    auto rank = op.getType().getRank();
    if (rank == 2) {
      rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
          op, ValueRange{left, right}, ValueRange{adaptor.getAcc()});
      return success();
    } else if (rank == 3) {
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, ValueRange{left, right}, ValueRange{adaptor.getAcc()});
      return success();
    }

    return failure();
  }
};

struct MmaFPattern : public OpConversionPattern<cuda_tile::MmaFOp> {
  using OpConversionPattern<cuda_tile::MmaFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::MmaFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: we can use matvec op sometimes
    auto rank = op.getType().getRank();
    if (rank == 2) {
      rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
          op, ValueRange{adaptor.getLhs(), adaptor.getRhs()}, adaptor.getAcc());
      return success();
    } else if (rank == 3) {
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, ValueRange{adaptor.getLhs(), adaptor.getRhs()}, adaptor.getAcc());
      return success();
    }

    return failure();
  }
};

// bitwise
using AndIPattern = ConvertElementwise<cuda_tile::AndIOp, arith::AndIOp>;

// integer
// TODO: ignoring overflow information. could preserve some by using linalg
// generic/map
using AbsIPattern = ConvertElementwise<cuda_tile::AbsIOp, math::AbsIOp>;
using AddIPattern = ConvertElementwise<cuda_tile::AddIOp,
                                       arith::AddIOp>; // ignore int overflow
using SubIPattern = ConvertElementwise<cuda_tile::SubIOp,
                                       arith::SubIOp>; // ignore int overflow
using MulIPattern = ConvertElementwise<cuda_tile::MulIOp,
                                       arith::MulIOp>; // ignore int overflow
using ShLIPattern = ConvertElementwise<cuda_tile::ShLIOp,
                                       arith::ShLIOp>; // ignore int overflow
using OrIPattern = ConvertElementwise<cuda_tile::OrIOp, arith::OrIOp>;
using XOrIPattern = ConvertElementwise<cuda_tile::XOrIOp, arith::XOrIOp>;
using MaxIPattern =
    MaxIMinIPattern<cuda_tile::MaxIOp, arith::MaxSIOp, arith::MaxUIOp>;
using MinIPattern =
    MaxIMinIPattern<cuda_tile::MinIOp, arith::MinSIOp, arith::MinUIOp>;
using ShRIPattern =
    SignedUnsignedPattern<cuda_tile::ShRIOp, arith::ShRSIOp, arith::ShRUIOp>;
using RemIPattern =
    SignedUnsignedPattern<cuda_tile::RemIOp, arith::RemSIOp, arith::RemUIOp>;

// floating point
using AbsFPattern = ConvertElementwise<cuda_tile::AbsFOp, math::AbsFOp>;
using CeilPattern = ConvertElementwise<cuda_tile::CeilOp, math::CeilOp>;
using FloorPattern = ConvertElementwise<cuda_tile::FloorOp, math::FloorOp>;
using Atan2Pattern = ConvertElementwise<cuda_tile::Atan2Op, math::Atan2Op>;
using CoshPattern = ConvertElementwise<cuda_tile::CosHOp, math::CoshOp>;
using CosPattern = ConvertElementwise<cuda_tile::CosOp, math::CosOp>;
using ExpPattern = ConvertElementwise<cuda_tile::ExpOp, math::ExpOp>;
using Log2Pattern = ConvertElementwise<cuda_tile::Log2Op, math::Log2Op>;
using LogPattern = ConvertElementwise<cuda_tile::LogOp, math::LogOp>;
using NegFPattern = ConvertElementwise<cuda_tile::NegFOp, arith::NegFOp>;
using SinhPattern = ConvertElementwise<cuda_tile::SinHOp, math::SinhOp>;
using SinPattern = ConvertElementwise<cuda_tile::SinOp, math::SinOp>;
using TanPattern = ConvertElementwise<cuda_tile::TanOp, math::TanOp>;
// TODO: specialize to square for pow 2
using PowPattern = ConvertElementwise<cuda_tile::PowOp, math::PowFOp>;
using AddFPattern =
    ConvertElementwise<cuda_tile::AddFOp,
                       arith::AddFOp>; // ignore rounding and ftz
using DivFPattern =
    ConvertElementwise<cuda_tile::DivFOp,
                       arith::DivFOp>; // ignore rounding and ftz
using Exp2Pattern =
    ConvertElementwise<cuda_tile::Exp2Op, math::Exp2Op>; // ignore ftz
using FmaPattern = ConvertElementwise<cuda_tile::FmaOp,
                                      math::FmaOp>; // ignore rounding and ftz
using MaxFPattern =
    MaxFMinFPattern<cuda_tile::MaxFOp, arith::MaximumFOp, arith::MaxNumFOp>;
using MinFPattern =
    MaxFMinFPattern<cuda_tile::MinFOp, arith::MinimumFOp, arith::MinNumFOp>;
using MulFPattern =
    ConvertElementwise<cuda_tile::MulFOp,
                       arith::MulFOp>; // ignore rounding and ftz
using RsqrtPattern =
    ConvertElementwise<cuda_tile::RsqrtOp, math::RsqrtOp>; // ignore ftz
using SubFPattern =
    ConvertElementwise<cuda_tile::SubFOp,
                       arith::SubFOp>; // ignore rounding and ftz
using SqrtPattern = ConvertElementwise<cuda_tile::SqrtOp,
                                       math::SqrtOp>; // ignore rounding and ftz
using TanHPattern =
    ConvertElementwise<cuda_tile::TanHOp, math::TanhOp>; // ignore rounding
using RemFPattern = ConvertElementwise<cuda_tile::RemFOp, arith::RemFOp>;

struct PrintTkoPattern : public OpConversionPattern<cuda_tile::PrintTkoOp> {
  using OpConversionPattern<cuda_tile::PrintTkoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::PrintTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.getNumOperands() == 0) {
      auto cpuPrint =
          cpu::PrintOp::create(rewriter, op.getLoc(), op.getStr(), {});
      rewriter.eraseOp(op);
      return success();
    }

    SmallVector<StringRef> splitStrings = splitFormatString(op.getStr());
    assert(splitStrings.size() >= op.getNumOperands());

    for (auto [arg, convertedArg, str] :
         llvm::zip(op.getArgs(), adaptor.getArgs(), splitStrings)) {
      if (!isa<TileType>(arg.getType())) {
        return rewriter.notifyMatchFailure(op,
                                           "print operands should be tiles");
      }

      // store the tensor arg in a memref to call print
      auto loc = op.getLoc();
      auto tile = cast<TileType>(arg.getType());
      Value printValue = convertedArg;
      Type convertedType = getTypeConverter()->convertType(arg.getType());
      RankedTensorType tensorType;
      if (isScalarType(convertedType)) {
        tensorType = RankedTensorType::get({}, convertedType);
        printValue = tensor::FromElementsOp::create(rewriter, loc, tensorType,
                                                    convertedArg);
      } else {
        tensorType = cast<RankedTensorType>(convertedType);
      }

      auto memType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      auto unrankedMem =
          UnrankedMemRefType::get(tensorType.getElementType(), {});

      auto bufferizeOp = bufferization::ToBufferOp::create(
          rewriter, loc, memType, printValue, /*read_only=*/true);
      auto castOp = memref::CastOp::create(rewriter, loc, unrankedMem,
                                           bufferizeOp.getResult());

      cpu::PrintOp::create(rewriter, loc, str, castOp.getResult());
    }

    if (op.getNumOperands() < splitStrings.size() &&
        splitStrings.back() != "") {
      cpu::PrintOp::create(rewriter, op.getLoc(), splitStrings.back(), {});
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  // split strings at format, do not include the format specifier itself
  static SmallVector<StringRef, 8> splitFormatString(StringRef str) {
    SmallVector<StringRef, 8> str_splits;

    size_t pos = 0;
    size_t start = 0;

    while (pos < str.size()) {
      if (str[pos] != '%') {
        ++pos;
        continue;
      }

      if (pos + 1 < str.size() && str[pos + 1] == '%') {
        pos += 2;
        continue;
      }

      str_splits.push_back(str.slice(start, pos));
      ++pos;

      while (pos < str.size() && !StringRef("df").contains(str[pos])) {
        ++pos;
      }

      if (pos < str.size()) {
        ++pos;
      }

      start = pos;
    }

    str_splits.push_back(str.slice(start, str.size()));
    return str_splits;
  }
};

struct YieldPattern : public OpConversionPattern<cuda_tile::YieldOp> {
  using OpConversionPattern<cuda_tile::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parent = op->getParentOp();
    if (dyn_cast<scf::IfOp>(*parent)) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
      return success();
    }
    if (dyn_cast<linalg::ReduceOp>(*parent)) {
      rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, adaptor.getOperands());
      return success();
    }
    // TODO: scan op also constains yield

    return failure();
  }
};

struct IfPattern : public OpConversionPattern<cuda_tile::IfOp> {
  using OpConversionPattern<cuda_tile::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto cond =
        extractScalarTensor(op.getLoc(), adaptor.getCondition(), rewriter);

    SmallVector<Type> types;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), types)
            .failed()) {
      return failure();
    }

    auto newOp =
        scf::IfOp::create(rewriter, op.getLoc(), types, cond, false, false);
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());

    if (op.getElseBlock() != nullptr) {
      rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                  newOp.getElseRegion().end());
    }

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ContinuePattern : public OpConversionPattern<cuda_tile::ContinueOp> {
  using OpConversionPattern<cuda_tile::ContinueOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ContinueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parent = op->getParentOp();
    // TODO: continue could be nested inside an if body. we cannot
    // directly convert to scf.yield in that case.
    // continue could also be in a loop op
    if (!isa<scf::ForOp>(parent)) {
      return failure();
    }

    auto operands = llvm::filter_to_vector(adaptor.getOperands(), [](auto v) {
      return !isa<cuda_tile::TokenType>(v.getType());
    });

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
    return success();
  }
};

struct ForPattern : public OpConversionPattern<cuda_tile::ForOp> {
  using OpConversionPattern<cuda_tile::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> initValues;
    SmallVector<uint32_t> droppedIndices;

    for (auto [idx, init] : llvm::enumerate(adaptor.getInitValues())) {
      if (isa<cuda_tile::TokenType>(init.getType())) {
        droppedIndices.push_back(idx);
        continue;
      }
      initValues.push_back(init);
    }

    auto newOp = scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), initValues, nullptr, adaptor.getUnsignedCmp());

    // scf::ForOp::create already made an empty body, remove it
    rewriter.eraseBlock(newOp.getBody());

    Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    Block &body = newRegion.front();

    // drop body args corresponding to token iter_args.
    for (uint32_t idx : llvm::reverse(droppedIndices)) {
      uint32_t argIndex = idx + 1; // +1 because body arg 0 is the loop iv
      body.eraseArgument(argIndex);
    }

    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return failure();
    }

    // replace only the non-token results
    // TODO: this will probably fail when building LLVM with assertions
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct LoadPtrTkoPattern : public OpConversionPattern<cuda_tile::LoadPtrTkoOp> {
  using OpConversionPattern<cuda_tile::LoadPtrTkoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrTy = getTypeConverter()->convertType(op.getResult());
    if (isScalarType(ptrTy)) {
      // TODO
      // assert(!op.getMask() && !op.getPaddingValue() &&
      //        "scalar ptr load was assumed not to have mask or padding val");
      rewriter.replaceOpWithNewOp<cpu::LoadPtrOp>(op, ptrTy,
                                                  adaptor.getSource());
      return success();
    }

    auto loadOp = rewriter.replaceOpWithNewOp<cpu::LoadPtrTileOp>(
        op, ptrTy, adaptor.getSource(), adaptor.getMask(),
        adaptor.getPaddingValue());
    return success();
  }
};

struct StorePtrTkoPattern
    : public OpConversionPattern<cuda_tile::StorePtrTkoOp> {
  using OpConversionPattern<cuda_tile::StorePtrTkoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isScalarValue(adaptor.getDestination())) {
      // assert(!op.getMask() &&
      //        "scalar ptr store was assumed not to have a mask");
      rewriter.replaceOpWithNewOp<cpu::StorePtrOp>(op, adaptor.getDestination(),
                                                   adaptor.getValue());
      return success();
    }

    rewriter.replaceOpWithNewOp<cpu::StorePtrTileOp>(
        op, adaptor.getDestination(), adaptor.getValue(), adaptor.getMask());
    return success();
  }
};

struct MakeTensorViewPattern
    : public OpConversionPattern<cuda_tile::MakeTensorViewOp> {
  using OpConversionPattern<cuda_tile::MakeTensorViewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MakeTensorViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!memrefType) {
      return failure();
    }

    Location loc = op.getLoc();
    Value base = extractScalarTensor(loc, adaptor.getBase(), rewriter);

    SmallVector<Value> dynamicSizes =
        llvm::map_to_vector(adaptor.getDynamicShape(), [&](auto size) {
          return extractScalarTensorAsIndex(loc, size, rewriter);
        });

    SmallVector<Value> dynamicStrides =
        llvm::map_to_vector(adaptor.getDynamicStrides(), [&](auto stride) {
          return extractScalarTensorAsIndex(loc, stride, rewriter);
        });

    rewriter.replaceOpWithNewOp<cpu::MakeMemRefOp>(
        op, memrefType, base, dynamicSizes, dynamicStrides);
    return success();
  }
};

struct MakePartitionViewPattern
    : public OpConversionPattern<cuda_tile::MakePartitionViewOp> {
  using OpConversionPattern<
      cuda_tile::MakePartitionViewOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::MakePartitionViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // partition type will be used in LoadViewTkoPattern directly, no op needed
    rewriter.replaceOp(op, adaptor.getTensorView());
    return success();
  }
};

// convert partition view tile indices to element offsets
static FailureOr<SmallVector<Value>>
calculateViewOffsets(ConversionPatternRewriter &rewriter, Location loc,
                     PartitionViewType pvt, ValueRange indices) {
  for (auto [index, dimMap] : llvm::enumerate(pvt.getDimMap())) {
    if (index != dimMap) {
      return failure(); // TODO
    }
  }

  auto shape = pvt.getTileShape().asArrayRef();
  SmallVector<Value> offsets;
  for (auto [size, idx] : llvm::zip_equal(shape, indices)) {
    auto c = arith::ConstantIndexOp::create(rewriter, loc, size);
    auto idxScalar = extractScalarTensorAsIndex(loc, idx, rewriter);
    auto off = arith::MulIOp::create(rewriter, loc, idxScalar, c);
    offsets.push_back(off);
  }

  return offsets;
}

struct LoadViewTkoPattern
    : public OpConversionPattern<cuda_tile::LoadViewTkoOp> {
  using OpConversionPattern<cuda_tile::LoadViewTkoOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::LoadViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto view = dyn_cast<PartitionViewType>(op.getView().getType());
    if (!view) {
      // partition view is the only subview type in cuda tile 13.3
      return failure();
    }

    auto offsets =
        calculateViewOffsets(rewriter, op.getLoc(), view, adaptor.getIndex());
    if (failed(offsets)) {
      return failure();
    }

    auto resTy = getTypeConverter()->convertType(op.getResult(0));
    auto newOp = cpu::LoadMemRefTileOp::create(rewriter, op.getLoc(), resTy,
                                               adaptor.getView(), *offsets,
                                               view.getPaddingValue());

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct StoreViewTkoPattern
    : public OpConversionPattern<cuda_tile::StoreViewTkoOp> {
  using OpConversionPattern<cuda_tile::StoreViewTkoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StoreViewTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto view = dyn_cast<PartitionViewType>(op.getView().getType());
    if (!view) {
      // partition view is the only subview type in cuda tile 13.3
      return failure();
    }

    auto offsets =
        calculateViewOffsets(rewriter, op.getLoc(), view, adaptor.getIndex());
    if (failed(offsets)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<cpu::StoreMemRefTileOp>(
        op, adaptor.getTile(), adaptor.getView(), *offsets);
    return success();
  }
};

struct GetIndexSpaceShapePattern
    : public OpConversionPattern<cuda_tile::GetIndexSpaceShapeOp> {
  using OpConversionPattern<
      cuda_tile::GetIndexSpaceShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetIndexSpaceShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<MakePartitionViewOp>(op.getSrc().getDefiningOp())) {
      return failure();
    }

    auto pview = cast<MakePartitionViewOp>(op.getSrc().getDefiningOp());
    auto tview = pview.getTensorView();

    auto tviewTy = tview.getType();

    SmallVector<Value> tviewSizes;
    SmallVector<Type> elemTys;
    for (auto [i, s] : llvm::enumerate(tviewTy.getShape())) {
      auto elemTy = cast<TileType>(op.getType(i)).getElementType();
      elemTys.push_back(elemTy);

      if (s == ShapedType::kDynamic) {
        auto dimOp = memref::DimOp::create(rewriter, op.getLoc(), adaptor.getSrc(), i);
        Value val = arith::IndexCastUIOp::create(rewriter, op.getLoc(), elemTy,
                                                 dimOp, true);
        tviewSizes.push_back(val);
      } else {
        Value constant = arith::ConstantOp::create(
            rewriter, op.getLoc(), rewriter.getIntegerAttr(elemTy, s));
        tviewSizes.push_back(constant);
      }
    }

    SmallVector<Value> newOps;
    auto pviewTy = pview.getType();
    for (auto [shape, type, pShape] :
         llvm::zip(tviewSizes, elemTys, pviewTy.getTileShape().asArrayRef())) {
      APInt apValue(type.getIntOrFloatBitWidth(), pShape);
      auto pviewSize = arith::ConstantOp::create(
          rewriter, op.getLoc(), type, rewriter.getIntegerAttr(type, apValue));

      Value div = arith::CeilDivUIOp::create(rewriter, op.getLoc(), shape, pviewSize);
      newOps.push_back(div);
    }

    rewriter.replaceOp(op, newOps);
    return success();
  }
};

struct GetTensorShapePattern
    : public OpConversionPattern<cuda_tile::GetTensorShapeOp> {
  using OpConversionPattern<cuda_tile::GetTensorShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetTensorShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> dimOps;
    for (int64_t i = 0; i < op.getNumResults(); i++) {
      Value memOp =
          memref::DimOp::create(rewriter, op.getLoc(), adaptor.getSrc());
      dimOps.push_back(memOp);
    }

    rewriter.replaceOp(op, dimOps);
    return success();
  }
};

struct GetNumTileBlocksPattern
    : public OpConversionPattern<cuda_tile::GetNumTileBlocksOp> {
  using OpConversionPattern<cuda_tile::GetNumTileBlocksOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::GetNumTileBlocksOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cpu::GetNumBlocksOp>(op);
    return success();
  }
};

struct GetTileBlockIdPattern
    : public OpConversionPattern<cuda_tile::GetTileBlockIdOp> {
  using OpConversionPattern<cuda_tile::GetTileBlockIdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::GetTileBlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cpu::GetBlockIdOp>(op);
    return success();
  }
};

struct AssumePattern : public OpConversionPattern<cuda_tile::AssumeOp> {
  using OpConversionPattern<cuda_tile::AssumeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: could maybe use llvm.assume?
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

struct MakeTokenPattern : public OpConversionPattern<cuda_tile::MakeTokenOp> {
  using OpConversionPattern<cuda_tile::MakeTokenOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::MakeTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // the tile kernel becomes a single thread on the cpu, so just ignore
    // tokens
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename Op>
struct ForwardPtrI64Source : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // the typeconverter already converts all cuda tile ptrs to i64 so nothing
    // needs to be done here
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

using PtrToPtrPattern = ForwardPtrI64Source<cuda_tile::PtrToPtrOp>;
using IntToPtrPattern = ForwardPtrI64Source<cuda_tile::IntToPtrOp>;
using PtrToIntPattern = ForwardPtrI64Source<cuda_tile::PtrToIntOp>;

struct MoveOutOfCudaTileModule
    : public OpConversionPattern<cuda_tile::ModuleOp> {
  using OpConversionPattern<cuda_tile::ModuleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // NOTE: we assume that there are no other functions defined inside the
    // mlir::ModuleOp, otherwise there might be a collision

    mlir::ModuleOp parent = op->getParentOfType<mlir::ModuleOp>();
    Block &parentBlock = parent.getBodyRegion().front();

    for (auto &innerOp :
         llvm::make_early_inc_range(op.getBodyRegion().front())) {
      innerOp.moveBefore(&parentBlock, parentBlock.end());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCudaTileToStandard
    : public mlir::cuda_tile::cpu::impl::ConvertCudaTileToStandardBase<
          ConvertCudaTileToStandard> {
  using ConvertCudaTileToStandardBase::ConvertCudaTileToStandardBase;

  ConvertCudaTileToStandard() : ConvertCudaTileToStandardBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    CudaTileTypeConverter typeConverter;

    RewritePatternSet patterns(context);
    patterns
        .add<EntryPattern, ReturnPattern, ConstantPattern, IotaPattern,
             ReshapePattern, BroadcastPattern, OffsetPattern, CatPattern,
             ExtractPattern, PermutePattern, SelectPattern, ReducePattern,
             AddIPattern, SubIPattern, CmpIPattern, ShLIPattern, ShRIPattern,
             MulIPattern, DivIPattern, NegIPattern, MulHiIPattern, OrIPattern,
             XOrIPattern, AndIPattern, MaxIPattern, MinIPattern, RemIPattern,
             AbsIPattern, FloorPattern, CeilPattern, AbsFPattern, Atan2Pattern,
             CoshPattern, CosPattern, ExpPattern, Log2Pattern, NegFPattern,
             SinhPattern, SinPattern, TanPattern, PowPattern, AddFPattern,
             SubFPattern, MulFPattern, DivFPattern, Exp2Pattern, FmaPattern,
             MaxFPattern, MinFPattern, RsqrtPattern, LogPattern, SqrtPattern,
             SqrtPattern, TanHPattern, RemFPattern, MmaFPattern, BitcastPattern,
             ExtiPattern, FToIPattern, FToFPattern, IToFPattern, TruncIPattern,
             MmaIPattern, YieldPattern, IfPattern, ForPattern, ContinuePattern,
             PrintTkoPattern, LoadPtrTkoPattern, StorePtrTkoPattern,
             MakeTensorViewPattern, MakePartitionViewPattern,
             LoadViewTkoPattern, StoreViewTkoPattern, GetTensorShapePattern,
             GetIndexSpaceShapePattern, MakeTokenPattern, PtrToPtrPattern,
             PtrToIntPattern, IntToPtrPattern, GetTileBlockIdPattern,
             GetNumTileBlocksPattern, AssumePattern, MoveOutOfCudaTileModule>(
            typeConverter, context);

    target.addIllegalDialect<CudaTileDialect>();
    target.addLegalDialect<
        arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
        bufferization::BufferizationDialect, linalg::LinalgDialect,
        tensor::TensorDialect, math::MathDialect, scf::SCFDialect,
        cpu::CudaTileCPUDialect>();

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir {
namespace cuda_tile {
namespace cpu {

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertCudaTileToStandard() {
  return std::make_unique<ConvertCudaTileToStandard>();
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir
