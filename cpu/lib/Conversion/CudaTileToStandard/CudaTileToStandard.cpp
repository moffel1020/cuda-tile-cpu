#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
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

    addConversion([&](cuda_tile::TileType type) -> Type {
      // if (auto ptrType =
      //         dyn_cast<cuda_tile::PointerType>(type.getElementType())) {
      //   return convertCudaTilePtrToPtr(ptrType);
      // }
      return convertTileToTensor(type);
    });
  }

private:
  Type convertTileToTensor(cuda_tile::TileType type) const {
    auto shape = type.getShape();
    Type elementType = type.getElementType();
    return RankedTensorType::get(shape, elementType);
  }

  // Type convertCudaTilePtrToPtr(cuda_tile::PointerType ptrType) const {
  //   return
  //   ptr::PtrType::get(ptr::GenericSpaceAttr::get(ptrType.getContext()));
  // }
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

struct EntryPattern : public OpConversionPattern<cuda_tile::EntryOp> {
  using OpConversionPattern<cuda_tile::EntryOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::EntryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

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
    auto tensorType =
        RankedTensorType::get(oldType.getShape(), oldType.getElementType());

    auto oldAttr = dyn_cast<DenseElementsAttr>(op.getValueAttr());
    if (!oldAttr) {
      return failure();
    }

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

    // auto opType = op.getResult().getType();
    // auto shape = opType.getShape();
    // if (shape.size() != 1) {
    //   return rewriter.notifyMatchFailure(op, "1d shape expected for iota
    //   op");
    // }

    // auto width = opType.getElementType().getIntOrFloatBitWidth();
    // SmallVector<APInt> indices(opType.getNumElements());
    // std::iota(indices.begin(), indices.end(), APInt(width, 0));

    // auto vecType = VectorType::get(shape, opType.getElementType());
    // rewriter.replaceOpWithNewOp<arith::ConstantOp>(
    //     op, DenseElementsAttr::get(vecType, indices));

    return failure();
  }
};

struct BroadcastPattern : public OpConversionPattern<cuda_tile::BroadcastOp> {
  using OpConversionPattern<cuda_tile::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto opTy = op.getType();
    auto rank = opTy.getRank();

    AffineMap inputMap =
        getBroadcastInputMap(op.getSource().getType().getShape(),
                             opTy.getRank(), rewriter.getContext());
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(
        opTy.getRank(), rewriter.getContext());

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());
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
    auto tensorType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, tensorType,
                                                   adaptor.getSource());
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
    auto resType =
        cast<TensorType>(getTypeConverter()->convertType(op.getResult()));
    auto resShape = resType.getShape();
    SmallVector<int64_t> staticStrides(resType.getRank(), 1);
    SmallVector<int64_t> staticOffsets(resType.getRank(), ShapedType::kDynamic);

    SmallVector<Value> offsets;
    for (auto [size, idx] : llvm::zip(resShape, adaptor.getIndices())) {
      auto shapeSizeOp =
          arith::ConstantIndexOp::create(rewriter, op.getLoc(), /*value=*/size);
      // TODO: check if this unpack is optimized away, if not try creating
      // scalar constants and then doing tensor.from_elements for other users
      auto unpackedIdx =
          tensor::ExtractOp::create(rewriter, op.getLoc(), idx, {});
      auto castedUnpackedIdx = arith::IndexCastUIOp::create(
          rewriter, op.getLoc(), IndexType::get(getContext()), unpackedIdx);

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

template <typename T, typename U>
struct ConvertWithMap : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    auto mapOp = linalg::MapOp::create(
        rewriter, op.getLoc(), adaptor.getOperands(), empty,
        [](OpBuilder &b, Location loc, ValueRange args) {
          auto newOp = U::create(b, loc, args.drop_back());
          linalg::YieldOp::create(b, loc, newOp.getResult());
        });

    rewriter.replaceOp(op, mapOp);
    return success();
  }
};

template <typename T, typename U>
struct ReplaceWithLinalg : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());
    rewriter.replaceOpWithNewOp<U>(op, adaptor.getOperands(),
                                   empty.getResult());
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
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    auto newOp = [&]() -> Operation * {
      switch (signedness) {
      case Signedness::Unsigned:
        return linalg::MapOp::create(
            rewriter, op.getLoc(), adaptor.getOperands(), empty,
            [](OpBuilder &b, Location loc, ValueRange args) {
              Value mapOp = UnsignedMapOp::create(b, loc, args.drop_back());
              linalg::YieldOp::create(b, loc, mapOp);
            });
      case Signedness::Signed:
        return SignedOp::create(rewriter, op.getLoc(),
                                ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                                ValueRange{empty});
      default:
        llvm_unreachable(
            "only unsigned and signed are valid"); // suppress warning because
                                                   // of templated class
      }
    }();

    rewriter.replaceOp(op, newOp);
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
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    if (op.getPropagateNan()) {
      rewriter.replaceOpWithNewOp<PropOp>(op, adaptor.getOperands(),
                                          ValueRange{empty});
    } else {
      rewriter.replaceOpWithNewOp<linalg::MapOp>(
          op, adaptor.getOperands(), empty,
          [](OpBuilder &b, Location loc, ValueRange args) {
            Value mapOp = NoPropOp::create(b, loc, args.drop_back());
            linalg::YieldOp::create(b, loc, mapOp);
          });
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
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    auto newOp = [&]() -> Operation * {
      switch (signedness) {
      case Signedness::Unsigned:
        return linalg::MapOp::create(
            rewriter, op.getLoc(), adaptor.getOperands(), empty,
            [](OpBuilder &b, Location loc, ValueRange args) {
              Value mapOp = UnsignedOp::create(b, loc, args.drop_back());
              linalg::YieldOp::create(b, loc, mapOp);
            });
      case Signedness::Signed:
        return linalg::MapOp::create(
            rewriter, op.getLoc(), adaptor.getOperands(), empty,
            [](OpBuilder &b, Location loc, ValueRange args) {
              Value mapOp = SignedOp::create(b, loc, args.drop_back());
              linalg::YieldOp::create(b, loc, mapOp);
            });
      default:
        llvm_unreachable(
            "only unsigned and signed are valid"); // suppress warning because
                                                   // of templated class
      }
    }();

    rewriter.replaceOp(op, newOp);
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
    auto cmpPred = [&] {
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
        llvm_unreachable();
      }
    }();

    auto opTy = op.getType();
    auto boolType = IntegerType::get(getContext(), 1);
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         boolType);

    rewriter.replaceOpWithNewOp<linalg::MapOp>(
        op, adaptor.getOperands(), empty,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value newOp = arith::CmpIOp::create(b, loc, boolType, cmpPred,
                                              args[0], args[1]);
          linalg::YieldOp::create(b, loc, newOp);
        });

    return success();
  }
};

struct DivIPattern : public OpConversionPattern<cuda_tile::DivIOp> {
  using OpConversionPattern<cuda_tile::DivIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sign = op.getSignedness();
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    CreateMapOp createMap{rewriter, op.getLoc(), adaptor.getOperands(), empty};

    auto newOp = [&]() -> Operation * {
      using S = cuda_tile::Signedness;
      using RM = cuda_tile::RoundingMode;

      switch (op.getRounding()) {
      case RM::NEGATIVE_INF:
        return createMap.withOp<arith::FloorDivSIOp>();
      case RM::POSITIVE_INF:
        return sign == S::Signed ? createMap.withOp<arith::CeilDivSIOp>()
                                 : createMap.withOp<arith::CeilDivUIOp>();
      case RM::ZERO:
        return sign == S::Signed
                   ? linalg::DivOp::create(rewriter, op.getLoc(),
                                           adaptor.getOperands(),
                                           ValueRange{empty})
                   : linalg::DivUnsignedOp::create(rewriter, op.getLoc(),
                                                   adaptor.getOperands(),
                                                   ValueRange{empty});
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

private:
  struct CreateMapOp {
    template <typename T>
    Operation *withOp() {
      return linalg::MapOp::create(
          rewriter, loc, args, init,
          [](OpBuilder &b, Location loc, ValueRange args) {
            Value val = T::create(b, loc, args.drop_back());
            linalg::YieldOp::create(b, loc, val);
          });
    }

    ConversionPatternRewriter &rewriter;
    Location loc;
    ValueRange args;
    Value init;
  };
};

struct NegIPattern : public OpConversionPattern<cuda_tile::NegIOp> {
  using OpConversionPattern<cuda_tile::NegIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(cuda_tile::NegIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) {

    auto elemTy = op.getType().getElementType();
    auto c0 = arith::ConstantIntOp::create(rewriter, op.getLoc(), elemTy, 0);

    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    rewriter.replaceOpWithNewOp<linalg::MapOp>(
        op, adaptor.getOperands(), empty,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value subOp = arith::SubIOp::create(
              b, loc, args[0], c0, convertOverflowFlags(op.getOverflow()));
          linalg::YieldOp::create(b, loc, subOp);
        });

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

    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());
    // according to the spec, this op is only defined for unsigned integers
    rewriter.replaceOpWithNewOp<linalg::MapOp>(
        op, adaptor.getOperands(), empty,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto mulOp = arith::MulUIExtendedOp::create(b, loc, args[0], args[1]);
          linalg::YieldOp::create(b, loc, mulOp.getHigh());
        });

    return success();
  }
};

template <typename T, typename U>
static Operation *createConversionMapOp(T op, typename T::Adaptor adaptor,
                                        ConversionPatternRewriter &rewriter) {
  auto opTy = op.getType();
  auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                       opTy.getElementType());
  return linalg::MapOp::create(
      rewriter, op.getLoc(), adaptor.getOperands(), empty,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value extOp =
            U::create(b, loc, empty.getType().getElementType(), args[0]);
        linalg::YieldOp::create(b, loc, extOp);
      });
}

struct BitcastPattern : public OpConversionPattern<cuda_tile::BitcastOp> {
  using OpConversionPattern<cuda_tile::BitcastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = createConversionMapOp<cuda_tile::BitcastOp, arith::BitcastOp>(
        op, adaptor, rewriter);
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
        return createConversionMapOp<cuda_tile::ExtIOp, arith::ExtSIOp>(
            op, adaptor, rewriter);
      } else {
        return createConversionMapOp<cuda_tile::ExtIOp, arith::ExtUIOp>(
            op, adaptor, rewriter);
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
        return createConversionMapOp<cuda_tile::FToIOp, arith::FPToSIOp>(
            op, adaptor, rewriter);
      } else {
        return createConversionMapOp<cuda_tile::FToIOp, arith::FPToUIOp>(
            op, adaptor, rewriter);
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
      // TODO: update llvm version and use arith.convertf here, for bf16 to f16
      return rewriter.notifyMatchFailure(
          op, "unsupported float to float conversion");
    }

    auto roundingMode = convertRoundingMode(op.getRoundingMode());

    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());
    auto newOp = [&] {
      if (fromWidth > toWidth) {
        return linalg::MapOp::create(
            rewriter, op.getLoc(), adaptor.getFrom(), empty,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value extOp = arith::ExtFOp::create(b, loc, toType, args[0]);
              linalg::YieldOp::create(b, loc, extOp);
            });
      } else /* if (fromWidth < toWidth) */ {
        return linalg::MapOp::create(
            rewriter, op.getLoc(), adaptor.getFrom(), empty,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value truncOp =
                  !roundingMode
                      ? arith::TruncFOp::create(b, loc, toType, args[0])
                      : arith::TruncFOp::create(
                            b, loc, toType, args[0],
                            arith::RoundingModeAttr::get(getContext(),
                                                         roundingMode.value()),
                            {});
              linalg::YieldOp::create(b, loc, truncOp);
            });
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
        return createConversionMapOp<cuda_tile::IToFOp, arith::SIToFPOp>(
            op, adaptor, rewriter);
      } else {
        return createConversionMapOp<cuda_tile::IToFOp, arith::UIToFPOp>(
            op, adaptor, rewriter);
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
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());

    rewriter.replaceOpWithNewOp<linalg::MapOp>(
        op, adaptor.getFrom(), empty,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value truncOp =
              arith::TruncIOp::create(b, loc, opTy.getElementType(), args[0],
                                      convertOverflowFlags(op.getOverflow()));
          linalg::YieldOp::create(b, loc, truncOp);
        });

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
    auto arithPred = [&] {
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
        llvm_unreachable();
      }
    }();

    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());
    rewriter.replaceOpWithNewOp<linalg::MapOp>(
        op, adaptor.getOperands(), empty,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value cmp =
              arith::CmpFOp::create(b, loc, arithPred, args[0], args[1]);
          linalg::YieldOp::create(b, loc, cmp);
        });

    return success();
  }
};

struct MmaIPattern : public OpConversionPattern<cuda_tile::MmaIOp> {
  using OpConversionPattern<cuda_tile::MmaIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cuda_tile::MmaIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // lhs and rhs must be i8 and acc must be i32 according to spec

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
using AndIPattern = ConvertWithMap<cuda_tile::AndIOp, arith::AndIOp>;

// integer
// TODO: ignoring overflow information. could preserve some by using linalg
// generic/map
using AbsIPattern = ConvertWithMap<cuda_tile::AbsIOp, math::AbsIOp>;
using AddIPattern = ReplaceWithLinalg<cuda_tile::AddIOp,
                                      linalg::AddOp>; // ignore int overflow
using SubIPattern = ReplaceWithLinalg<cuda_tile::SubIOp,
                                      linalg::SubOp>; // ignore int overflow
using MulIPattern = ReplaceWithLinalg<cuda_tile::MulIOp,
                                      linalg::MulOp>; // ignore int overflow
using ShLIPattern = ConvertWithMap<cuda_tile::ShLIOp,
                                   arith::ShLIOp>; // ignore int overflow
using OrIPattern = ConvertWithMap<cuda_tile::OrIOp, arith::OrIOp>;
using XOrIPattern = ConvertWithMap<cuda_tile::XOrIOp, arith::XOrIOp>;
using MaxIPattern =
    MaxIMinIPattern<cuda_tile::MaxIOp, linalg::MaxOp, arith::MaxUIOp>;
using MinIPattern =
    MaxIMinIPattern<cuda_tile::MinIOp, linalg::MinOp, arith::MinUIOp>;
using ShRIPattern =
    SignedUnsignedPattern<cuda_tile::ShRIOp, arith::ShRSIOp, arith::ShRUIOp>;
using RemIPattern =
    SignedUnsignedPattern<cuda_tile::RemIOp, arith::RemSIOp, arith::RemUIOp>;

// floating point
using AbsFPattern = ReplaceWithLinalg<cuda_tile::AbsFOp, linalg::AbsOp>;
using CeilPattern = ReplaceWithLinalg<cuda_tile::CeilOp, linalg::CeilOp>;
using FloorPattern = ReplaceWithLinalg<cuda_tile::FloorOp, linalg::FloorOp>;
using Atan2Pattern = ConvertWithMap<cuda_tile::Atan2Op, math::Atan2Op>;
using CoshPattern = ConvertWithMap<cuda_tile::CosHOp, math::CoshOp>;
using CosPattern = ConvertWithMap<cuda_tile::CosOp, math::CosOp>;
using ExpPattern = ReplaceWithLinalg<cuda_tile::ExpOp, linalg::ExpOp>;
using Log2Pattern = ConvertWithMap<cuda_tile::Log2Op, math::Log2Op>;
using LogPattern = ReplaceWithLinalg<cuda_tile::LogOp, linalg::LogOp>;
using NegFPattern = ReplaceWithLinalg<cuda_tile::NegFOp, linalg::NegFOp>;
using SinhPattern = ConvertWithMap<cuda_tile::SinHOp, math::SinhOp>;
using SinPattern = ConvertWithMap<cuda_tile::SinOp, math::SinOp>;
using TanPattern = ConvertWithMap<cuda_tile::TanOp, math::TanOp>;
// TODO: specialize to square for pow 2
using PowPattern = ReplaceWithLinalg<cuda_tile::PowOp, linalg::PowFOp>;
using AddFPattern = ReplaceWithLinalg<cuda_tile::AddFOp,
                                      linalg::AddOp>; // ignore rounding and ftz
using DivFPattern = ReplaceWithLinalg<cuda_tile::DivFOp,
                                      linalg::DivOp>; // ignore rounding and ftz
using Exp2Pattern =
    ConvertWithMap<cuda_tile::Exp2Op, math::Exp2Op>; // ignore ftz
using FmaPattern =
    ConvertWithMap<cuda_tile::FmaOp, math::FmaOp>; // ignore rounding and ftz
using MaxFPattern =
    MaxFMinFPattern<cuda_tile::MaxFOp, linalg::MaxOp, arith::MaxNumFOp>;
using MinFPattern =
    MaxFMinFPattern<cuda_tile::MinFOp, linalg::MinOp, arith::MinNumFOp>;
using MulFPattern = ReplaceWithLinalg<cuda_tile::MulFOp,
                                      linalg::MulOp>; // ignore rounding and ftz
using RsqrtPattern =
    ReplaceWithLinalg<cuda_tile::RsqrtOp, linalg::RsqrtOp>; // ingore ftz
using SubFPattern = ReplaceWithLinalg<cuda_tile::SubFOp,
                                      linalg::SubOp>; // ignore rounding and ftz
using SqrtPattern =
    ReplaceWithLinalg<cuda_tile::SqrtOp,
                      linalg::SqrtOp>; // ignore rounding and ftz
using TanHPattern =
    ReplaceWithLinalg<cuda_tile::TanHOp, linalg::TanhOp>; // ignore rounding
using RemFPattern = ConvertWithMap<cuda_tile::RemFOp, arith::RemFOp>;

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

    for (auto [arg, str] : llvm::zip(op.getArgs(), splitStrings)) {
      if (!isa<TileType>(arg.getType())) {
        return rewriter.notifyMatchFailure(op,
                                           "print operands should be tiles");
      }

      // store the tensor arg in a memref to call print
      auto loc = op.getLoc();
      auto tile = cast<TileType>(arg.getType());
      auto memType = MemRefType::get(tile.getShape(), tile.getElementType());
      auto unrankedMem = UnrankedMemRefType::get(tile.getElementType(), {});

      auto bufferizeOp = bufferization::ToBufferOp::create(
          rewriter, loc, memType, rewriter.getRemappedValue(arg),
          /*read_only=*/true);
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

    return failure();
  }
};

struct IfPattern : public OpConversionPattern<cuda_tile::IfOp> {
  using OpConversionPattern<cuda_tile::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // the original condition is in a tile, we just want the scalar
    auto cond = tensor::ExtractOp::create(rewriter, op.getLoc(),
                                          adaptor.getCondition(), {});

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

struct LoadPtrTkoPattern : public OpConversionPattern<cuda_tile::LoadPtrTkoOp> {
  using OpConversionPattern<cuda_tile::LoadPtrTkoOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    return success();
  }
};

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
             ReshapePattern, BroadcastPattern, CatPattern, ExtractPattern,
             PermutePattern, AddIPattern, SubIPattern, CmpIPattern, ShLIPattern,
             ShRIPattern, MulIPattern, DivIPattern, NegIPattern, MulHiIPattern,
             OrIPattern, XOrIPattern, AndIPattern, MaxIPattern, MinIPattern,
             RemIPattern, AbsIPattern, FloorPattern, CeilPattern, AbsFPattern,
             Atan2Pattern, CoshPattern, CosPattern, ExpPattern, Log2Pattern,
             NegFPattern, SinhPattern, SinPattern, TanPattern, PowPattern,
             AddFPattern, DivFPattern, Exp2Pattern, FmaPattern, MaxFPattern,
             MinFPattern, RsqrtPattern, SqrtPattern, SqrtPattern, TanHPattern,
             RemFPattern, MmaFPattern, BitcastPattern, ExtiPattern, FToIPattern,
             FToFPattern, IToFPattern, TruncIPattern, MmaIPattern, YieldPattern,
             IfPattern, PrintTkoPattern, MoveOutOfCudaTileModule>(typeConverter,
                                                                  context);

    target.addIllegalDialect<CudaTileDialect>();
    target.addLegalDialect<
        arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
        bufferization::BufferizationDialect, linalg::LinalgDialect,
        tensor::TensorDialect, math::MathDialect, ptr::PtrDialect,
        scf::SCFDialect, cuda_tile::cpu::CudaTileCPUDialect>();

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
