#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <memory>

namespace mlir {
namespace cuda_tile {
namespace cpu {

#define GEN_PASS_DEF_LOWERCUDATILECPUMEMOPS
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

namespace {

using namespace mlir;
using namespace mlir::cuda_tile;
using namespace llvm;

static bool hasOnlyZeroIndices(ValueRange indices) {
  return llvm::all_of(
      indices, [](Value index) { return matchPattern(index, m_Zero()); });
}
static SmallVector<OpFoldResult> toOpFold(ValueRange vr) {
  return llvm::map_to_vector(vr, [](Value v) { return OpFoldResult{v}; });
};

static Value readTensorAndCastTo1dVector(ConversionPatternRewriter &rewriter,
                                         Location loc, Value tensorVal,
                                         ShapedType origType,
                                         ArrayRef<Value> zeroIndices) {
  auto elemTy = origType.getElementType();
  auto vecTy = VectorType::get(origType.getShape(), elemTy);
  auto vec1dTy = VectorType::get({origType.getNumElements()}, elemTy);

  auto vec = vector::TransferReadOp::create(rewriter, loc, vecTy, tensorVal,
                                            zeroIndices, std::nullopt);
  auto vec1d = vector::ShapeCastOp::create(rewriter, loc, vec1dTy, vec);
  return vec1d;
}

struct GatherTilePattern : public OpConversionPattern<cpu::GatherTileOp> {
  using OpConversionPattern<cpu::GatherTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::GatherTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = op.getType();
    auto offsetsType = op.getOffsets().getType();
    int64_t numElements = resultType.getNumElements();

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeroIndices(resultType.getRank(), zero);

    Value mask;
    if (op.getMask()) {
      mask = readTensorAndCastTo1dVector(rewriter, loc, op.getMask(),
                                         op.getMask().getType(), zeroIndices);
    } else {
      auto maskType = VectorType::get({numElements}, rewriter.getI1Type());
      auto trueSplat =
          DenseElementsAttr::get(maskType, rewriter.getBoolAttr(true));
      mask = arith::ConstantOp::create(rewriter, loc, maskType, trueSplat);
    }

    auto vectorType =
        VectorType::get({numElements}, resultType.getElementType());
    Value passThru;
    if (op.getPaddingValue()) {
      passThru = readTensorAndCastTo1dVector(
          rewriter, loc, op.getPaddingValue(), op.getPaddingValue().getType(),
          zeroIndices);
    } else {
      passThru = ub::PoisonOp::create(rewriter, loc, vectorType);
    }

    auto baseType =
        MemRefType::get({ShapedType::kDynamic}, resultType.getElementType());
    Value baseSize = arith::ConstantIndexOp::create(rewriter, loc, numElements);
    Value base =
        cpu::MakeMemRefOp::create(rewriter, loc, baseType, op.getBase(),
                                  ValueRange{baseSize}, ValueRange{});

    Value indices = readTensorAndCastTo1dVector(rewriter, loc, op.getOffsets(),
                                                offsetsType, zeroIndices);
    Value gathered =
        vector::GatherOp::create(rewriter, loc, vectorType, base,
                                 ValueRange{zero}, indices, mask, passThru);

    auto resultVectorType =
        VectorType::get(resultType.getShape(), resultType.getElementType());
    Value shaped =
        vector::ShapeCastOp::create(rewriter, loc, resultVectorType, gathered);
    Value empty = tensor::EmptyOp::create(rewriter, loc, resultType, {});
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(op, shaped, empty,
                                                         zeroIndices);
    return success();
  }
};

struct ScatterTilePattern : public OpConversionPattern<cpu::ScatterTileOp> {
  using OpConversionPattern<cpu::ScatterTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::ScatterTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto valueType = op.getValue().getType();
    auto offsetsType = op.getOffsets().getType();
    int64_t numElements = valueType.getNumElements();

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeroIndices(valueType.getRank(), zero);

    Value mask;
    if (op.getMask()) {
      mask = readTensorAndCastTo1dVector(rewriter, loc, op.getMask(),
                                         op.getMask().getType(), zeroIndices);
    } else {
      auto maskType = VectorType::get({numElements}, rewriter.getI1Type());
      auto trueSplat =
          DenseElementsAttr::get(maskType, rewriter.getBoolAttr(true));
      mask = arith::ConstantOp::create(rewriter, loc, maskType, trueSplat);
    }

    auto baseType =
        MemRefType::get({ShapedType::kDynamic}, valueType.getElementType());
    Value baseSize = arith::ConstantIndexOp::create(rewriter, loc, numElements);
    Value base =
        cpu::MakeMemRefOp::create(rewriter, loc, baseType, op.getBase(),
                                  ValueRange{baseSize}, ValueRange{});

    Value indices = readTensorAndCastTo1dVector(rewriter, loc, op.getOffsets(),
                                                offsetsType, zeroIndices);
    Value value = readTensorAndCastTo1dVector(rewriter, loc, op.getValue(),
                                              valueType, zeroIndices);

    auto scatter =
        vector::ScatterOp::create(rewriter, loc, /*resultType=*/nullptr, base,
                                  ValueRange{zero}, indices, mask, value);
    rewriter.replaceOp(op, scatter.getResults());
    return success();
  }
};

// innerBodyBuilder must always create a yield op of the innermost loop.
// if useInnerIterArg is set to to true, that yield must return a result.
// the result is propagated to the outermost loop
static scf::ForOp
createLoopNest(ConversionPatternRewriter &rewriter, Location loc,
               ArrayRef<int64_t> shape, bool useInnerIterArg,
               ValueRange iterArgs,
               std::function<void(OpBuilder &, Location, ValueRange /*ivs*/,
                                  ValueRange /*iterArgs*/)>
                   innerBodyBuilder) {
  OpBuilder::InsertionGuard insertGuard(rewriter);
  SmallVector<Value> ivs;
  SmallVector<scf::ForOp> forOps;

  auto removeYieldTerminator = [&](scf::ForOp op) {
    for (auto y : op.getOps<scf::YieldOp>()) {
      rewriter.eraseOp(y);
    }
  };

  auto dim = shape.size();
  assert(dim >= 1);

  auto lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);

  // create nested for loops
  for (size_t d = 0; d < dim; d++) {
    auto ub = arith::ConstantIndexOp::create(rewriter, loc, shape[d]);

    auto lastIterArgs =
        forOps.empty() ? iterArgs : forOps.back().getRegionIterArgs();

    scf::ForOp forOp;
    // user creates body of innermost loop
    if (d == dim - 1) {
      forOp = scf::ForOp::create(
          rewriter, loc, lb, ub, step, lastIterArgs,
          [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            ivs.push_back(iv);
            innerBodyBuilder(b, loc, ivs, iterArgs);
          });
    } else {
      forOp = scf::ForOp::create(rewriter, loc, lb, ub, step, lastIterArgs);
      ivs.push_back(forOp.getInductionVar());
    }

    rewriter.setInsertionPointToStart(forOp.getBody());
    forOps.push_back(forOp);
  }

  // propagate yielded result to outermost loop
  if (useInnerIterArg) {
    for (int i = 0; i < forOps.size() - 1; i++) {
      auto outerFor = forOps[i];
      auto innerFor = forOps[i + 1];

      rewriter.setInsertionPointToEnd(outerFor.getBody());
      scf::YieldOp::create(rewriter, loc, innerFor.getResults());
    }
  }

  return forOps.front();
}

struct LoadPtrTilePattern : public OpConversionPattern<cpu::LoadPtrTileOp> {
  using OpConversionPattern<cpu::LoadPtrTileOp>::OpConversionPattern;

  static void lower0dLoad(cpu::LoadPtrTileOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) {
    auto elemTy = cast<ShapedType>(op.getResult().getType()).getElementType();
    auto ptr =
        tensor::ExtractOp::create(rewriter, op.getLoc(), op.getSource(), {});
    auto val = cpu::LoadPtrOp::create(rewriter, op.getLoc(), elemTy, ptr);
    auto tensor = tensor::FromElementsOp::create(rewriter, op.getLoc(), {val});
    rewriter.replaceOp(op, val);
  }

  static void lowerNdLoadWithLoop(cpu::LoadPtrTileOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) {
    auto ptrTileTy = op.getSource().getType();
    auto c0 = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    SmallVector<Value> zeroIndices;
    for (size_t i = 0; i < ptrTileTy.getShape().size(); i++) {
      zeroIndices.push_back(c0);
    }

    auto sourceVec1d =
        readTensorAndCastTo1dVector(rewriter, op.getLoc(), op.getSource(),
                                    op.getSource().getType(), zeroIndices);
    Value maskVec1d = nullptr;
    if (op.getMask()) {
      maskVec1d =
          readTensorAndCastTo1dVector(rewriter, op.getLoc(), op.getMask(),
                                      op.getMask().getType(), zeroIndices);
    }

    // TODO: possible optimization. if vector padding value is from a broadcast,
    // we could skip the vector extract and just get the value
    Value paddingValueVec1d = nullptr;
    if (op.getPaddingValue()) {
      paddingValueVec1d = readTensorAndCastTo1dVector(
          rewriter, op.getLoc(), op.getPaddingValue(),
          op.getPaddingValue().getType(), zeroIndices);
    }

    auto resTy = op.getResult().getType();
    auto resElemTy = resTy.getElementType();
    auto numElems = ptrTileTy.getNumElements();
    Value initVec = ub::PoisonOp::create(
        rewriter, op.getLoc(),
        VectorType::get({numElems}, resTy.getElementType()));

    auto lb = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto ub = arith::ConstantIndexOp::create(rewriter, op.getLoc(), numElems);
    auto step = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);

    auto loop = scf::ForOp::create(
        rewriter, op.getLoc(), lb, ub, step, initVec,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          auto ptr = vector::ExtractOp::create(b, loc, sourceVec1d, iv);

          Value val;
          if (maskVec1d) {
            auto cond = vector::ExtractOp::create(b, loc, maskVec1d, iv);
            val = scf::IfOp::create(
                      b, loc, cond,
                      [&](OpBuilder &b, Location loc) {
                        Value v =
                            cpu::LoadPtrOp::create(b, loc, resElemTy, ptr);
                        scf::YieldOp::create(b, loc, v);
                      },
                      [&](OpBuilder &b, Location loc) {
                        Value v;
                        if (paddingValueVec1d) {
                          v = vector::ExtractOp::create(b, loc,
                                                        paddingValueVec1d, iv);
                        } else {
                          v = ub::PoisonOp::create(b, loc, resElemTy);
                        }
                        scf::YieldOp::create(b, loc, v);
                      })
                      .getResult(0);
          } else {
            val = cpu::LoadPtrOp::create(b, loc, resTy.getElementType(), ptr);
          }

          Value insert = vector::InsertOp::create(b, loc, val, iterArgs[0], iv);
          scf::YieldOp::create(b, loc, insert);
        });

    auto resVecTy = VectorType::get(op.getType().getShape(), resElemTy);
    auto shapeCast = vector::ShapeCastOp::create(rewriter, op.getLoc(),
                                                 resVecTy, loop.getResult(0));
    auto empty =
        tensor::EmptyOp::create(rewriter, op.getLoc(), op.getType(), {});
    auto write = vector::TransferWriteOp::create(rewriter, op.getLoc(),
                                                 shapeCast, empty, zeroIndices);
    rewriter.replaceOp(op, write);
  }

  LogicalResult
  matchAndRewrite(cpu::LoadPtrTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = op.getType();

    auto memTy = MemRefType::get(ty.getShape(), ty.getElementType());
    auto elemTy = cast<ShapedType>(op.getResult().getType()).getElementType();

    if (ty.getRank() == 0) {
      lower0dLoad(op, adaptor, rewriter);
      return success();
    }

    lowerNdLoadWithLoop(op, adaptor, rewriter);
    return success();
  }
};

struct StorePtrTilePattern : OpConversionPattern<cpu::StorePtrTileOp> {
  using OpConversionPattern<cpu::StorePtrTileOp>::OpConversionPattern;

  static void lower0dStore(cpu::StorePtrTileOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) {
    auto ptr =
        tensor::ExtractOp::create(rewriter, op.getLoc(), op.getDestination());
    auto val = tensor::ExtractOp::create(rewriter, op.getLoc(), op.getValue());
    auto storeOp = cpu::StorePtrOp::create(rewriter, op.getLoc(), ptr, val);
    rewriter.replaceOp(op, storeOp);
  }

  static void lowerNdStoreWithLoop(cpu::StorePtrTileOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) {
    auto ptrTileTy = op.getDestination().getType();

    auto c0 = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    SmallVector<Value> zeroIndices;
    for (size_t i = 0; i < ptrTileTy.getShape().size(); i++) {
      zeroIndices.push_back(c0);
    }

    auto ptrVec1d =
        readTensorAndCastTo1dVector(rewriter, op.getLoc(), op.getDestination(),
                                    op.getDestination().getType(), zeroIndices);
    auto valVec1d =
        readTensorAndCastTo1dVector(rewriter, op.getLoc(), op.getValue(),
                                    op.getValue().getType(), zeroIndices);

    Value maskVec1d = nullptr;
    if (op.getMask()) {
      maskVec1d =
          readTensorAndCastTo1dVector(rewriter, op.getLoc(), op.getMask(),
                                      op.getMask().getType(), zeroIndices);
    }

    auto numElems = ptrTileTy.getNumElements();
    auto lb = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto ub = arith::ConstantIndexOp::create(rewriter, op.getLoc(), numElems);
    auto step = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
    auto loop = scf::ForOp::create(
        rewriter, op.getLoc(), lb, ub, step, {},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
          auto ptr = vector::ExtractOp::create(b, loc, ptrVec1d, iv);
          auto val = vector::ExtractOp::create(b, loc, valVec1d, iv);
          if (maskVec1d) {
            auto cond = vector::ExtractOp::create(b, loc, maskVec1d, iv);
            scf::IfOp::create(b, loc, cond, [&](OpBuilder &b, Location loc) {
              cpu::StorePtrOp::create(b, loc, ptr, val);
              scf::YieldOp::create(b, loc);
            });
          } else {
            cpu::StorePtrOp::create(b, loc, ptr, val);
          }
          scf::YieldOp::create(b, loc);
        });

    rewriter.replaceOp(op, loop);
  }

  LogicalResult
  matchAndRewrite(cpu::StorePtrTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = op.getValue().getType();
    if (ty.getRank() == 0) {
      lower0dStore(op, adaptor, rewriter);
      return success();
    }

    lowerNdStoreWithLoop(op, adaptor, rewriter);
    return success();
  }
};

struct LoadMemRefTilePattern
    : public OpConversionPattern<cpu::LoadMemRefTileOp> {
  using OpConversionPattern<cpu::LoadMemRefTileOp>::OpConversionPattern;

  static FailureOr<TypedAttr>
  convertPaddingAttr(ConversionPatternRewriter &rewriter,
                     cuda_tile::PaddingValue val, Type elemTy) {
    if (!(elemTy.isFloat() ||
          (elemTy.isIntOrFloat() && val == PaddingValue::zero))) {
      return failure();
    }

    switch (val) {
    case PaddingValue::neg_zero: {
      llvm::APFloat floatVal(cast<FloatType>(elemTy).getFloatSemantics());
      floatVal.changeSign();
      return static_cast<TypedAttr>(rewriter.getFloatAttr(elemTy, floatVal));
    }
    case PaddingValue::nan:
      return static_cast<TypedAttr>(rewriter.getFloatAttr(
          elemTy, std::numeric_limits<double>::quiet_NaN()));
    case PaddingValue::pos_inf:
      return static_cast<TypedAttr>(rewriter.getFloatAttr(
          elemTy, std::numeric_limits<double>::infinity()));
    case PaddingValue::neg_inf:
      return static_cast<TypedAttr>(rewriter.getFloatAttr(
          elemTy, -std::numeric_limits<double>::infinity()));
    case PaddingValue::zero:
      return static_cast<TypedAttr>(rewriter.getZeroAttr(elemTy));
    default:
      return failure();
    }
  }

  LogicalResult
  matchAndRewrite(cpu::LoadMemRefTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = cast<RankedTensorType>(op.getType());

    auto c0 = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    SmallVector<Value> zeroIndices;
    SmallVector<bool> notInBounds;
    for (size_t i = 0; i < tileType.getShape().size(); i++) {
      zeroIndices.push_back(c0);
      notInBounds.push_back(false);
    }

    std::optional<Value> padding;
    if (auto paddingVal = op.getPaddingValue()) {
      auto attr =
          convertPaddingAttr(rewriter, *paddingVal, tileType.getElementType());
      if (failed(attr)) {
        return failure();
      }
      padding = arith::ConstantOp::create(rewriter, op.getLoc(), *attr);
    }

    auto vecTy =
        VectorType::get(tileType.getShape(), tileType.getElementType());
    auto vecRead = vector::TransferReadOp::create(
        rewriter, op.getLoc(), vecTy, op.getSource(), op.getOffsets(), padding,
        notInBounds);

    // this write should get optimized away by canonicalizer, there is likely
    // a transfer read after it
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), tileType, {});
    auto vecWrite = vector::TransferWriteOp::create(
        rewriter, op.getLoc(), vecRead, empty, zeroIndices, notInBounds);

    rewriter.replaceOp(op, vecWrite);
    return success();
  }
};

struct StoreMemRefTilePattern
    : public OpConversionPattern<cpu::StoreMemRefTileOp> {
  using OpConversionPattern<cpu::StoreMemRefTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::StoreMemRefTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = cast<RankedTensorType>(op.getValue().getType());

    auto c0 = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    SmallVector<Value> zeroIndices;
    SmallVector<bool> notInBounds;
    for (size_t i = 0; i < tileType.getShape().size(); i++) {
      zeroIndices.push_back(c0);
      notInBounds.push_back(false);
    }

    auto vecTy =
        VectorType::get(tileType.getShape(), tileType.getElementType());
    // this read should go away after canonicalizer, there is likely a tranfer
    // write before it
    auto valueVec = vector::TransferReadOp::create(rewriter, op.getLoc(), vecTy,
                                                   op.getValue(), zeroIndices,
                                                   std::nullopt, std::nullopt);

    auto vecWrite = vector::TransferWriteOp::create(
        rewriter, op.getLoc(), valueVec, op.getDestination(), op.getOffsets(),
        notInBounds);
    rewriter.replaceOp(op, vecWrite);

    return success();
  }
};

struct LowerCudaTileCPUMemOps
    : public mlir::cuda_tile::cpu::impl::LowerCudaTileCPUMemOpsBase<
          LowerCudaTileCPUMemOps> {
  using LowerCudaTileCPUMemOpsBase::LowerCudaTileCPUMemOpsBase;

  LowerCudaTileCPUMemOps() : LowerCudaTileCPUMemOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    patterns.add<GatherTilePattern, ScatterTilePattern, LoadPtrTilePattern,
                 StorePtrTilePattern, LoadMemRefTilePattern,
                 StoreMemRefTilePattern>(context);

    target.addIllegalOp<cpu::GatherTileOp, cpu::ScatterTileOp,
                        cpu::LoadPtrTileOp, cpu::StorePtrTileOp,
                        cpu::LoadMemRefTileOp, cpu::StoreMemRefTileOp>();

    target.addLegalDialect<
        ub::UBDialect, arith::ArithDialect, affine::AffineDialect,
        func::FuncDialect, memref::MemRefDialect,
        bufferization::BufferizationDialect, linalg::LinalgDialect,
        tensor::TensorDialect, math::MathDialect, scf::SCFDialect,
        vector::VectorDialect, cuda_tile::cpu::CudaTileCPUDialect>();

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet cleanupPatterns(context);
    vector::populateCastAwayVectorLeadingOneDimPatterns(cleanupPatterns);
    vector::populateDropUnitDimWithShapeCastPatterns(cleanupPatterns);

    if (failed(applyPatternsGreedily(mod, std::move(cleanupPatterns)))) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(context);
    vector::transferOpflowOpt(rewriter, mod);
  }
};

} // namespace

namespace mlir {
namespace cuda_tile {
namespace cpu {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerCudaTileCPUMemOps() {
  return std::make_unique<LowerCudaTileCPUMemOps>();
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir
