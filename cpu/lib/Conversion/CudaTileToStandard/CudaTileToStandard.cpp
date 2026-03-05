#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
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
      if (auto ptrType =
              dyn_cast<cuda_tile::PointerType>(type.getElementType())) {
        return convertPtrTileToMemref(ptrType);
      }

      return convertTileToVector(type);
    });
  }

private:
  Type convertTileToVector(cuda_tile::TileType type) const {
    auto shape = type.getShape();
    Type elementType = type.getElementType();

    return VectorType::get(shape, elementType);
  }

  Type convertPtrTileToMemref(cuda_tile::PointerType ptrType) const {
    auto pointee = ptrType.getPointeeType();
    return MemRefType::get({ShapedType::kDynamic}, pointee);
  }
};

struct ConvertEntryToFunc : public OpConversionPattern<cuda_tile::EntryOp> {
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

struct ConvertCudaTileReturn : public OpConversionPattern<cuda_tile::ReturnOp> {
  using OpConversionPattern<cuda_tile::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (op.getNumOperands() > 0) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "return with operands not supported";
      });
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

struct ConvertCudaTileConstant
    : public OpConversionPattern<cuda_tile::ConstantOp> {
  using OpConversionPattern<cuda_tile::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto oldType = op.getResult().getType();
    auto vecType =
        dyn_cast<VectorType>(getTypeConverter()->convertType(oldType));

    if (!vecType) {
      return failure();
    }

    auto oldAttr = dyn_cast<DenseElementsAttr>(op.getValueAttr());
    if (!oldAttr) {
      return failure();
    }

    auto newAttr = oldAttr.reshape(vecType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, vecType, newAttr);

    return success();
  }
};

template <typename T, typename U>
struct ConvertBinaryBitwiseOp : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto left = adaptor.getLhs();
    auto right = adaptor.getRhs();
    rewriter.replaceOpWithNewOp<U>(op, left, right);
    return success();
  }
};

using ConvertCudaTileOrI =
    ConvertBinaryBitwiseOp<cuda_tile::OrIOp, arith::OrIOp>;
using ConvertCudaTileXOrI =
    ConvertBinaryBitwiseOp<cuda_tile::XOrIOp, arith::XOrIOp>;
using ConvertCudaTileAndI =
    ConvertBinaryBitwiseOp<cuda_tile::AndIOp, arith::AndIOp>;

template <typename T, typename U>
struct ConvertBinaryIntOverflowOp : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto overflow = std::invoke(
        [](auto ctOverflow) {
          using aro = arith::IntegerOverflowFlags;
          using cto = cuda_tile::IntegerOverflow;

          switch (ctOverflow) {
          case cto::NONE:
            return aro::none;
          case cto::NSW:
            return aro::nsw;
          case cto::NUW:
            return aro::nuw;
          case cto::NW:
            return aro::nsw | aro::nuw;
          default:
            return aro::none;
          }
        },
        op.getOverflow());

    auto left = adaptor.getLhs();
    auto right = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<U>(op, left, right, overflow);
    return success();
  }
};

using ConvertCudaTileAddI =
    ConvertBinaryIntOverflowOp<cuda_tile::AddIOp, arith::AddIOp>;
using ConvertCudaTileSubI =
    ConvertBinaryIntOverflowOp<cuda_tile::SubIOp, arith::SubIOp>;
using ConvertCudaTileMulI =
    ConvertBinaryIntOverflowOp<cuda_tile::MulIOp, arith::MulIOp>;

struct ConvertCudaTilePrint : public OpConversionPattern<cuda_tile::PrintOp> {
  using OpConversionPattern<cuda_tile::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::PrintOp op, OpAdaptor adaptor,
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

      // store the vector arg in a memref to call print
      auto loc = op.getLoc();
      auto tile = cast<TileType>(arg.getType());
      auto memType = MemRefType::get(tile.getShape(), tile.getElementType());
      auto unrankedMem = UnrankedMemRefType::get(tile.getElementType(), {});

      auto allocOp = memref::AllocOp::create(rewriter, loc, memType);
      auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

      SmallVector<Value> indices(tile.getRank(), c0);
      auto storeOp =
          vector::StoreOp::create(rewriter, loc, rewriter.getRemappedValue(arg),
                                  allocOp.getResult(), indices);
      auto castOp = memref::CastOp::create(rewriter, op.getLoc(), unrankedMem,
                                           allocOp.getResult());

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
        .add<ConvertEntryToFunc, ConvertCudaTileReturn, ConvertCudaTileConstant,
             ConvertCudaTileAddI, ConvertCudaTileSubI, ConvertCudaTileMulI,
             ConvertCudaTileOrI, ConvertCudaTileXOrI, ConvertCudaTileAndI,
             ConvertCudaTilePrint, MoveOutOfCudaTileModule>(typeConverter,
                                                            context);

    target.addIllegalDialect<CudaTileDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect, vector::VectorDialect,
                           cuda_tile::cpu::CudaTileCPUDialect>();

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