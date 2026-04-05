#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
      return convertTileToVector(type);
    });
  }

private:
  Type convertTileToVector(cuda_tile::TileType type) const {
    auto shape = type.getShape();
    Type elementType = type.getElementType();
    return RankedTensorType::get(shape, elementType);
  }

  // Type convertCudaTilePtrToPtr(cuda_tile::PointerType ptrType) const {
  //   return
  //   ptr::PtrType::get(ptr::GenericSpaceAttr::get(ptrType.getContext()));
  // }
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

struct ConvertCudaTileIota : public OpConversionPattern<cuda_tile::IotaOp> {
  using OpConversionPattern<cuda_tile::IotaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto opType = op.getResult().getType();
    auto shape = opType.getShape();
    if (shape.size() != 1) {
      return rewriter.notifyMatchFailure(op, "1d shape expected for iota op");
    }

    auto width = opType.getElementType().getIntOrFloatBitWidth();
    SmallVector<APInt> indices(opType.getNumElements());
    std::iota(indices.begin(), indices.end(), APInt(width, 0));

    auto vecType = VectorType::get(shape, opType.getElementType());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, DenseElementsAttr::get(vecType, indices));

    return success();
  }
};

struct ConvertCudaTileBroadcast
    : public OpConversionPattern<cuda_tile::BroadcastOp> {

  using OpConversionPattern<cuda_tile::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto newVec = getTypeConverter()->convertType(op.getResult());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, newVec,
                                                     adaptor.getSource());
    return success();
  }
};

struct ConvertCudaTileReshape
    : public OpConversionPattern<cuda_tile::ReshapeOp> {
  using OpConversionPattern<cuda_tile::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto opType = op.getType();
    auto src = adaptor.getSource();

    // we convert tile<!cuda_tile.ptr<_>> to just a !ptr.ptr so to convert
    // !ptr.ptr to vector<1xptr> would be a broadcast instead of a reshape
    if (isa<ptr::PtrType>(src.getType())) {
      auto vecType = VectorType::get(opType.getShape(), src.getType());
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, vecType, src);
      return success();
    }

    if (!isa<VectorType>(src.getType())) {
      return failure();
    }

    auto vecType = VectorType::get(opType.getShape(), opType.getElementType());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecType,
                                                     adaptor.getSource());
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
struct ConvertCudaTileArith : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res = op.getResult().getType();
    auto opTy = op.getType();
    auto empty = tensor::EmptyOp::create(rewriter, op.getLoc(), opTy.getShape(),
                                         opTy.getElementType());
    rewriter.replaceOpWithNewOp<U>(op, adaptor.getOperands(),
                                   empty.getResult());
    return success();
  }
};

// TODO: ignoring overflow hints here, could we use them by using linalg map?
// in addition to these, ShLI also has int overflow flag
using ConvertCudaTileAddI =
    ConvertCudaTileArith<cuda_tile::AddIOp, linalg::AddOp>;
using ConvertCudaTileSubI =
    ConvertCudaTileArith<cuda_tile::SubIOp, linalg::SubOp>;
using ConvertCudaTileMulI =
    ConvertCudaTileArith<cuda_tile::MulIOp, linalg::MulOp>;

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
    patterns.add<ConvertEntryToFunc, ConvertCudaTileReturn,
                 ConvertCudaTileConstant, ConvertCudaTileIota,
                 ConvertCudaTileReshape, ConvertCudaTileBroadcast,
                 ConvertCudaTileAddI, ConvertCudaTileSubI, ConvertCudaTileMulI,
                 ConvertCudaTileOrI, ConvertCudaTileXOrI, ConvertCudaTileAndI,
                 ConvertCudaTilePrint, MoveOutOfCudaTileModule>(typeConverter,
                                                                context);

    target.addIllegalDialect<CudaTileDialect>();
    target.addLegalDialect<
        arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
        vector::VectorDialect, bufferization::BufferizationDialect,
        linalg::LinalgDialect, tensor::TensorDialect, ptr::PtrDialect,
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