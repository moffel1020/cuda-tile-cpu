#include "cuda_tile/Conversion/CudaTileToStd/Passes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace cuda_tile {

#define GEN_PASS_DEF_CUDATILECONVERTTOSTD
#include "cuda_tile/Conversion/CudaTileToStd/Passes.h.inc"

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

struct ConvertCudaTileAddi : public OpConversionPattern<cuda_tile::AddIOp> {
  using OpConversionPattern<cuda_tile::AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AddIOp op, OpAdaptor adaptor,
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
          }
        },
        op.getOverflow());

    auto left = adaptor.getLhs();
    auto right = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, left, right, overflow);
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

struct CudaTileConvertToStd
    : public mlir::cuda_tile::impl::CudaTileConvertToStdBase<
          CudaTileConvertToStd> {

  using CudaTileConvertToStdBase::CudaTileConvertToStdBase;

  CudaTileConvertToStd() : CudaTileConvertToStdBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    CudaTileTypeConverter typeConverter;

    RewritePatternSet patterns(context);
    patterns
        .add<ConvertEntryToFunc, ConvertCudaTileReturn, ConvertCudaTileConstant,
             ConvertCudaTileAddi, MoveOutOfCudaTileModule>(typeConverter,
                                                           context);

    target.addIllegalDialect<CudaTileDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();

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

std::unique_ptr<OperationPass<mlir::ModuleOp>> createCudaTileConvertToStd() {
  return std::make_unique<CudaTileConvertToStd>();
}

} // namespace cuda_tile
} // namespace mlir