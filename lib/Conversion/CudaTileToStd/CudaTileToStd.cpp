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
      return convertTileToVector(type);
    });
  }

private:
  Type convertTileToVector(cuda_tile::TileType type) const {
    auto shape = type.getShape();
    Type elementType = type.getElementType();

    return VectorType::get(shape, elementType);
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
    rewriter.eraseOp(op);

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

struct CudaTileConvertToStd
    : public mlir::cuda_tile::impl::CudaTileConvertToStdBase<
          CudaTileConvertToStd> {

  using CudaTileConvertToStdBase::CudaTileConvertToStdBase;

  CudaTileConvertToStd() : CudaTileConvertToStdBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    cuda_tile::ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    CudaTileTypeConverter typeConverter;

    target.addLegalDialect<BuiltinDialect, arith::ArithDialect, CudaTileDialect,
                           func::FuncDialect>();
    target.addIllegalOp<cuda_tile::ConstantOp, IotaOp, EntryOp,
                        cuda_tile::ReturnOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertEntryToFunc, ConvertCudaTileReturn,
                 ConvertCudaTileConstant>(typeConverter, context);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir {
namespace cuda_tile {

std::unique_ptr<OperationPass<cuda_tile::ModuleOp>>
createCudaTileConvertToStd() {
  return std::make_unique<CudaTileConvertToStd>();
}

} // namespace cuda_tile
} // namespace mlir