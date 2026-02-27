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
          default:
            return aro::none;
          }
        },
        op.getOverflow());

    auto left = adaptor.getLhs();
    auto right = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, left, right, overflow);
    return success();
  }
};

struct ConvertCudaTilePrint : public OpConversionPattern<cuda_tile::PrintOp> {
  using OpConversionPattern<cuda_tile::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: emit function calls when going to llvm

    if (op.getNumOperands() == 0) {
      auto cpuPrint =
          cpu::PrintOp::create(rewriter, op.getLoc(), op.getStr(), {});
      rewriter.eraseOp(op);
      return success();
    }

    for (auto arg : op.getArgs()) {
      if (!isa<TileType>(arg.getType())) {
        llvm_unreachable("unimplemented print type");
        return failure();
      }

      // store the vector arg in a memref to call print
      auto loc = op.getLoc();
      auto tile = cast<TileType>(arg.getType());
      auto memType = MemRefType::get(tile.getShape(), tile.getElementType());
      auto allocOp = memref::AllocOp::create(rewriter, loc, memType);
      auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto storeOp =
          vector::StoreOp::create(rewriter, loc, rewriter.getRemappedValue(arg),
                                  allocOp.getResult(), c0.getResult());

      // TODO: convert to unranked memref first? depends on how i will implement the runtime
      cpu::PrintOp::create(rewriter, loc, op.getStr(), allocOp.getResult());
    }

    rewriter.eraseOp(op);
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
    patterns.add<ConvertEntryToFunc, ConvertCudaTileReturn,
                 ConvertCudaTileConstant, ConvertCudaTileAddi,
                 ConvertCudaTilePrint /*, MoveOutOfCudaTileModule*/>(
        typeConverter, context);

    target.addIllegalDialect<CudaTileDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect, vector::VectorDialect,
                           cuda_tile::cpu::CudaTileCPUDialect>();
    target.addLegalOp<cuda_tile::ModuleOp>();

    // populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
    //     patterns, typeConverter);

    // target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    //   return typeConverter.isSignatureLegal(op.getFunctionType()) &&
    //          typeConverter.isLegal(&op.getBody());
    // });

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