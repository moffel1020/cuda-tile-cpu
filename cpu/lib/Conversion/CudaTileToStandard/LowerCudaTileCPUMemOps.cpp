#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
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

#define GEN_PASS_DEF_LOWERCUDATILECPUMEMOPS
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

namespace {

using namespace mlir;
using namespace mlir::cuda_tile;
using namespace llvm;

struct LoadPtrTilePattern : public OpConversionPattern<cpu::LoadPtrTileOp> {
  using OpConversionPattern<cpu::LoadPtrTileOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cpu::LoadPtrTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ty = op.getType();
    auto memTy = MemRefType::get(ty.getShape(), ty.getElementType());
    auto values = memref::AllocOp::create(rewriter, op.getLoc(), memTy);

    auto elemTy = cast<ShapedType>(op.getResult().getType()).getElementType();

    if (ty.getRank() == 0) {
      auto ptr =
          tensor::ExtractOp::create(rewriter, op.getLoc(), op.getSource(), {});
      auto val = cpu::LoadPtrOp::create(rewriter, op.getLoc(), elemTy, ptr);
      memref::StoreOp::create(rewriter, op.getLoc(), val, values);
      auto buffer = bufferization::ToTensorOp::create(rewriter, op.getLoc(), ty,
                                                      values, true);
      rewriter.replaceOp(op, buffer);
      return success();
    } else if (ty.getRank() != 1) { // TODO: fix
      return failure();
    }

    auto loop = affine::AffineForOp::create(
        rewriter, op.getLoc(), 0, ty.getShape()[0], 1, {},
        [&](OpBuilder &b, Location loc, Value i, ValueRange) {
          auto ptr =
              tensor::ExtractOp::create(b, op.getLoc(), op.getSource(), {i});
          auto val = cpu::LoadPtrOp::create(b, op.getLoc(), elemTy, ptr);
          affine::AffineStoreOp::create(b, loc, val, values, {i});
          affine::AffineYieldOp::create(b, loc);
        });

    auto buffer = bufferization::ToTensorOp::create(rewriter, op.getLoc(), ty,
                                                    values, true);
    rewriter.replaceOp(op, buffer);
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
    patterns.add<LoadPtrTilePattern>(context);

    target.addIllegalOp<cpu::LoadPtrTileOp>();
    target.addLegalOp<cpu::LoadPtrOp>();
    target.addLegalDialect<
        arith::ArithDialect, affine::AffineDialect, func::FuncDialect,
        memref::MemRefDialect, bufferization::BufferizationDialect,
        linalg::LinalgDialect, tensor::TensorDialect, math::MathDialect,
        scf::SCFDialect, cuda_tile::cpu::CudaTileCPUDialect>();

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

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerCudaTileCPUMemOps() {
  return std::make_unique<LowerCudaTileCPUMemOps>();
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir
