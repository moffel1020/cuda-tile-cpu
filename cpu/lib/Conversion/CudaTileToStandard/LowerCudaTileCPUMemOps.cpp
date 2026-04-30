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

struct LoadPtrTilePattern : public OpConversionPattern<cpu::LoadPtrTileOp> {
  using OpConversionPattern<cpu::LoadPtrTileOp>::OpConversionPattern;

  // usually when fusing and vectorizing we get the following pattern:
  // a = transfer_read of vector to tensor
  // b = load_ptr_tile a
  // transfer_write b from tensor to vector
  // to prevent bufferization, we add this rewrite, which operates on the vector
  // directly
  static LogicalResult
  rewriteWithInlineVectorTransfer(cpu::LoadPtrTileOp op,
                                  ConversionPatternRewriter &rewriter,
                                  RankedTensorType ty) {
    // TODO: could allow only a transfer read or only a transfer write, and use
    // bufferization for one side only

    if (!op->hasOneUse()) {
      return failure();
    }

    auto transferRead =
        dyn_cast<vector::TransferReadOp>(*op->getUsers().begin());
    if (!transferRead || transferRead.getBase() != op.getResult() ||
        transferRead.getMask() ||
        !hasOnlyZeroIndices(transferRead.getIndices())) {
      return failure();
    }

    auto vectorTy = transferRead.getVectorType();
    if (vectorTy.getRank() != 1 || vectorTy.isScalable() ||
        vectorTy.getDimSize(0) != ty.getDimSize(0)) {
      return failure();
    }

    auto transferWrite =
        op.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!transferWrite || transferWrite.getResult() != op.getSource() ||
        transferWrite.getMask() ||
        !hasOnlyZeroIndices(transferWrite.getIndices())) {
      return failure();
    }

    auto ptrVectorTy = transferWrite.getVectorType();
    if (ptrVectorTy.getRank() != 1 || ptrVectorTy.isScalable() ||
        ptrVectorTy.getDimSize(0) != ty.getDimSize(0)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto elemTy = ty.getElementType();
    auto lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto ub = arith::ConstantIndexOp::create(rewriter, loc, ty.getDimSize(0));
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto init = vector::BroadcastOp::create(rewriter, loc, vectorTy,
                                            transferRead.getPadding());

    auto loop = scf::ForOp::create(
        rewriter, loc, lb, ub, step, ValueRange{init.getResult()},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          auto ptr =
              vector::ExtractOp::create(b, loc, transferWrite.getValueToStore(),
                                        ArrayRef<OpFoldResult>{iv});
          auto val = cpu::LoadPtrOp::create(b, loc, elemTy, ptr);
          auto next = vector::InsertOp::create(b, loc, val, iterArgs[0],
                                               ArrayRef<OpFoldResult>{iv});
          scf::YieldOp::create(b, loc, next.getResult());
        });

    rewriter.replaceOp(transferRead, loop.getResult(0));
    rewriter.eraseOp(op);
    if (transferWrite->use_empty()) {
      rewriter.eraseOp(transferWrite);
    }
    return success();
  }

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
    }

    if (ty.getRank() != 1) { // TODO: support higher dims
      return failure();
    }

    // try to use vector transfer read and write values directly
    if (succeeded(rewriteWithInlineVectorTransfer(op, rewriter, ty))) {
      return success();
    }

    // fallback path, use bufferization from and to tensors using intermediate
    // memrefs
    auto sourceTy = op.getSource().getType();
    auto srcMemref = bufferization::ToBufferOp::create(
        rewriter, op.getLoc(),
        MemRefType::get(sourceTy.getShape(), sourceTy.getElementType()),
        op.getSource(), true);

    auto lb = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto ub =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), ty.getShape()[0]);
    auto step = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
    scf::ForOp::create(
        rewriter, op.getLoc(), lb, ub, step, ValueRange{},
        [&](OpBuilder &b, Location loc, Value i, ValueRange) {
          auto ptr = memref::LoadOp::create(b, loc, srcMemref, i);
          auto val = cpu::LoadPtrOp::create(b, op.getLoc(), elemTy, ptr);
          memref::StoreOp::create(b, loc, val, values, i);
          scf::YieldOp::create(b, loc);
        });

    auto tensor = bufferization::ToTensorOp::create(rewriter, op.getLoc(), ty,
                                                    values, true);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

struct StorePtrTilePattern : OpConversionPattern<cpu::StorePtrTileOp> {
  using OpConversionPattern<cpu::StorePtrTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::StorePtrTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ty = op.getValue().getType();
    if (ty.getRank() != 1) {
      return failure(); // TODO
    }

    auto destTy = op.getDestination().getType();
    auto destMemref = bufferization::ToBufferOp::create(
        rewriter, op.getLoc(),
        MemRefType::get(destTy.getShape(), destTy.getElementType()),
        op.getDestination(), true);

    auto valTy = op.getValue().getType();
    auto valMemref = bufferization::ToBufferOp::create(
        rewriter, op.getLoc(),
        MemRefType::get(valTy.getShape(), valTy.getElementType()),
        op.getValue(), true);

    auto lb = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto ub =
        arith::ConstantIndexOp::create(rewriter, op.getLoc(), ty.getShape()[0]);
    auto step = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
    auto loop = scf::ForOp::create(
        rewriter, op.getLoc(), lb, ub, step, ValueRange{},
        [&](OpBuilder &b, Location loc, Value i, ValueRange) {
          auto ptr = memref::LoadOp::create(b, loc, destMemref, i);
          auto val = memref::LoadOp::create(b, loc, valMemref, i);
          cpu::StorePtrOp::create(b, loc, ptr, val);
          scf::YieldOp::create(b, loc);
        });

    rewriter.replaceOp(op, loop);
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
    patterns.add<LoadPtrTilePattern, StorePtrTilePattern>(context);

    target.addIllegalOp<cpu::LoadPtrTileOp, cpu::StorePtrTileOp>();
    target.addLegalDialect<
        arith::ArithDialect, affine::AffineDialect, func::FuncDialect,
        memref::MemRefDialect, bufferization::BufferizationDialect,
        linalg::LinalgDialect, tensor::TensorDialect, math::MathDialect,
        scf::SCFDialect, vector::VectorDialect,
        cuda_tile::cpu::CudaTileCPUDialect>();

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
