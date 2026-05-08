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

  static void lower0dLoad(cpu::LoadPtrTileOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) {
    auto ty = op.getType();
    auto memTy = MemRefType::get(ty.getShape(), ty.getElementType());
    auto elemTy = cast<ShapedType>(op.getResult().getType()).getElementType();
    auto ptr =
        tensor::ExtractOp::create(rewriter, op.getLoc(), op.getSource(), {});
    auto val = cpu::LoadPtrOp::create(rewriter, op.getLoc(), elemTy, ptr);
    auto tensor = tensor::FromElementsOp::create(rewriter, op.getLoc(), {val});
    rewriter.replaceOp(op, val);
  }

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

  // if the source tensor value is from a transfer write, return the vector
  // value to work on to avoid bufferization
  static Value getReplacementVectorFromTransferWrite(Operation *op) {
    auto transferWrite = dyn_cast<vector::TransferWriteOp>(op);
    if (transferWrite && !transferWrite.getMask() &&
        hasOnlyZeroIndices(transferWrite.getIndices()) &&
        transferWrite.getPermutationMap().isIdentity()) {
      return transferWrite.getValueToStore();
    }
    return nullptr;
  }

  static bool shouldForwardVectorThroughTransferRead(cpu::LoadPtrTileOp op) {
    auto transferRead =
        dyn_cast<vector::TransferReadOp>(*op->getUsers().begin());

    return op->hasOneUse() && transferRead != nullptr &&
           !transferRead.getMask() &&
           hasOnlyZeroIndices(transferRead.getIndices()) &&
           transferRead.getPermutationMap().isIdentity();
  }

  static void lowerNdLoad(cpu::LoadPtrTileOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) {
    auto toOpFold = [](ValueRange vr) {
      return llvm::map_to_vector(vr, [](Value v) { return OpFoldResult{v}; });
    };
    // TODO: this code sucks. maybe make it possible to have vector operands in
    // the load_ptr_tile directly
    auto ty = op.getType();
    auto memTy = MemRefType::get(ty.getShape(), ty.getElementType());
    auto elemTy = cast<ShapedType>(op.getResult().getType()).getElementType();

    auto ptrVec =
        getReplacementVectorFromTransferWrite(op.getSource().getDefiningOp());

    auto maybeTransferRead =
        dyn_cast<vector::TransferReadOp>(*op->getUsers().begin());
    const bool forwardVec = shouldForwardVectorThroughTransferRead(op);

    auto extractPtrElem = [&](OpBuilder &b, Location loc, ValueRange ivs,
                              Value src) -> Value {
      if (ptrVec == nullptr) {
        return tensor::ExtractOp::create(b, loc, src, ivs);
      } else {
        return vector::ExtractOp::create(b, loc, src, toOpFold(ivs));
      }
    };

    // will be cleaned up by canonicalizer if not used because of transfer read
    auto values = memref::AllocOp::create(rewriter, op.getLoc(), memTy);

    if (forwardVec) {
      // create 1d vector. mlir can only lower vector.insert if only the lower
      // dim is dynamic
      auto numElems = maybeTransferRead.getVectorType().getNumElements();
      Value initVec = ub::PoisonOp::create(rewriter, op.getLoc(),
                                           VectorType::get({numElems}, elemTy));
      auto lb = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
      auto ub = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                               ty.getNumElements());
      auto step = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);

      // get ptr tile input values and make 1d
      Value src;
      if (ptrVec != nullptr) {
        auto ptrVecTy =
            VectorType::get(ArrayRef<int64_t>{numElems}, rewriter.getI64Type());
        src = vector::ShapeCastOp::create(rewriter, op.getLoc(), ptrVecTy,
                                          ptrVec);
      } else {
        auto shapeType = RankedTensorType::get({1}, rewriter.getIndexType());
        auto attr = DenseElementsAttr::get(
            shapeType, APInt{64, static_cast<uint64_t>(numElems), false});
        auto constShape =
            arith::ConstantOp::create(rewriter, op.getLoc(), attr);
        auto reshapeTy =
            RankedTensorType::get({numElems}, rewriter.getI64Type());
        src = tensor::ReshapeOp::create(rewriter, op.getLoc(), reshapeTy,
                                        op.getSource(), constShape);
      }

      auto forOp = scf::ForOp::create(
          rewriter, op.getLoc(), lb, ub, step, initVec,
          [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
            Value ptr = extractPtrElem(b, loc, iv, src);
            auto val = cpu::LoadPtrOp::create(b, loc, elemTy, ptr);
            Value inserted = vector::InsertOp::create(b, loc, val, iterArgs[0],
                                                      toOpFold(iv));
            scf::YieldOp::create(b, loc, inserted);
          });

      // shape cast back to Nd vector
      auto result = vector::ShapeCastOp::create(
          rewriter, op.getLoc(), maybeTransferRead.getVectorType(),
          forOp->getResult(0));
      rewriter.replaceOp(maybeTransferRead, result);
    } else {
      // create a loop nest which stores in a memref

      auto outerFor = createLoopNest(
          rewriter, op.getLoc(), ty.getShape(), false, {},
          [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange iterArgs) {
            Value ptr = extractPtrElem(b, loc, ivs, op.getSource());
            auto val = cpu::LoadPtrOp::create(rewriter, loc, elemTy, ptr);
            memref::StoreOp::create(rewriter, loc, val, values, ivs);
            scf::YieldOp::create(b, loc);
          });

      auto tensor = bufferization::ToTensorOp::create(rewriter, op.getLoc(), ty,
                                                      values, true);
      rewriter.replaceOp(op, tensor);
    }
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

    lowerNdLoad(op, adaptor, rewriter);
    return success();
  }
};

struct StorePtrTilePattern : OpConversionPattern<cpu::StorePtrTileOp> {
  using OpConversionPattern<cpu::StorePtrTileOp>::OpConversionPattern;

  // When vectorization produces:
  //   %ptr_tensor = vector.transfer_write %ptr_vec, ...
  //   %val_tensor = vector.transfer_write %val_vec, ...
  //   cuda_tile_cpu.store_ptr_tile %ptr_tensor, %val_tensor
  // avoid materializing both tensors and store directly from the vectors.
  static LogicalResult
  rewriteWithoutVectorTransfer(cpu::StorePtrTileOp op,
                               ConversionPatternRewriter &rewriter,
                               RankedTensorType ty) {
    auto destTransferWrite =
        op.getDestination().getDefiningOp<vector::TransferWriteOp>();
    if (!destTransferWrite ||
        destTransferWrite.getResult() != op.getDestination() ||
        destTransferWrite.getMask() ||
        !hasOnlyZeroIndices(destTransferWrite.getIndices())) {
      return failure();
    }

    auto valueTransferWrite =
        op.getValue().getDefiningOp<vector::TransferWriteOp>();
    if (!valueTransferWrite ||
        valueTransferWrite.getResult() != op.getValue() ||
        valueTransferWrite.getMask() ||
        !hasOnlyZeroIndices(valueTransferWrite.getIndices())) {
      return failure();
    }

    auto ptrVectorTy = destTransferWrite.getVectorType();
    auto valueVectorTy = valueTransferWrite.getVectorType();
    if (ptrVectorTy.getRank() != 1 || ptrVectorTy.isScalable() ||
        ptrVectorTy.getDimSize(0) != ty.getDimSize(0) ||
        valueVectorTy.getRank() != 1 || valueVectorTy.isScalable() ||
        valueVectorTy.getDimSize(0) != ty.getDimSize(0)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto ub = arith::ConstantIndexOp::create(rewriter, loc, ty.getDimSize(0));
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    scf::ForOp::create(rewriter, loc, lb, ub, step, ValueRange{},
                       [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
                         auto ptr = vector::ExtractOp::create(
                             b, loc, destTransferWrite.getValueToStore(),
                             ArrayRef<OpFoldResult>{iv});
                         auto val = vector::ExtractOp::create(
                             b, loc, valueTransferWrite.getValueToStore(),
                             ArrayRef<OpFoldResult>{iv});
                         cpu::StorePtrOp::create(b, loc, ptr, val);
                         scf::YieldOp::create(b, loc);
                       });

    rewriter.eraseOp(op);
    if (destTransferWrite->use_empty()) {
      rewriter.eraseOp(destTransferWrite);
    }
    if (valueTransferWrite->use_empty()) {
      rewriter.eraseOp(valueTransferWrite);
    }
    return success();
  }

  LogicalResult
  matchAndRewrite(cpu::StorePtrTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ty = op.getValue().getType();
    if (ty.getRank() != 1) {
      return failure(); // TODO
    }

    if (succeeded(rewriteWithoutVectorTransfer(op, rewriter, ty))) {
      return success();
    }

    // fallback path, use tensor buffers
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

static FailureOr<memref::SubViewOp>
createMemRefTileSubview(Operation *op, Value source, ValueRange offsets,
                        RankedTensorType tileType,
                        ConversionPatternRewriter &rewriter) {
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType) {
    return failure();
  }
  if (sourceType.getRank() != tileType.getRank() ||
      offsets.size() != sourceType.getRank()) {
    return op->emitOpError("expected memref, tensor tile, and offset ranks to "
                           "match");
  }
  if (!tileType.hasStaticShape()) {
    return op->emitOpError("dynamic load/store memref tile shapes are not "
                           "supported yet");
  }

  SmallVector<OpFoldResult> mixedOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  sizes.reserve(tileType.getRank());
  strides.reserve(tileType.getRank());
  for (int64_t size : tileType.getShape()) {
    sizes.push_back(rewriter.getIndexAttr(size));
    strides.push_back(rewriter.getIndexAttr(1));
  }

  return memref::SubViewOp::create(rewriter, op->getLoc(), source, mixedOffsets,
                                   sizes, strides);
}

struct LoadMemRefTilePattern
    : public OpConversionPattern<cpu::LoadMemRefTileOp> {
  using OpConversionPattern<cpu::LoadMemRefTileOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cpu::LoadMemRefTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = cast<RankedTensorType>(op.getType());
    FailureOr<memref::SubViewOp> subview = createMemRefTileSubview(
        op, adaptor.getSource(), adaptor.getOffsets(), tileType, rewriter);
    if (failed(subview)) {
      return failure();
    }

    auto tensor = bufferization::ToTensorOp::create(
        rewriter, op.getLoc(), op.getType(), subview->getResult(),
        /*restrict=*/true, /*writable=*/false);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

struct StoreMemRefTilePattern
    : public OpConversionPattern<cpu::StoreMemRefTileOp> {
  using OpConversionPattern<cpu::StoreMemRefTileOp>::OpConversionPattern;

  static LogicalResult rewriteWithoutVectorTransfer(
      cpu::StoreMemRefTileOp op, memref::SubViewOp subview,
      ConversionPatternRewriter &rewriter, RankedTensorType tileType) {
    auto transferWrite = op.getValue().getDefiningOp<vector::TransferWriteOp>();
    if (!transferWrite || transferWrite.getResult() != op.getValue() ||
        transferWrite.getMask() ||
        !hasOnlyZeroIndices(transferWrite.getIndices())) {
      return failure();
    }

    auto write = vector::TransferWriteOp::create(
        rewriter, op.getLoc(), transferWrite.getValueToStore(),
        subview.getResult(), transferWrite.getIndices(),
        transferWrite.getPermutationMapAttr(), transferWrite.getMask(),
        transferWrite.getInBoundsAttr());

    rewriter.replaceOp(op, write);
    if (transferWrite->use_empty()) {
      rewriter.eraseOp(transferWrite);
    }
    return success();
  }

  LogicalResult
  matchAndRewrite(cpu::StoreMemRefTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tileType = cast<RankedTensorType>(op.getValue().getType());

    FailureOr<memref::SubViewOp> subview = createMemRefTileSubview(
        op, op.getDestination(), op.getOffsets(), tileType, rewriter);
    if (failed(subview)) {
      return failure();
    }

    if (succeeded(
            rewriteWithoutVectorTransfer(op, *subview, rewriter, tileType))) {
      return success();
    }

    // fallback
    auto materialize = bufferization::MaterializeInDestinationOp::create(
        rewriter, op.getLoc(), Type{}, adaptor.getValue(), subview->getResult(),
        /*restrict=*/false, /*writable=*/true);
    rewriter.replaceOp(op, materialize);
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
    patterns.add<LoadPtrTilePattern, StorePtrTilePattern, LoadMemRefTilePattern,
                 StoreMemRefTilePattern>(context);

    target.addIllegalOp<cpu::LoadPtrTileOp, cpu::StorePtrTileOp,
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
