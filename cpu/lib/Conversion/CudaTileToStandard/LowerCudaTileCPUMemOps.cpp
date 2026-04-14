#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile_cpu/Conversion/CudaTileToStandard/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.h"
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

struct LowerCudaTileCPUMemOps
    : public mlir::cuda_tile::cpu::impl::LowerCudaTileCPUMemOpsBase<
          LowerCudaTileCPUMemOps> {
  using LowerCudaTileCPUMemOpsBase::LowerCudaTileCPUMemOpsBase;

  LowerCudaTileCPUMemOps() : LowerCudaTileCPUMemOpsBase() {}

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    mlir::ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    target.addIllegalOp<cpu::LoadPtrOp>();
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
