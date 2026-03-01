#include "cuda_tile_cpu/Conversion/CudaTileCPUToLLVM/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
#include <memory>

namespace mlir {
namespace cuda_tile {
namespace cpu {

#define GEN_PASS_DEF_CONVERTCUDATILECPUTOLLVM
#include "cuda_tile_cpu/Conversion/CudaTileCPUToLLVM/Passes.h.inc"

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir

namespace {

using namespace mlir;
using namespace cuda_tile;
using namespace llvm;

struct ConvertPrintOp : public OpConversionPattern<cpu::PrintOp> {
  using OpConversionPattern<cpu::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    auto *context = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(context);

    auto printFunc = getOrInsertPrintFuncDecl(rewriter, mod);
    // TODO: we can only have one print now because the symbol would get
    // overridden, should create unique symbols instead.
    std::string strTerminated = op.getStr().str() + '\0';
    mlir::Value fmtPtr = LLVM::createGlobalString(
        loc, rewriter, "print_fmt", strTerminated, LLVM::Linkage::Internal);

    if (op.getNumOperands() > 0) {
      return rewriter.notifyMatchFailure(op,
                                         "print with operands unimplemented");
    }

    LLVM::CallOp::create(rewriter, loc, printFunc, {fmtPtr});
    rewriter.eraseOp(op);

    return success();
  }

private:
  static LLVM::LLVMFuncOp
  getOrInsertPrintFuncDecl(ConversionPatternRewriter &rewriter, ModuleOp mod) {
    // check if func was already created
    if (auto funcOp =
            mod.lookupSymbol<LLVM::LLVMFuncOp>("cuda_tile_cpu_print")) {
      return funcOp;
    }

    // create func
    auto *context = mod.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(context);
    auto fnTy =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), ptrTy,
                                    /*isVarArg=*/false);

    OpBuilder::InsertionGuard insertGaurd(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funcOp =
        LLVM::LLVMFuncOp::create(rewriter, mod.getLoc(), "cuda_tile_cpu_print",
                                 fnTy, LLVM::Linkage::External);

    return funcOp;
  }
};

struct ConvertCudaTileCPUToLLVM
    : public mlir::cuda_tile::cpu::impl::ConvertCudaTileCPUToLLVMBase<
          ConvertCudaTileCPUToLLVM> {

  using ConvertCudaTileCPUToLLVMBase::ConvertCudaTileCPUToLLVMBase;

  ConvertCudaTileCPUToLLVM() : ConvertCudaTileCPUToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    ConversionTarget target(*context);
    LLVMTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);
    patterns.add<ConvertPrintOp>(typeConverter, context);

    target.addIllegalDialect<cpu::CudaTileCPUDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();

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
createConvertCudaTileCPUToLLVM() {
  return std::make_unique<ConvertCudaTileCPUToLLVM>();
}

} // namespace cpu
} // namespace cuda_tile
} // namespace mlir