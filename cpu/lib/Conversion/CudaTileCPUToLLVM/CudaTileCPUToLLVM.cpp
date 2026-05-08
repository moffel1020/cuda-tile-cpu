#include "cuda_tile_cpu/Conversion/CudaTileCPUToLLVM/Passes.h"
#include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"

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

struct PrintPattern : public OpConversionPattern<cpu::PrintOp> {
  using OpConversionPattern<cpu::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto mod = op->getParentOfType<ModuleOp>();
    auto *context = rewriter.getContext();

    std::string strTerminated = op.getStr().str() + '\0';
    mlir::Value printStr = LLVM::createGlobalString(
        loc, rewriter, getUniqueName(), strTerminated, LLVM::Linkage::Internal);

    if (op.getNumOperands() == 0) {
      auto printFunc = getOrCreateStrPrintFunc(rewriter, mod);
      LLVM::CallOp::create(rewriter, loc, printFunc, {printStr});
      rewriter.eraseOp(op);
      return success();
    }

    auto memrefElemType = op.getValue().getType().getElementType();
    auto printFunc = getOrCreateMemRefPrintFunc(rewriter, mod, memrefElemType);
    LLVM::CallOp::create(rewriter, loc, printFunc,
                         {printStr, adaptor.getValue()});

    rewriter.eraseOp(op);
    return success();
  }

private:
  static LLVM::LLVMFuncOp getOrCreateFunc(ConversionPatternRewriter &rewriter,
                                          ModuleOp mod, StringRef fnName,
                                          LLVM::LLVMFunctionType fnTy) {

    if (auto funcOp = mod.lookupSymbol<LLVM::LLVMFuncOp>(fnName)) {
      return funcOp;
    }

    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto funcOp = LLVM::LLVMFuncOp::create(rewriter, mod.getLoc(), fnName, fnTy,
                                           LLVM::Linkage::External);

    return funcOp;
  }

  static LLVM::LLVMFuncOp
  getOrCreateStrPrintFunc(ConversionPatternRewriter &rewriter, ModuleOp mod) {
    auto *context = mod.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(context);
    auto fnTy =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), ptrTy);

    return getOrCreateFunc(rewriter, mod, "ct_cpu_print_str", fnTy);
  }

  static LLVM::LLVMFuncOp
  getOrCreateMemRefPrintFunc(ConversionPatternRewriter &rewriter, ModuleOp mod,
                             mlir::Type type) {
    StringRef fnName =
        llvm::TypeSwitch<mlir::Type, StringRef>(type)
            .Case<Float32Type>([](auto) { return "ct_cpu_print_memref_f32"; })
            .Case<Float64Type>([](auto) { return "ct_cpu_print_memref_f64"; })
            .Case<mlir::IntegerType>([](auto intType) {
              const auto w = intType.getWidth();
              switch (w) {
              case 1:
              case 8:
                return "ct_cpu_print_memref_i8";
              case 32:
                return "ct_cpu_print_memref_i32";
              case 64:
                return "ct_cpu_print_memref_i64";
              default:
                return "unimplemented"; // TODO: proper error here
              }
            });

    // fn (const char* str, struct {int64_t rank, void *descriptor} )
    auto *ctx = mod.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Type = mlir::IntegerType::get(ctx, 64);
    auto unrankedMemStruct =
        LLVM::LLVMStructType::getLiteral(ctx, {i64Type, ptrTy});
    auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                            {ptrTy, unrankedMemStruct});
    return getOrCreateFunc(rewriter, mod, fnName, fnTy);
  }

  static std::string getUniqueName() {
    return "__print_str_" + std::to_string(strCounter++);
  }

  inline static int strCounter = 0; // thread safety? whats that
};

struct LoadPtrPattern : public OpConversionPattern<cpu::LoadPtrOp> {
  using OpConversionPattern<cpu::LoadPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cpu::LoadPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // auto attr = LLVM::DereferenceableAttr::get(getContext(), 8, false);
    auto ptr = LLVM::IntToPtrOp::create(
        rewriter, op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
        op.getSource());
    auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), op.getType(), ptr);
    rewriter.replaceOp(op, load);
    return success();
  }
};

struct StorePtrPattern : public OpConversionPattern<cpu::StorePtrOp> {
  using OpConversionPattern<cpu::StorePtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cpu::StorePtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = LLVM::IntToPtrOp::create(
        rewriter, op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
        op.getDestination());
    auto store =
        LLVM::StoreOp::create(rewriter, op.getLoc(), op.getValue(), ptr);
    rewriter.replaceOp(op, store);
    return success();
  }
};

struct MakeMemRefPattern : public OpConversionPattern<cpu::MakeMemRefOp> {
  using OpConversionPattern<cpu::MakeMemRefOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cpu::MakeMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memTy = dyn_cast<MemRefType>(op.getType());
    if (!memTy) {
      return failure();
    }

    auto structMemTy = getTypeConverter()->convertType(memTy);
    auto desc = MemRefDescriptor::poison(rewriter, op.getLoc(), structMemTy);

    auto ptr = LLVM::IntToPtrOp::create(
        rewriter, op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
        op.getPtr());

    desc.setAllocatedPtr(rewriter, op.getLoc(), ptr);
    desc.setAlignedPtr(rewriter, op.getLoc(), ptr);

    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(memTy.getStridesAndOffset(strides, offset))) {
      return failure();
    }
    ArrayRef<int64_t> shape = memTy.getShape();
    if (offset == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(op, "dynamic offset not supported");
    }

    desc.setConstantOffset(rewriter, op.getLoc(), offset);

    uint32_t dynIdx = 0;
    auto dynSizes = adaptor.getDynamicSizes();
    for (auto [i, dim] : llvm::enumerate(shape)) {
      if (dim == ShapedType::kDynamic) {
        desc.setSize(rewriter, op.getLoc(), i, dynSizes[dynIdx++]);
      } else {
        desc.setConstantSize(rewriter, op.getLoc(), i, dim);
      }
    }

    dynIdx = 0;
    auto dynStrides = adaptor.getDynamicStrides();
    for (auto [i, stride] : llvm::enumerate(strides)) {
      if (stride == ShapedType::kDynamic) {
        desc.setStride(rewriter, op.getLoc(), i, dynStrides[dynIdx++]);
      } else {
        desc.setConstantStride(rewriter, op.getLoc(), i, stride);
      }
    }

    rewriter.replaceOp(op, static_cast<mlir::Value>(desc));
    return success();
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
    patterns
        .add<PrintPattern, LoadPtrPattern, StorePtrPattern, MakeMemRefPattern>(
            typeConverter, context);

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
