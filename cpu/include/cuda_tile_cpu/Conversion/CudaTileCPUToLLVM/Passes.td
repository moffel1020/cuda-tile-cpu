#ifndef CUDATILECPUTOLLVM_CONVERSION_PASSES
#define CUDATILECPUTOLLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCudaTileCPUToLLVM : Pass<"convert-cuda-tile-cpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert cuda tile cpu ops to llvm ops";

    let constructor = "mlir::cuda_tile::cpu::createConvertCudaTileCPUToLLVM()";
    let dependentDialects = ["mlir::cuda_tile::cpu::CudaTileCPUDialect", 
                             "mlir::LLVM::LLVMDialect"];
}

#endif
