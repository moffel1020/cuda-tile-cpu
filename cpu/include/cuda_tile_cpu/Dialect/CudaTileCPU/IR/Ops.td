#ifndef CUDATILE_CPU_OPS_TD
#define CUDATILE_CPU_OPS_TD

include "mlir/IR/OpBase.td"
include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.td"

class CudaTileCPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<CudaTileCPU_Dialect, mnemonic, !listconcat(traits, [])> {}

def CudaTileCPU_PrintOp : CudaTileCPU_Op<"print"> {
  let summary = "Print a formatted string";

  let description = [{
    Like cuda tile PrintOp, but it accepts memref as operand type.
  }];

  let arguments = (ins StrAttr:$prefix, 
    Variadic<AnyTypeOf<[AnyRankedOrUnrankedMemRef]>>:$args);
}

#endif
