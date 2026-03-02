#ifndef CUDATILE_CPU_OPS_TD
#define CUDATILE_CPU_OPS_TD

include "mlir/IR/OpBase.td"
include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.td"

class CudaTileCPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<CudaTileCPU_Dialect, mnemonic, !listconcat(traits, [])> {}

def CudaTileCPU_PrintOp : CudaTileCPU_Op<"print"> {
  let summary = "Print a string with optional unranked memref at the end";

  let arguments = (ins StrAttr:$str, Optional<AnyUnrankedMemRef>:$value);

  let assemblyFormat = [{
    $str (`,` $value^)? attr-dict (`:` type($value)^)?
  }];
}

#endif
