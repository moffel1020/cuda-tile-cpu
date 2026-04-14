#ifndef CUDATILE_CPU_TYPES_TD
#define CUDATILE_CPU_TYPES_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Types.td"

class CudaTileCPUTypeDef<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<CudaTileCPU_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def CudaTileCPU_PointerType : CudaTileCPUTypeDef<"Pointer", "ptr"> {
  let summary = "Pointer type that contains the element type it points to";
  let parameters = (ins CudaTile_NumberType:$pointeeType);

  let assemblyFormat = "`<` $pointeeType `>`";
}

#endif // CUDATILE_CPU_TYPES_TD
