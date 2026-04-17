#ifndef CUDATILE_CPU_OPS_TD
#define CUDATILE_CPU_OPS_TD

include "mlir/IR/OpBase.td"

include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.td"
include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.td"

include "cuda_tile/Dialect/CudaTile/IR/Types.td"

class CudaTileCPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<CudaTileCPU_Dialect, mnemonic, !listconcat(traits, [])> {}

def CudaTileCPU_PrintOp : CudaTileCPU_Op<"print"> {
  let summary = "Print a string with optional unranked memref at the end";

  let arguments = (ins StrAttr:$str, Optional<AnyUnrankedMemRef>:$value);

  let assemblyFormat = [{
    $str (`,` $value^)? attr-dict (`:` type($value)^)?
  }];
}

def CudaTileCPU_MakeMemRefOp : CudaTileCPU_Op<"make_memref"> {
  let summary = "Make a memref from a pointer";

  let arguments = (ins I64:$ptr);
  let results = (outs AnyRankedOrUnrankedMemRef:$result); // TODO: dynamic shape/strides

  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";
}

def CudaTileCPU_LoadPtrTileOp : CudaTileCPU_Op<"load_ptr_tile"> {
  let summary = "load a tensor from a tensor of pointers";

  let arguments = (ins RankedTensorOf<[I64]>:$source); // TODO: mask, padding?
  let results = (outs AnyRankedTensor:$result); // TODO: could verify type

  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def CudaTileCPU_LoadPtrOp : CudaTileCPU_Op<"load_ptr"> {
  let summary = "load a scalar from a pointer";

  let arguments = (ins I64:$source);
  let results = (outs CudaTile_NumberType:$result);

  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` type($result)";
}

def CudaTileCPU_StorePtrTileOp : CudaTileCPU_Op<"store_ptr_tile"> {
  let summary = "store tensor to the addresses of a tensor of  pointers";

  let arguments = (ins RankedTensorOf<[I64]>:$destination, AnyRankedTensor:$value); // TODO mask

  let assemblyFormat = "$destination `,` $value attr-dict `:` type($destination) `,` type($value)";
}

def CudaTileCPU_StorePtrOp : CudaTileCPU_Op<"store_ptr"> {
  let summary = "store a scalar to an address specified by the pointer";

  let arguments = (ins I64:$destination, CudaTile_NumberType:$value);

  let assemblyFormat = "$destination `,` $value attr-dict `:` type($destination) `,` type($value)";
}

#endif
