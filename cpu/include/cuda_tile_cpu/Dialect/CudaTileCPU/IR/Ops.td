#ifndef CUDATILE_CPU_OPS_TD
#define CUDATILE_CPU_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Dialect.td"
include "cuda_tile_cpu/Dialect/CudaTileCPU/IR/Types.td"

include "cuda_tile/Dialect/CudaTile/IR/Types.td"


class CudaTileCPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<CudaTileCPU_Dialect, mnemonic, !listconcat(traits, [])> {}

def CudaTileCPU_PrintOp : CudaTileCPU_Op<"print", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Print a string with optional unranked memref at the end";

  let arguments = (ins StrAttr:$str, Optional<AnyUnrankedMemRef>:$value);

  let assemblyFormat = [{
    $str (`,` $value^)? attr-dict (`:` type($value)^)?
  }];
}

def CudaTileCPU_MakeMemRefOp : CudaTileCPU_Op<"make_memref", [
  Pure,
  AttrSizedOperandSegments
]> {
  let summary = "Make a memref from a pointer";

  let arguments = (ins I64:$ptr,
                       Variadic<Index>:$dynamicSizes,
                       Variadic<Index>:$dynamicStrides);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let assemblyFormat = [{
    $ptr
    `sizes` `[` $dynamicSizes `]`
    `strides` `[` $dynamicStrides `]`
    attr-dict `:` type($ptr) `->` type($result)
  }];
}

def CudaTileCPU_LoadMemRefTileOp : CudaTileCPU_Op<"load_memref_tile", [
  MemoryEffects<[MemRead]>
]> {
  let summary = "Load a tensor tile from a structured memref";

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source,
                       Variadic<Index>:$offsets);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$source `[` $offsets `]` attr-dict `:` type($source) `->` type($result)";
}

def CudaTileCPU_StoreMemRefTileOp : CudaTileCPU_Op<"store_memref_tile", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "Store a tensor tile to a structured memref";

  let arguments = (ins AnyRankedTensor:$value,
                       AnyRankedOrUnrankedMemRef:$destination,
                       Variadic<Index>:$offsets);

  let assemblyFormat = "$value `,` $destination `[` $offsets `]` attr-dict `:` type($value) `,` type($destination)";
}

def CudaTileCPU_GetNumBlocksOp : CudaTileCPU_Op<"get_num_blocks", [
  Pure,
  AllTypesMatch<["gridSize_x", "gridSize_y", "gridSize_z"]>
]> {
  let summary = "Like cuda_tile.get_num_tile_blocks but with tensors";

  let results = (outs TensorRankOf<[I32], [0]>:$gridSize_x,
                      TensorRankOf<[I32], [0]>:$gridSize_y,
                      TensorRankOf<[I32], [0]>:$gridSize_z);

  let assemblyFormat = "attr-dict `:` type($gridSize_x)";

  let builders = [
    OpBuilder<(ins), [{
      auto type = RankedTensorType::get({}, $_builder.getI32Type());
      build($_builder, $_state, type, type, type);
    }]>
  ];
}

def CudaTileCPU_GetBlockIdOp : CudaTileCPU_Op<"get_block_id", [
  Pure,
  AllTypesMatch<["blockId_x", "blockId_y", "blockId_z"]>
]> {
  let summary = "Like cuda_tile.get_tile_block_id but with tensors";

  let results = (outs TensorRankOf<[I32], [0]>:$blockId_x,
                      TensorRankOf<[I32], [0]>:$blockId_y,
                      TensorRankOf<[I32], [0]>:$blockId_z);

  let assemblyFormat = "attr-dict `:` type($blockId_x)";

  let builders = [
    OpBuilder<(ins), [{
      auto type = RankedTensorType::get({}, $_builder.getI32Type());
      build($_builder, $_state, type, type, type);
    }]>
  ];
}

def CudaTileCPU_LoadPtrTileOp : CudaTileCPU_Op<"load_ptr_tile", [MemoryEffects<[MemRead]>]> {
  let summary = "load a tensor from a tensor of pointers";

  let arguments = (ins RankedTensorOf<[I64]>:$source); // TODO: mask, padding?
  let results = (outs AnyRankedTensor:$result); // TODO: could verify type

  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def CudaTileCPU_LoadPtrOp : CudaTileCPU_Op<"load_ptr", [MemoryEffects<[MemRead]>]> {
  let summary = "load a scalar from a pointer";

  let arguments = (ins I64:$source);
  let results = (outs CudaTile_NumberType:$result);

  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` type($result)";
}

def CudaTileCPU_StorePtrTileOp : CudaTileCPU_Op<"store_ptr_tile", [MemoryEffects<[MemWrite]>]> {
  let summary = "store tensor to the addresses of a tensor of  pointers";

  let arguments = (ins RankedTensorOf<[I64]>:$destination, AnyRankedTensor:$value); // TODO mask

  let assemblyFormat = "$destination `,` $value attr-dict `:` type($destination) `,` type($value)";
}

def CudaTileCPU_StorePtrOp : CudaTileCPU_Op<"store_ptr", [MemoryEffects<[MemWrite]>]> {
  let summary = "store a scalar to an address specified by the pointer";

  let arguments = (ins I64:$destination, CudaTile_NumberType:$value);

  let assemblyFormat = "$destination `,` $value attr-dict `:` type($destination) `,` type($value)";
}

#endif
