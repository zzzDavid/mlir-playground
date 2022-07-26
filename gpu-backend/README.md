# Linalg to GPU

- LLVM: 14.0.0

This example demonstrates how to target NVIDIA GPU backend from Linalg dialect.
I successfully lowered the Linalg matmul example to LLVM/NVVM dialect and created both a JiT and an executable.

## Lowering
The lowering pipeline is: 
```shell
mlir-opt matmul.mlir \
    --convert-linalg-to-parallel-loops \
    --test-gpu-greedy-parallel-loop-mapping \
    --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining \
    --lower-affine \
    --convert-scf-to-std \
    --canonicalize \
    --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" --gpu-to-llvm
```

## Creating a JiT (mlir-cpu-runner)
```shell
./jit.sh
```
Expected output:
```text
Unranked Memref base@ = 0x393ff00 rank = 2 offset = 0 sizes = [8, 8] strides = [8, 1] data = 
[[8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8]]
```
## Compiling an executable
The script is `compile.sh`, but I found that the assember `as` compilains the input assembly has illegal symbol. I poked around and disovered that the path name is not how the assembler wants it. So we do the following steps.

### Generate the assembly
```shell
export LLVM_INSTALL_DIR=/work/shared/common/llvm-project-gpu
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/shared/common/usr/local/lib:/work/shared/common/usr/local/lib64:$LD_LIBRARY_PATH
export PATH=/work/shared/common/llvm-project-gpu/build/bin/:$PATH

mlir-opt matmul.mlir \
    --convert-linalg-to-parallel-loops \
    --test-gpu-greedy-parallel-loop-mapping \
    --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining \
    --lower-affine \
    --convert-scf-to-std \
    --canonicalize \
    --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" --gpu-to-llvm > matmul.mlir.llvm
mlir-translate matmul.mlir.llvm -mlir-to-llvmir > matmul.ll 
opt matmul.ll -O3 -S | llc -O3 -o matmul.s 
```

### Edit the `matmul.s`

There's a line in this file stating the path and file name:
```
	.file	1 "/home/some/path/mlir-playground/gpu-backend" "matmul.mlir.llvm"
```
Change it to:
```
	.file	1 "/home/nz264/shared/mlir-playground/gpu-backend"
```

### Assemble, link, and run
```shell
export LLVM_INSTALL_DIR=/work/shared/common/llvm-project-gpu
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/shared/common/usr/local/lib:/work/shared/common/usr/local/lib64:$LD_LIBRARY_PATH
export PATH=/work/shared/common/llvm-project-gpu/build/bin/:$PATH
as -o matmul.o matmul.s 
clang++ matmul.o -L$LLVM_INSTALL_DIR/build/lib -o exec -lcuda -lmlir_cuda_runtime -lmlir_runner_utils -lmlir_c_runner_utils
./exec
```

The expected output is:
```
[[8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8], 
 [8,   8,   8,   8,   8,   8,   8,   8]]
```

## Lessons

### Register host memref

One must register the host memref with GPU dialect, such as  
```mlir
%cast_A = memref.cast %A : memref<8x8xf32> to memref<*xf32>
gpu.host_register %cast_A : memref<*xf32>
```
Or else the compiled executable throws the following illegal address error:
```
'cuStreamSynchronize(stream)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuStreamDestroy(stream)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuModuleUnload(module)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuModuleLoadData(&module, data)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuModuleGetFunction(&function, module, name)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamSynchronize(stream)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamDestroy(stream)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuModuleUnload(module)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuModuleLoadData(&module, data)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuModuleGetFunction(&function, module, name)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamSynchronize(stream)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamDestroy(stream)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuModuleUnload(module)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuModuleLoadData(&module, data)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuModuleGetFunction(&function, module, name)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)' failed with 'CUDA_ERROR_ILLEGAL_ADDRESS'
'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamSynchronize(stream)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuStreamDestroy(stream)' failed with 'CUDA_ERROR_INVALID_HANDLE'
'cuModuleUnload(module)' failed with 'CUDA_ERROR_INVALID_HANDLE'
```