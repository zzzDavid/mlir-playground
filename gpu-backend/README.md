# GPU Backend

## Compile LLVM with NVIDIA GPU support
I'm using LLVM 14.0.0.
1. Change `MLIR_ENABLE_CUDA_RUNNER` to be `1` [here](https://github.com/llvm/llvm-project/blob/da38bcfd52d75e95f44e363288e3ed4a0cbf0e04/mlir/CMakeLists.txt#L108)
2. Run cmake
- using makefile
```sh
cmake -G "Unix Makefiles" ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=/home/nz264/anaconda3/envs/mlir/bin/python
```

- using ninja
```sh
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=/home/nz264/anaconda3/envs/mlir/bin/python
```
3. Run `make` or `ninja`

## What does each pass do?
Using `matmul.mlir` as an example:

### `--convert-linalg-to-parallel-loops`

As the name suggests, it converts linalg to scf.parallel loops.

```mlir
module {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      scf.for %arg5 = %c0 to %c8 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg5] : memref<8x8xf32>
        %1 = memref.load %arg1[%arg5, %arg4] : memref<8x8xf32>
        %2 = memref.load %arg2[%arg3, %arg4] : memref<8x8xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        memref.store %4, %arg2[%arg3, %arg4] : memref<8x8xf32>
      }
      scf.yield
    }
    return
  }
  func @main() {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<8x8xf32>
    %1 = memref.alloc() : memref<8x8xf32>
    %2 = memref.alloc() : memref<8x8xf32>
    %3 = memref.cast %0 : memref<8x8xf32> to memref<*xf32>
    gpu.host_register %3 : memref<*xf32>
    %4 = memref.cast %1 : memref<8x8xf32> to memref<*xf32>
    gpu.host_register %4 : memref<*xf32>
    %5 = memref.cast %2 : memref<8x8xf32> to memref<*xf32>
    gpu.host_register %5 : memref<*xf32>
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      memref.store %cst, %0[%arg0, %arg1] : memref<8x8xf32>
      scf.yield
    }
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      memref.store %cst, %1[%arg0, %arg1] : memref<8x8xf32>
      scf.yield
    }
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      memref.store %cst_0, %2[%arg0, %arg1] : memref<8x8xf32>
      scf.yield
    }
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  func private @print_memref_f32(memref<*xf32>)
}
```