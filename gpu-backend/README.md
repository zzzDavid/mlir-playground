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

<details>
  <summary> code </summary>

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
</details>

### `--test-gpu-greedy-parallel-loop-mapping`

We see that mapping attributes are added to `scf.parallel` loops:

<details>
  <summary> code </summary>

```mlir
#map = affine_map<(d0) -> (d0)>
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
    } {mapping = [{bound = #map, map = #map, processor = 0 : i64}, {bound = #map, map = #map, processor = 1 : i64}]}
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
    } {mapping = [{bound = #map, map = #map, processor = 0 : i64}, {bound = #map, map = #map, processor = 1 : i64}]}
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      memref.store %cst, %1[%arg0, %arg1] : memref<8x8xf32>
      scf.yield
    } {mapping = [{bound = #map, map = #map, processor = 0 : i64}, {bound = #map, map = #map, processor = 1 : i64}]}
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      memref.store %cst_0, %2[%arg0, %arg1] : memref<8x8xf32>
      scf.yield
    } {mapping = [{bound = #map, map = #map, processor = 0 : i64}, {bound = #map, map = #map, processor = 1 : i64}]}
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  func private @print_memref_f32(memref<*xf32>)
}
```
</details>

### `--convert-parallel-loops-to-gpu`

`scf.parallel` loops with mapping info are converted to gpu launch blocks:

<details>
  <summary> code </summary>

```mlir
#map0 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map0(%c8)[%c0, %c1]
    %1 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %1, %arg11 = %c1_0) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1_0, %arg13 = %c1_0, %arg14 = %c1_0) {
      %2 = affine.apply #map1(%arg3)[%c1, %c0]
      %3 = affine.apply #map1(%arg4)[%c1, %c0]
      scf.for %arg15 = %c0 to %c8 step %c1 {
        %4 = memref.load %arg0[%2, %arg15] : memref<8x8xf32>
        %5 = memref.load %arg1[%arg15, %3] : memref<8x8xf32>
        %6 = memref.load %arg2[%2, %3] : memref<8x8xf32>
        %7 = arith.mulf %4, %5 : f32
        %8 = arith.addf %6, %7 : f32
        memref.store %8, %arg2[%2, %3] : memref<8x8xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
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
    %c1_1 = arith.constant 1 : index
    %6 = affine.apply #map0(%c8)[%c0, %c1]
    %7 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %6, %arg7 = %7, %arg8 = %c1_1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1_1, %arg10 = %c1_1, %arg11 = %c1_1) {
      %12 = affine.apply #map1(%arg0)[%c1, %c0]
      %13 = affine.apply #map1(%arg1)[%c1, %c0]
      memref.store %cst, %0[%12, %13] : memref<8x8xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    %c1_2 = arith.constant 1 : index
    %8 = affine.apply #map0(%c8)[%c0, %c1]
    %9 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %8, %arg7 = %9, %arg8 = %c1_2) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1_2, %arg10 = %c1_2, %arg11 = %c1_2) {
      %12 = affine.apply #map1(%arg0)[%c1, %c0]
      %13 = affine.apply #map1(%arg1)[%c1, %c0]
      memref.store %cst, %1[%12, %13] : memref<8x8xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    %c1_3 = arith.constant 1 : index
    %10 = affine.apply #map0(%c8)[%c0, %c1]
    %11 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %10, %arg7 = %11, %arg8 = %c1_3) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1_3, %arg10 = %c1_3, %arg11 = %c1_3) {
      %12 = affine.apply #map1(%arg0)[%c1, %c0]
      %13 = affine.apply #map1(%arg1)[%c1, %c0]
      memref.store %cst_0, %2[%12, %13] : memref<8x8xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  func private @print_memref_f32(memref<*xf32>)
}
```
</details>

### `--gpu-kernel-outlining`

The gpu launch blocks are outlined to `gpu.module` and `gpu.func`:

<details>
  <summary> code </summary>

```mlir
#map0 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map0(%c8)[%c0, %c1]
    %1 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch_func  @matmul_linalg_kernel::@matmul_linalg_kernel blocks in (%0, %1, %c1_0) threads in (%c1_0, %c1_0, %c1_0) args(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>, %arg2 : memref<8x8xf32>)
    return
  }
  gpu.module @matmul_linalg_kernel {
    gpu.func @matmul_linalg_kernel(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %12 = affine.apply #map1(%0)[%c1, %c0]
      %13 = affine.apply #map1(%1)[%c1, %c0]
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %14 = memref.load %arg0[%12, %arg3] : memref<8x8xf32>
        %15 = memref.load %arg1[%arg3, %13] : memref<8x8xf32>
        %16 = memref.load %arg2[%12, %13] : memref<8x8xf32>
        %17 = arith.mulf %14, %15 : f32
        %18 = arith.addf %16, %17 : f32
        memref.store %18, %arg2[%12, %13] : memref<8x8xf32>
      }
      gpu.return
    }
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
    %c1_1 = arith.constant 1 : index
    %6 = affine.apply #map0(%c8)[%c0, %c1]
    %7 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%6, %7, %c1_1) threads in (%c1_1, %c1_1, %c1_1) args(%0 : memref<8x8xf32>)
    %c1_2 = arith.constant 1 : index
    %8 = affine.apply #map0(%c8)[%c0, %c1]
    %9 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%8, %9, %c1_2) threads in (%c1_2, %c1_2, %c1_2) args(%1 : memref<8x8xf32>)
    %c1_3 = arith.constant 1 : index
    %10 = affine.apply #map0(%c8)[%c0, %c1]
    %11 = affine.apply #map0(%c8)[%c0, %c1]
    gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%10, %11, %c1_3) threads in (%c1_3, %c1_3, %c1_3) args(%2 : memref<8x8xf32>)
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.000000e+00 : f32
      %12 = affine.apply #map1(%0)[%c1, %c0]
      %13 = affine.apply #map1(%1)[%c1, %c0]
      memref.store %cst, %arg0[%12, %13] : memref<8x8xf32>
      gpu.return
    }
  }
  gpu.module @main_kernel_0 {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.000000e+00 : f32
      %12 = affine.apply #map1(%0)[%c1, %c0]
      %13 = affine.apply #map1(%1)[%c1, %c0]
      memref.store %cst, %arg0[%12, %13] : memref<8x8xf32>
      gpu.return
    }
  }
  gpu.module @main_kernel_1 {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = affine.apply #map1(%0)[%c1, %c0]
      %13 = affine.apply #map1(%1)[%c1, %c0]
      memref.store %cst, %arg0[%12, %13] : memref<8x8xf32>
      gpu.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
}
```
</details>

Note that all the memref load/store are `memref.load`, `memref.store` instead of `affine.load` and `affine.store`.

### `--lower-affine`

Lowers all the affine operations. Here, it lowers the affine apply operations into arithmetic operations.

<details>
  <summary> code </summary>

```mlir
module attributes {gpu.container_module} {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c8_1 = arith.constant 8 : index
    %c8_2 = arith.constant 8 : index
    gpu.launch_func  @matmul_linalg_kernel::@matmul_linalg_kernel blocks in (%c8_1, %c8_2, %c1_0) threads in (%c1_0, %c1_0, %c1_0) args(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>, %arg2 : memref<8x8xf32>)
    return
  }
  gpu.module @matmul_linalg_kernel {
    gpu.func @matmul_linalg_kernel(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %12 = arith.muli %0, %c1 : index
      %13 = arith.addi %12, %c0 : index
      %14 = arith.muli %1, %c1 : index
      %15 = arith.addi %14, %c0 : index
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %16 = memref.load %arg0[%13, %arg3] : memref<8x8xf32>
        %17 = memref.load %arg1[%arg3, %15] : memref<8x8xf32>
        %18 = memref.load %arg2[%13, %15] : memref<8x8xf32>
        %19 = arith.mulf %16, %17 : f32
        %20 = arith.addf %18, %19 : f32
        memref.store %20, %arg2[%13, %15] : memref<8x8xf32>
      }
      gpu.return
    }
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
    %c1_1 = arith.constant 1 : index
    %c8_2 = arith.constant 8 : index
    %c8_3 = arith.constant 8 : index
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8_2, %c8_3, %c1_1) threads in (%c1_1, %c1_1, %c1_1) args(%0 : memref<8x8xf32>)
    %c1_4 = arith.constant 1 : index
    %c8_5 = arith.constant 8 : index
    %c8_6 = arith.constant 8 : index
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%c8_5, %c8_6, %c1_4) threads in (%c1_4, %c1_4, %c1_4) args(%1 : memref<8x8xf32>)
    %c1_7 = arith.constant 1 : index
    %c8_8 = arith.constant 8 : index
    %c8_9 = arith.constant 8 : index
    gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%c8_8, %c8_9, %c1_7) threads in (%c1_7, %c1_7, %c1_7) args(%2 : memref<8x8xf32>)
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.000000e+00 : f32
      %12 = arith.muli %0, %c1 : index
      %13 = arith.addi %12, %c0 : index
      %14 = arith.muli %1, %c1 : index
      %15 = arith.addi %14, %c0 : index
      memref.store %cst, %arg0[%13, %15] : memref<8x8xf32>
      gpu.return
    }
  }
  gpu.module @main_kernel_0 {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.000000e+00 : f32
      %12 = arith.muli %0, %c1 : index
      %13 = arith.addi %12, %c0 : index
      %14 = arith.muli %1, %c1 : index
      %15 = arith.addi %14, %c0 : index
      memref.store %cst, %arg0[%13, %15] : memref<8x8xf32>
      gpu.return
    }
  }
  gpu.module @main_kernel_1 {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %12 = arith.muli %0, %c1 : index
      %13 = arith.addi %12, %c0 : index
      %14 = arith.muli %1, %c1 : index
      %15 = arith.addi %14, %c0 : index
      memref.store %cst, %arg0[%13, %15] : memref<8x8xf32>
      gpu.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
}
```
</details>

###  `--convert-scf-to-std` and `--canonicalize`

This pass lowers scf loops to std control blocks. But here it doesn't do anything because we are already free of scf operations.

After canonicalization: 

<details>
  <summary> code </summary>

```mlir
module attributes {gpu.container_module} {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @matmul_linalg_kernel::@matmul_linalg_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>, %arg2 : memref<8x8xf32>)
    return
  }
  gpu.module @matmul_linalg_kernel {
    gpu.func @matmul_linalg_kernel(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) kernel {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      br ^bb1(%c0 : index)
    ^bb1(%2: index):  // 2 preds: ^bb0, ^bb2
      %3 = arith.cmpi slt, %2, %c8 : index
      cond_br %3, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %4 = memref.load %arg0[%0, %2] : memref<8x8xf32>
      %5 = memref.load %arg1[%2, %1] : memref<8x8xf32>
      %6 = memref.load %arg2[%0, %1] : memref<8x8xf32>
      %7 = arith.mulf %4, %5 : f32
      %8 = arith.addf %6, %7 : f32
      memref.store %8, %arg2[%0, %1] : memref<8x8xf32>
      %9 = arith.addi %2, %c1 : index
      br ^bb1(%9 : index)
    ^bb3:  // pred: ^bb1
      gpu.return
    }
  }
  func @main() {
    %c8 = arith.constant 8 : index
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
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%1 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%2 : memref<8x8xf32>)
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %cst = arith.constant 1.000000e+00 : f32
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      memref.store %cst, %arg0[%0, %1] : memref<8x8xf32>
      gpu.return
    }
  }
  gpu.module @main_kernel_0 {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %cst = arith.constant 1.000000e+00 : f32
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      memref.store %cst, %arg0[%0, %1] : memref<8x8xf32>
      gpu.return
    }
  }
  gpu.module @main_kernel_1 {
    gpu.func @main_kernel(%arg0: memref<8x8xf32>) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      memref.store %cst, %arg0[%0, %1] : memref<8x8xf32>
      gpu.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
}
```

</details>


### `strip-debuginfo`

Remove debug info. Cannot be removed, or `gpu-to-cubin` fails.

### `convert-gpu-to-nvvm`

This pass does most of the heavy-lifting. 

<details>
  <summary> code </summary>

```mlir
module attributes {gpu.container_module} {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @matmul_linalg_kernel::@matmul_linalg_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>, %arg2 : memref<8x8xf32>)
    return
  }
  gpu.module @matmul_linalg_kernel {
    llvm.func @matmul_linalg_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %24 = llvm.mlir.constant(8 : index) : i64
      %25 = llvm.mlir.constant(0 : index) : i64
      %26 = llvm.mlir.constant(1 : index) : i64
      %27 = nvvm.read.ptx.sreg.ctaid.x : i32
      %28 = llvm.sext %27 : i32 to i64
      %29 = nvvm.read.ptx.sreg.ctaid.y : i32
      %30 = llvm.sext %29 : i32 to i64
      llvm.br ^bb1(%25 : i64)
    ^bb1(%31: i64):  // 2 preds: ^bb0, ^bb2
      %32 = llvm.icmp "slt" %31, %24 : i64
      llvm.cond_br %32, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %33 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %34 = llvm.mlir.constant(8 : index) : i64
      %35 = llvm.mul %28, %34  : i64
      %36 = llvm.add %35, %31  : i64
      %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %38 = llvm.load %37 : !llvm.ptr<f32>
      %39 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %40 = llvm.mlir.constant(8 : index) : i64
      %41 = llvm.mul %31, %40  : i64
      %42 = llvm.add %41, %30  : i64
      %43 = llvm.getelementptr %39[%42] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %44 = llvm.load %43 : !llvm.ptr<f32>
      %45 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %46 = llvm.mlir.constant(8 : index) : i64
      %47 = llvm.mul %28, %46  : i64
      %48 = llvm.add %47, %30  : i64
      %49 = llvm.getelementptr %45[%48] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %50 = llvm.load %49 : !llvm.ptr<f32>
      %51 = llvm.fmul %38, %44  : f32
      %52 = llvm.fadd %50, %51  : f32
      %53 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %54 = llvm.mlir.constant(8 : index) : i64
      %55 = llvm.mul %28, %54  : i64
      %56 = llvm.add %55, %30  : i64
      %57 = llvm.getelementptr %53[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %52, %57 : !llvm.ptr<f32>
      %58 = llvm.add %31, %26  : i64
      llvm.br ^bb1(%58 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
  func @main() {
    %c8 = arith.constant 8 : index
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
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%1 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%2 : memref<8x8xf32>)
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  gpu.module @main_kernel_0 {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  gpu.module @main_kernel_1 {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
}

```

</details>

### `gpu-to-cubin`

Some magic happens here, the GPU kernels are compiled to bitcode and become binary format:

<details>
  <summary> code </summary>

```mlir
module attributes {gpu.container_module} {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @matmul_linalg_kernel::@matmul_linalg_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>, %arg2 : memref<8x8xf32>)
    return
  }
  gpu.module @matmul_linalg_kernel attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00\A0\0C\00\00\00\00\00\00`\0A\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.matmul_linalg_kernel\00.nv.info.matmul_linalg_kernel\00.nv.shared.matmul_linalg_kernel\00.nv.constant0.matmul_linalg_kernel\00.rel.nv.constant0.matmul_linalg_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00matmul_linalg_kernel\00.text.matmul_linalg_kernel\00.nv.info.matmul_linalg_kernel\00.nv.shared.matmul_linalg_kernel\00.rel.nv.constant0.matmul_linalg_kernel\00.nv.constant0.matmul_linalg_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00G\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C7\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F1\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00@\03\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\0D\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\08\00\B0\00\00\00\18\03\00\00\04\1D\08\00\18\00\00\000\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\04\17\0C\00\00\00\00\00\0D\00h\00\00\F0!\00\04\17\0C\00\00\00\00\00\0E\00p\00\00\F0!\00\04\17\0C\00\00\00\00\00\0F\00x\00\00\F0!\00\04\17\0C\00\00\00\00\00\10\00\80\00\00\F0!\00\04\17\0C\00\00\00\00\00\11\00\88\00\00\F0!\00\04\17\0C\00\00\00\00\00\12\00\90\00\00\F0!\00\04\17\0C\00\00\00\00\00\13\00\98\00\00\F0!\00\04\17\0C\00\00\00\00\00\14\00\A0\00\00\F0!\00\03\19\A8\00\04\0A\08\00\02\00\00\00@\01\A8\00\01*\00\00\010\00\00\047\04\00r\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07\00\FE\00X\1C\00\01\00\87\00\80\07\98L\FF\FF\97\FF\FF\FF\1F\1C\00\00W\02\00\00\C8\F0\F0\07 \E6\00\C4?\08\07\FF\F7\0F\80\0Bg[\02\00g\02\00\00\C8\F0\03\00\F7\01\00\00)8\F1\17 \FE@\94\1F\00\05\02\F7\01\00\00)8\04\00'\00\80\81\D7[\07\00\B7\01\00\00)8\F1\07\C2\FC\00\C4\1F\00\03\00W0\C0\01\DF[\05\00'\05\80\82\18L\07\077\05\00\08\10L\F6\07\E0\FD\00\A0\1F\00\00\04\E7\06\00\81\D7K\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\F0\07\C0\FE\00\C4\1F\08\03\04\F7\06\80\01\17\1A\0F\00\08\00\00\00\00\E3\06\05\F7\0F\00\80\D7[\F5\07\22\FE\00\98\1F\00\04\02\E7\01\00\00)8\07\05\F7\0F\C0\03\D8[\02\02\07\06\00\81\18L\F1\07\C0\FE@\C4\1F\00\05\04\17\06\00\08\10L\04\02\F7\0F\00\80\D7[\05\02\F7\0F\C0\02\D9[\F6\07\02\FE\00\D8\1E\00\02\00\F7\0F\00\80\D7[\03\00\F7\0F\C0\01\DA[\00\06\07\00\00\00\90\80\B7\07@\F6\00\98\1F\04\09\04\07\00\00\00\90\84\08\02\07\00\00\00\90\88\09\00\97\00\00\00h\\\E4\07 \1E\00\C4\1E\00\08\08\97\00\00\00X\\\08\02\07\00\00\00\90\A8\00\06G\00\00\00\90\80\B1\07@\F6\00\98\1F\04\09\04\07\02\00\00\90\84\0A\02\07\00\00\00\90\88\00\00\97\00\00\00h\\\E4\07 >\00\C4\1E\00\00\0A\07\00\00\00X\\\00\02\07\00\00\00\90\A8\09\06\87\00\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\04\00\00\90\84\0B\02\07\00\00\00\90\88\09\09\A7\00\00\00h\\\E4\07 ^\00\C4>\00\09\0B\97\00\00\00X\\\09\02\07\00\00\00\90\A8\08\06\C7\00\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\06\00\00\90\84\0B\02\07\00\00\00\90\88\08\08\A7\00\00\00h\\\E4\07 \1E\00\C4^\00\08\0B\87\00\00\00X\\\08\02\07\00\00\00\90\A8\00\06\07\01\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\08\00\00\90\84\0B\02\07\00\00\00\90\88\00\00\A7\00\00\00h\\\E4\07 >\00\C4\9E\00\00\0B\07\00\00\00X\\\00\02\07\00\00\00\90\A8\09\06G\01\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\0A\00\00\90\84\0B\02\07\00\00\00\90\88\09\09\A7\00\00\00h\\\E4\07 ^\00\C4>\00\0B\0B\97\00\00\00X\\\0B\02\07\00\00\00\90\A8\08\06\87\01\00\00\90\80\B1\07@\F6\00\98\1F\04\09\04\07\0C\00\00\90\84\0A\02\07\00\00\00\90\88\08\08\97\00\00\00h\\\E4\07 ^\00\C4^\00\0C\0A\87\00\00\00X\\\0C\02\07\00\00\00\90\A8\00\06\C7\01\00\00\90\80\B1\07@\F6\00\98\1F\04\09\04\07\0E\00\00\90\84\08\02\07\00\00\00\90\88\09\00\97\00\00\00h\\\E6\07 \FE\04\FC\1F\00\08\08\97\00\00\00X\\\08\02\07\00\00\00\90\A8\0F\00\07\00\00\00\00\E3\FF\07\00\FC\00\80\1F\00\0F\00\07\FF\FF\0F@\E2\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\E4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$\01\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\02\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A0\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00M\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C4\02\00\00\00\00\00\00\8C\01\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D5\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\04\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\8B\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\05\00\00\00\00\00\00\E8\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00 \07\00\00\00\00\00\00@\03\00\00\00\00\00\00\03\00\00\00\04\00\00\0D \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\A0\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00(\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\05\00\00\00\00\00\008\05\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @matmul_linalg_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %24 = llvm.mlir.constant(8 : index) : i64
      %25 = llvm.mlir.constant(0 : index) : i64
      %26 = llvm.mlir.constant(1 : index) : i64
      %27 = nvvm.read.ptx.sreg.ctaid.x : i32
      %28 = llvm.sext %27 : i32 to i64
      %29 = nvvm.read.ptx.sreg.ctaid.y : i32
      %30 = llvm.sext %29 : i32 to i64
      llvm.br ^bb1(%25 : i64)
    ^bb1(%31: i64):  // 2 preds: ^bb0, ^bb2
      %32 = llvm.icmp "slt" %31, %24 : i64
      llvm.cond_br %32, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %33 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %34 = llvm.mlir.constant(8 : index) : i64
      %35 = llvm.mul %28, %34  : i64
      %36 = llvm.add %35, %31  : i64
      %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %38 = llvm.load %37 : !llvm.ptr<f32>
      %39 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %40 = llvm.mlir.constant(8 : index) : i64
      %41 = llvm.mul %31, %40  : i64
      %42 = llvm.add %41, %30  : i64
      %43 = llvm.getelementptr %39[%42] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %44 = llvm.load %43 : !llvm.ptr<f32>
      %45 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %46 = llvm.mlir.constant(8 : index) : i64
      %47 = llvm.mul %28, %46  : i64
      %48 = llvm.add %47, %30  : i64
      %49 = llvm.getelementptr %45[%48] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %50 = llvm.load %49 : !llvm.ptr<f32>
      %51 = llvm.fmul %38, %44  : f32
      %52 = llvm.fadd %50, %51  : f32
      %53 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %54 = llvm.mlir.constant(8 : index) : i64
      %55 = llvm.mul %28, %54  : i64
      %56 = llvm.add %55, %30  : i64
      %57 = llvm.getelementptr %53[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %52, %57 : !llvm.ptr<f32>
      %58 = llvm.add %31, %26  : i64
      llvm.br ^bb1(%58 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
  func @main() {
    %c8 = arith.constant 8 : index
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
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%1 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%2 : memref<8x8xf32>)
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00`\08\00\00\00\00\00\00 \06\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.constant0.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00main_kernel\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.constant0.main_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00>\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BB\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\05\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\04\00\90\00\00\00\04\1D\08\00\10\00\00\00\18\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\03\198\00\04\0A\08\00\02\00\00\00@\018\00\01*\00\00\010\00\00\047\04\00r\00\00\00\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07 \E2\00\FC\1C\00\01\00\87\00\80\07\98L\00\00W\02\00\00\C8\F0\02\00g\02\00\00\C8\F0\F1\0F\22\FE\02\98\1F\00\04\00\D7\01\00\00)8\03\00'\00\80\81\18\\\02\02\F7\01\00\00)8\F1\07\C0\FE@\C4\1F\00\02\02G\00\00\08\10\\\00\03'\05\00\81\D7K\03\037\05\00\01\17\1A\F6\07\22\FC\00\B0\1F\00\02\00\F7\0F\00\80\D7[\03\00\F7\0F\C0\01\D8[\00\F0\07\00\00\F8\03\01\F1\07\E0\FF\00\FC\1F\00\00\02\07\00\00\00\90\A0\0F\00\07\00\00\00\00\E3\0F\00\87\FF\FF\0F@\E2\E0\07\00\FC\00\80\1F\00\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\B7\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F7\00\00\00\00\00\00\00\CA\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\01\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00d\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00p\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\00x\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\05\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\03\00\00\00\04\00\00\05 \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00`\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\02\00\00\00\00\00\008\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  gpu.module @main_kernel_0 attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00`\08\00\00\00\00\00\00 \06\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.constant0.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00main_kernel\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.constant0.main_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00>\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BB\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\05\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\04\00\90\00\00\00\04\1D\08\00\10\00\00\00\18\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\03\198\00\04\0A\08\00\02\00\00\00@\018\00\01*\00\00\010\00\00\047\04\00r\00\00\00\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07 \E2\00\FC\1C\00\01\00\87\00\80\07\98L\00\00W\02\00\00\C8\F0\02\00g\02\00\00\C8\F0\F1\0F\22\FE\02\98\1F\00\04\00\D7\01\00\00)8\03\00'\00\80\81\18\\\02\02\F7\01\00\00)8\F1\07\C0\FE@\C4\1F\00\02\02G\00\00\08\10\\\00\03'\05\00\81\D7K\03\037\05\00\01\17\1A\F6\07\22\FC\00\B0\1F\00\02\00\F7\0F\00\80\D7[\03\00\F7\0F\C0\01\D8[\00\F0\07\00\00\F8\03\01\F1\07\E0\FF\00\FC\1F\00\00\02\07\00\00\00\90\A0\0F\00\07\00\00\00\00\E3\0F\00\87\FF\FF\0F@\E2\E0\07\00\FC\00\80\1F\00\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\B7\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F7\00\00\00\00\00\00\00\CA\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\01\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00d\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00p\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\00x\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\05\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\03\00\00\00\04\00\00\05 \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00`\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\02\00\00\00\00\00\008\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  gpu.module @main_kernel_1 attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00`\08\00\00\00\00\00\00 \06\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.constant0.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00main_kernel\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.constant0.main_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00>\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BB\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\06\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\04\00\88\00\00\00\04\1D\08\00\10\00\00\00\18\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\03\198\00\04\0A\08\00\02\00\00\00@\018\00\01*\00\00\010\00\00\047\04\00r\00\00\00\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07 \E2\00\FC\1C\00\01\00\87\00\80\07\98L\00\00W\02\00\00\C8\F0\02\00g\02\00\00\C8\F0\F1\0F\22\FE\02\98\1F\00\04\00\D7\01\00\00)8\03\00'\00\80\81\18\\\02\02\F7\01\00\00)8\F1\07\C0\FE@\C4\1F\00\02\02G\00\00\08\10\\\00\03'\05\00\81\D7K\02\037\05\00\01\17\1A\F6\07\A2\FD\00\C4\1F\00\04\00\F7\0F\00\80\D7[\05\00\F7\0F@\01\D8[\FF\04\07\00\00\00\90\A0\FF\07\E0\FF\00\80\1F\00\0F\00\07\00\00\00\00\E3\0F\00\87\FF\FF\0F@\E2\00\0F\07\00\00\00\B0P\E0\07\00\FC\00\80\1F\00\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\B7\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F7\00\00\00\00\00\00\00\CA\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\01\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00d\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00p\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\00x\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\05\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\03\00\00\00\04\00\00\06 \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00`\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\02\00\00\00\00\00\008\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
}

```

</details>

### `--gpu-to-llvm`

The final step, convert everything to LLVM dialect. The GPU kernels stay as binary bitcode.
<details>
  <summary> code </summary>

```mlir
module attributes {gpu.container_module} {
  func @matmul_linalg(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @matmul_linalg_kernel::@matmul_linalg_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>, %arg2 : memref<8x8xf32>)
    return
  }
  gpu.module @matmul_linalg_kernel attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00\A0\0C\00\00\00\00\00\00`\0A\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.matmul_linalg_kernel\00.nv.info.matmul_linalg_kernel\00.nv.shared.matmul_linalg_kernel\00.nv.constant0.matmul_linalg_kernel\00.rel.nv.constant0.matmul_linalg_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00matmul_linalg_kernel\00.text.matmul_linalg_kernel\00.nv.info.matmul_linalg_kernel\00.nv.shared.matmul_linalg_kernel\00.rel.nv.constant0.matmul_linalg_kernel\00.nv.constant0.matmul_linalg_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00G\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C7\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F1\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00@\03\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\0D\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\08\00\B0\00\00\00\18\03\00\00\04\1D\08\00\18\00\00\000\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\04\17\0C\00\00\00\00\00\0D\00h\00\00\F0!\00\04\17\0C\00\00\00\00\00\0E\00p\00\00\F0!\00\04\17\0C\00\00\00\00\00\0F\00x\00\00\F0!\00\04\17\0C\00\00\00\00\00\10\00\80\00\00\F0!\00\04\17\0C\00\00\00\00\00\11\00\88\00\00\F0!\00\04\17\0C\00\00\00\00\00\12\00\90\00\00\F0!\00\04\17\0C\00\00\00\00\00\13\00\98\00\00\F0!\00\04\17\0C\00\00\00\00\00\14\00\A0\00\00\F0!\00\03\19\A8\00\04\0A\08\00\02\00\00\00@\01\A8\00\01*\00\00\010\00\00\047\04\00r\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07\00\FE\00X\1C\00\01\00\87\00\80\07\98L\FF\FF\97\FF\FF\FF\1F\1C\00\00W\02\00\00\C8\F0\F0\07 \E6\00\C4?\08\07\FF\F7\0F\80\0Bg[\02\00g\02\00\00\C8\F0\03\00\F7\01\00\00)8\F1\17 \FE@\94\1F\00\05\02\F7\01\00\00)8\04\00'\00\80\81\D7[\07\00\B7\01\00\00)8\F1\07\C2\FC\00\C4\1F\00\03\00W0\C0\01\DF[\05\00'\05\80\82\18L\07\077\05\00\08\10L\F6\07\E0\FD\00\A0\1F\00\00\04\E7\06\00\81\D7K\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\F0\07\C0\FE\00\C4\1F\08\03\04\F7\06\80\01\17\1A\0F\00\08\00\00\00\00\E3\06\05\F7\0F\00\80\D7[\F5\07\22\FE\00\98\1F\00\04\02\E7\01\00\00)8\07\05\F7\0F\C0\03\D8[\02\02\07\06\00\81\18L\F1\07\C0\FE@\C4\1F\00\05\04\17\06\00\08\10L\04\02\F7\0F\00\80\D7[\05\02\F7\0F\C0\02\D9[\F6\07\02\FE\00\D8\1E\00\02\00\F7\0F\00\80\D7[\03\00\F7\0F\C0\01\DA[\00\06\07\00\00\00\90\80\B7\07@\F6\00\98\1F\04\09\04\07\00\00\00\90\84\08\02\07\00\00\00\90\88\09\00\97\00\00\00h\\\E4\07 \1E\00\C4\1E\00\08\08\97\00\00\00X\\\08\02\07\00\00\00\90\A8\00\06G\00\00\00\90\80\B1\07@\F6\00\98\1F\04\09\04\07\02\00\00\90\84\0A\02\07\00\00\00\90\88\00\00\97\00\00\00h\\\E4\07 >\00\C4\1E\00\00\0A\07\00\00\00X\\\00\02\07\00\00\00\90\A8\09\06\87\00\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\04\00\00\90\84\0B\02\07\00\00\00\90\88\09\09\A7\00\00\00h\\\E4\07 ^\00\C4>\00\09\0B\97\00\00\00X\\\09\02\07\00\00\00\90\A8\08\06\C7\00\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\06\00\00\90\84\0B\02\07\00\00\00\90\88\08\08\A7\00\00\00h\\\E4\07 \1E\00\C4^\00\08\0B\87\00\00\00X\\\08\02\07\00\00\00\90\A8\00\06\07\01\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\08\00\00\90\84\0B\02\07\00\00\00\90\88\00\00\A7\00\00\00h\\\E4\07 >\00\C4\9E\00\00\0B\07\00\00\00X\\\00\02\07\00\00\00\90\A8\09\06G\01\00\00\90\80\B1\07@\F6\00\98\1F\04\0A\04\07\0A\00\00\90\84\0B\02\07\00\00\00\90\88\09\09\A7\00\00\00h\\\E4\07 ^\00\C4>\00\0B\0B\97\00\00\00X\\\0B\02\07\00\00\00\90\A8\08\06\87\01\00\00\90\80\B1\07@\F6\00\98\1F\04\09\04\07\0C\00\00\90\84\0A\02\07\00\00\00\90\88\08\08\97\00\00\00h\\\E4\07 ^\00\C4^\00\0C\0A\87\00\00\00X\\\0C\02\07\00\00\00\90\A8\00\06\C7\01\00\00\90\80\B1\07@\F6\00\98\1F\04\09\04\07\0E\00\00\90\84\08\02\07\00\00\00\90\88\09\00\97\00\00\00h\\\E6\07 \FE\04\FC\1F\00\08\08\97\00\00\00X\\\08\02\07\00\00\00\90\A8\0F\00\07\00\00\00\00\E3\FF\07\00\FC\00\80\1F\00\0F\00\07\FF\FF\0F@\E2\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\E4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$\01\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\02\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A0\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00M\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C4\02\00\00\00\00\00\00\8C\01\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D5\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\04\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\8B\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\05\00\00\00\00\00\00\E8\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00 \07\00\00\00\00\00\00@\03\00\00\00\00\00\00\03\00\00\00\04\00\00\0D \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\A0\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00(\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\05\00\00\00\00\00\008\05\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @matmul_linalg_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %24 = llvm.mlir.constant(8 : index) : i64
      %25 = llvm.mlir.constant(0 : index) : i64
      %26 = llvm.mlir.constant(1 : index) : i64
      %27 = nvvm.read.ptx.sreg.ctaid.x : i32
      %28 = llvm.sext %27 : i32 to i64
      %29 = nvvm.read.ptx.sreg.ctaid.y : i32
      %30 = llvm.sext %29 : i32 to i64
      llvm.br ^bb1(%25 : i64)
    ^bb1(%31: i64):  // 2 preds: ^bb0, ^bb2
      %32 = llvm.icmp "slt" %31, %24 : i64
      llvm.cond_br %32, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %33 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %34 = llvm.mlir.constant(8 : index) : i64
      %35 = llvm.mul %28, %34  : i64
      %36 = llvm.add %35, %31  : i64
      %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %38 = llvm.load %37 : !llvm.ptr<f32>
      %39 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %40 = llvm.mlir.constant(8 : index) : i64
      %41 = llvm.mul %31, %40  : i64
      %42 = llvm.add %41, %30  : i64
      %43 = llvm.getelementptr %39[%42] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %44 = llvm.load %43 : !llvm.ptr<f32>
      %45 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %46 = llvm.mlir.constant(8 : index) : i64
      %47 = llvm.mul %28, %46  : i64
      %48 = llvm.add %47, %30  : i64
      %49 = llvm.getelementptr %45[%48] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %50 = llvm.load %49 : !llvm.ptr<f32>
      %51 = llvm.fmul %38, %44  : f32
      %52 = llvm.fadd %50, %51  : f32
      %53 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %54 = llvm.mlir.constant(8 : index) : i64
      %55 = llvm.mul %28, %54  : i64
      %56 = llvm.add %55, %30  : i64
      %57 = llvm.getelementptr %53[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %52, %57 : !llvm.ptr<f32>
      %58 = llvm.add %31, %26  : i64
      llvm.br ^bb1(%58 : i64)
    ^bb3:  // pred: ^bb1
      llvm.return
    }
  }
  func @main() {
    %c8 = arith.constant 8 : index
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
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%0 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%1 : memref<8x8xf32>)
    gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%2 : memref<8x8xf32>)
    call @matmul_linalg(%0, %1, %2) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00`\08\00\00\00\00\00\00 \06\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.constant0.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00main_kernel\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.constant0.main_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00>\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BB\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\05\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\04\00\90\00\00\00\04\1D\08\00\10\00\00\00\18\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\03\198\00\04\0A\08\00\02\00\00\00@\018\00\01*\00\00\010\00\00\047\04\00r\00\00\00\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07 \E2\00\FC\1C\00\01\00\87\00\80\07\98L\00\00W\02\00\00\C8\F0\02\00g\02\00\00\C8\F0\F1\0F\22\FE\02\98\1F\00\04\00\D7\01\00\00)8\03\00'\00\80\81\18\\\02\02\F7\01\00\00)8\F1\07\C0\FE@\C4\1F\00\02\02G\00\00\08\10\\\00\03'\05\00\81\D7K\03\037\05\00\01\17\1A\F6\07\22\FC\00\B0\1F\00\02\00\F7\0F\00\80\D7[\03\00\F7\0F\C0\01\D8[\00\F0\07\00\00\F8\03\01\F1\07\E0\FF\00\FC\1F\00\00\02\07\00\00\00\90\A0\0F\00\07\00\00\00\00\E3\0F\00\87\FF\FF\0F@\E2\E0\07\00\FC\00\80\1F\00\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\B7\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F7\00\00\00\00\00\00\00\CA\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\01\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00d\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00p\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\00x\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\05\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\03\00\00\00\04\00\00\05 \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00`\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\02\00\00\00\00\00\008\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  gpu.module @main_kernel_0 attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00`\08\00\00\00\00\00\00 \06\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.constant0.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00main_kernel\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.constant0.main_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00>\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BB\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\05\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\04\00\90\00\00\00\04\1D\08\00\10\00\00\00\18\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\03\198\00\04\0A\08\00\02\00\00\00@\018\00\01*\00\00\010\00\00\047\04\00r\00\00\00\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07 \E2\00\FC\1C\00\01\00\87\00\80\07\98L\00\00W\02\00\00\C8\F0\02\00g\02\00\00\C8\F0\F1\0F\22\FE\02\98\1F\00\04\00\D7\01\00\00)8\03\00'\00\80\81\18\\\02\02\F7\01\00\00)8\F1\07\C0\FE@\C4\1F\00\02\02G\00\00\08\10\\\00\03'\05\00\81\D7K\03\037\05\00\01\17\1A\F6\07\22\FC\00\B0\1F\00\02\00\F7\0F\00\80\D7[\03\00\F7\0F\C0\01\D8[\00\F0\07\00\00\F8\03\01\F1\07\E0\FF\00\FC\1F\00\00\02\07\00\00\00\90\A0\0F\00\07\00\00\00\00\E3\0F\00\87\FF\FF\0F@\E2\E0\07\00\FC\00\80\1F\00\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\B7\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F7\00\00\00\00\00\00\00\CA\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\01\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00d\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00p\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\00x\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\05\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\03\00\00\00\04\00\00\05 \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00`\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\02\00\00\00\00\00\008\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  gpu.module @main_kernel_1 attributes {gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00r\00\00\00\00\00\00\00\00\00\00\00`\08\00\00\00\00\00\00 \06\00\00\00\00\00\00=\05#\00@\008\00\03\00@\00\09\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.nv.constant0.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00main_kernel\00.text.main_kernel\00.nv.info.main_kernel\00.nv.shared.main_kernel\00.rel.nv.constant0.main_kernel\00.nv.constant0.main_kernel\00_param\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00>\00\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\9A\00\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BB\00\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\08\00\00\00\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\04\11\08\00\04\00\00\00\00\00\00\00\04/\08\00\04\00\00\00\06\00\00\00\04\12\08\00\04\00\00\00\00\00\00\00\04\1C\04\00\88\00\00\00\04\1D\08\00\10\00\00\00\18\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\03\198\00\04\0A\08\00\02\00\00\00@\018\00\01*\00\00\010\00\00\047\04\00r\00\00\00\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F6\07 \E2\00\FC\1C\00\01\00\87\00\80\07\98L\00\00W\02\00\00\C8\F0\02\00g\02\00\00\C8\F0\F1\0F\22\FE\02\98\1F\00\04\00\D7\01\00\00)8\03\00'\00\80\81\18\\\02\02\F7\01\00\00)8\F1\07\C0\FE@\C4\1F\00\02\02G\00\00\08\10\\\00\03'\05\00\81\D7K\02\037\05\00\01\17\1A\F6\07\A2\FD\00\C4\1F\00\04\00\F7\0F\00\80\D7[\05\00\F7\0F@\01\D8[\FF\04\07\00\00\00\90\A0\FF\07\E0\FF\00\80\1F\00\0F\00\07\00\00\00\00\E3\0F\00\87\FF\FF\0F@\E2\00\0F\07\00\00\00\B0P\E0\07\00\FC\00\80\1F\00\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\0F\07\00\00\00\B0P\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00\B7\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F7\00\00\00\00\00\00\00\CA\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C8\01\00\00\00\00\00\00x\00\00\00\00\00\00\00\02\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\02\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00d\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\03\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00p\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\00x\01\00\00\00\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\05\00\00\00\00\00\00\C0\00\00\00\00\00\00\00\03\00\00\00\04\00\00\06 \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00`\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\E8\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\008\02\00\00\00\00\00\008\02\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @main_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %8 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = llvm.sext %9 : i32 to i64
      %11 = nvvm.read.ptx.sreg.ctaid.y : i32
      %12 = llvm.sext %11 : i32 to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %14 = llvm.mlir.constant(8 : index) : i64
      %15 = llvm.mul %10, %14  : i64
      %16 = llvm.add %15, %12  : i64
      %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %8, %17 : !llvm.ptr<f32>
      llvm.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
}

```

</details>