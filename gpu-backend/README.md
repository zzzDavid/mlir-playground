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