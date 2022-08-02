module {
  func @top(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) -> memref<32x32xf32> attributes {itypes = "__", otypes = "_"} {
    %0 = memref.alloc() {name = "C"} : memref<32x32xf32>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c32_1 = arith.constant 32 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c32, %arg9 = %c1_3, %arg10 = %c1_3) threads(%arg5, %arg6, %arg7) in (%arg11 = %c32_1, %arg12 = %c1_3, %arg13 = %c1_3) {
      %1 = arith.addi %c0, %arg2 : index
      %2 = arith.addi %c0_0, %arg5 : index
      %3 = memref.alloc() {name = "sum_rv"} : memref<1xf32>
      %c0_4 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      memref.store %cst, %3[%c0_4] {to = "sum_rv"} : memref<1xf32>
      affine.for %arg14 = 0 to 32 {
        %5 = memref.load %arg0[%1, %arg14] {from = "A"} : memref<32x32xf32>
        %6 = memref.load %arg1[%arg14, %2] {from = "B"} : memref<32x32xf32>
        %7 = arith.mulf %5, %6 : f32
        %8 = memref.load %3[%c0_4] {from = "sum_rv"} : memref<1xf32>
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %3[%c0_4] {to = "sum_rv"} : memref<1xf32>
      } {loop_name = "k", reduction}
      %4 = memref.load %3[%c0_4] {from = "sum_rv"} : memref<1xf32>
      memref.store %4, %0[%1, %2] {to = "C"} : memref<32x32xf32>
      gpu.terminator
    }
    return %0 : memref<32x32xf32>
  }
}

