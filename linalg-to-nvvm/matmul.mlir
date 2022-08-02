// compile linalg operators to GPU dialect 
// https://gist.github.com/chudur-budur/cbf24834f22c23e9aec0c2fbcfde1b0d

// refer to /test/Conversion/SCFToGPU/parallel_loop.mlir
module {
    func @matmul_linalg(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
        // A x B + C
        linalg.matmul ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
            outs(%C: memref<8x8xf32>)
        return
    }

    func @main() -> (){
        %A = memref.alloc() : memref<8x8xf32>
        %B = memref.alloc() : memref<8x8xf32>
        %C = memref.alloc() : memref<8x8xf32>

        %cast_A = memref.cast %A : memref<8x8xf32> to memref<*xf32>
        gpu.host_register %cast_A : memref<*xf32>
        %cast_B = memref.cast %B : memref<8x8xf32> to memref<*xf32>
        gpu.host_register %cast_B : memref<*xf32>
        %cast_C = memref.cast %C : memref<8x8xf32> to memref<*xf32>
        gpu.host_register %cast_C : memref<*xf32>
        
        %cf1 = std.constant 1.0 : f32
        %cf0 = std.constant 0.0 : f32
        
        linalg.fill(%cf1, %A) : f32, memref<8x8xf32>
        linalg.fill(%cf1, %B) : f32, memref<8x8xf32>
        linalg.fill(%cf0, %C) : f32, memref<8x8xf32>
        
        call @matmul_linalg(%A, %B, %C) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>) -> ()
        call @print_memref_f32(%cast_C) : (memref<*xf32>) -> ()
        return
    }
    func private @print_memref_f32(%ptr : memref<*xf32>)
}