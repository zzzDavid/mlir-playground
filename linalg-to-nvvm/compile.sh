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
as -o matmul.o matmul.s 
clang++ matmul.o -L$LLVM_INSTALL_DIR/build/lib -o exec -lcuda -lmlir_cuda_runtime -lmlir_runner_utils -lmlir_c_runner_utils
./exec