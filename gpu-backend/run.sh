export LLVM_INSTALL_DIR=/work/shared/common/llvm-project-gpu
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/shared/common/usr/local/lib:/work/shared/common/usr/local/lib64:$LD_LIBRARY_PATH
export PATH=/work/shared/common/llvm-project-gpu/build/bin/:$PATH
# export PATH=/work/shared/common/usr/local/bin:$PATH

mlir-opt matmul.mlir \
    --convert-linalg-to-parallel-loops \
    --test-gpu-greedy-parallel-loop-mapping \
    --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining \
    --lower-affine \
    --convert-scf-to-std \
    --canonicalize \
    --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" --gpu-to-llvm \
    --print-ir-after-failure | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 > matmul.asm


