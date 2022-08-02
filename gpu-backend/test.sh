export LLVM_INSTALL_DIR=/work/shared/users/phd/nz264/llvm-project-14.0.0
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/shared/common/usr/local/lib:/work/shared/common/usr/local/lib64:$LD_LIBRARY_PATH
export PATH=$LLVM_INSTALL_DIR/build/bin/:$PATH


# mlir-opt matmul.mlir \
#     --convert-linalg-to-parallel-loops \
#     --test-gpu-greedy-parallel-loop-mapping \
#     --convert-parallel-loops-to-gpu \
#     --gpu-kernel-outlining \
#     --lower-affine \
#     --convert-scf-to-std \
#     --canonicalize \
#     --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" --gpu-to-llvm \
# | mlir-cpu-runner \
#     --shared-libs=$LLVM_INSTALL_DIR/build/lib/libmlir_cuda_runtime.so \
#     --shared-libs=$LLVM_INSTALL_DIR/build/lib/libmlir_runner_utils.so \
#     --shared-libs=$LLVM_INSTALL_DIR/build/lib/libmlir_c_runner_utils.so \
#     --entry-point-result=void

mlir-opt matmul.mlir \
    --convert-linalg-to-parallel-loops \
    # --test-gpu-greedy-parallel-loop-mapping \
    # --convert-parallel-loops-to-gpu \
    # --gpu-kernel-outlining \
    # --lower-affine \
    # --convert-scf-to-std \
    # --canonicalize \
    # --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" --gpu-to-llvm \