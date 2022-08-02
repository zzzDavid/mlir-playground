export LLVM_INSTALL_DIR=/work/shared/users/phd/nz264/llvm-project-14.0.0
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/work/shared/common/usr/local/lib:/work/shared/common/usr/local/lib64:$LD_LIBRARY_PATH
export PATH=$LLVM_INSTALL_DIR/build/bin/:$PATH

mlir-opt lowered-promoted.mlir \
    --gpu-kernel-outlining \
    --lower-affine \
    --convert-scf-to-std \
    --canonicalize \
    --pass-pipeline="gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)" \
    --gpu-to-llvm 