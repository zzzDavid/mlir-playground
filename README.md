# mlir-playground
My Playground with MLIR stuff

## GPU Backend
This example demonstrates how to target NVIDIA GPU backend from Linalg dialect.
I successfully lowered the Linalg matmul example to LLVM/NVVM dialect and created both a JiT and an executable.