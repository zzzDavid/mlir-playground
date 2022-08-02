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

