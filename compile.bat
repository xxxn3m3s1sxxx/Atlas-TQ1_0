@echo off
REM Build atlas.dll from source using Clang (LLVM MinGW)
REM Requires: llvm-mingw in PATH
REM
REM OpenMP note: This toolchain lacks a static libomp.a, so -fopenmp is omitted.
REM The #pragma omp parallel for in atlas_matmul_f32 is gated by #ifdef _OPENMP.
REM If you have a toolchain with static OpenMP, add -fopenmp to enable multi-core.

set CC=clang++

echo [Atlas] Compiling atlas.dll...
%CC% -shared -o atlas.dll atlas_api.cpp -O2 -mavx2 -mfma -std=c++17

if %ERRORLEVEL% EQU 0 (
    echo [Atlas] OK -- atlas.dll built successfully
) else (
    echo [Atlas] FAILED -- see errors above
    echo [Atlas] Try: set CC=x86_64-w64-mingw32-clang++
    exit /b 1
)
