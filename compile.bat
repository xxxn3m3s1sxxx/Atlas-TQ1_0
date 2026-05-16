@echo off
REM Build atlas.dll from source using Clang (LLVM MinGW)
REM Requires: llvm-mingw in PATH
REM
REM OpenMP: enabled with -fopenmp. libomp.dll must be discoverable at runtime.
REM The LLVM-MinGW distro ships libomp.dll in x86_64-w64-mingw32\bin — copy
REM it next to atlas.dll, or add that directory to PATH.
REM
REM On Windows, numpy's MKL may load a different OpenMP runtime (libiomp5md.dll).
REM This can cause "OMP: Error #15" at import time. atlas_infer.py sets
REM KMP_DUPLICATE_LIB_OK=TRUE automatically to resolve this.

set CC=clang++

echo [Atlas] Compiling atlas.dll...
%CC% -shared -o atlas.dll atlas_api.cpp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17 -fopenmp

if %ERRORLEVEL% EQU 0 (
    echo [Atlas] OK -- atlas.dll built successfully
) else (
    echo [Atlas] FAILED -- see errors above
    echo [Atlas] Try: set CC=x86_64-w64-mingw32-clang++
    exit /b 1
)
