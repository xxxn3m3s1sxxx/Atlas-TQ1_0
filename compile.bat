@echo off
REM Build atlas.dll from source using Clang (LLVM MinGW)
REM Requires: llvm-mingw in PATH

set CC=clang++

echo [Atlas] Compiling atlas.dll...
%CC% -shared -o atlas.dll atlas_api.cpp -O2 -mavx2 -mfma -fopenmp -std=c++17

if %ERRORLEVEL% EQU 0 (
    echo [Atlas] OK -- atlas.dll built successfully
) else (
    echo [Atlas] FAILED -- see errors above
    echo [Atlas] Try: set CC=x86_64-w64-mingw32-clang++
    exit /b 1
)
