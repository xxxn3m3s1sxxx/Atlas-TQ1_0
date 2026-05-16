#!/bin/bash
# Compile libatlas.so for Linux x86-64 (AVX2 + FMA)
set -euo pipefail
SRC="atlas_api.cpp"
OUT="libatlas.so"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -mavx2 -mfma -mf16c -ffast-math -fopenmp -fPIC -std=c++17 -fvisibility=hidden}"
echo "[Atlas] Compiling libatlas.so..."
$CXX -shared $CXXFLAGS -o "$OUT" "$SRC"
ls -lh "$OUT"
echo "[Atlas] OK -- $OUT built successfully"
