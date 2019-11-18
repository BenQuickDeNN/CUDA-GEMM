CC="nvcc –cubin"
SRC="source/gemm.cpp"
INC="-I include"
BIN="–o bin/gemm"
OPT=""
#$CC $SRC $BIN $OPT
nvcc -x cu source/gemm.cpp -o bin/gemm -I include