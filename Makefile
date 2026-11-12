# Match paper-style flags: g++ -O3/-Ofast -march=core-avx2 + OpenMP
CXX      ?= g++
CXXFLAGS ?= -O3 -Ofast -march=core-avx2 -fPIC -shared -fopenmp -std=c++14
LDFLAGS  ?= -fopenmp

SRC = cpp/rbk_kernels.cpp
OUT = librbk_kernels.so

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) cpp/rbk_kernels.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(OUT) fill_capacity.so
