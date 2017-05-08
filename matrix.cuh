/*
* matrix.h
*
*/
#include <stdio.h>
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    int* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32

__global__ void ChainMatMulKernel(const Matrix, const Matrix, Matrix);

