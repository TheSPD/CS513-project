#include <stdio.h>
#include <math.h>

class Matrix {
    int m, n;
    double** hElements;
    double** dElements;
    
    public:
    __host__ Matrix(int n1, int n2, double arr[]){
        m = n1;
        n = n2;
        hElements = new double*[m];
        for(int i = 0; i < m; ++i) {
            hElements[i] = new double[n];
        }
        
        for(int i = 0; i < m; ++i) {
            for(int j=0; j < n; ++j) {
                hElements[i][j] = arr[i * n + j];
            }
        }
        
        toDevice();
    }
    
    __host__ void toDevice() {
        cudaMalloc((void**)&dElements, sizeof(double) * m * n);
        cudaMemcpy(&dElements, &hElements, n*m*sizeof(double),cudaMemcpyHostToDevice);
    }
    
    __global__ void traverseMatrix() {
        
    }
    
    void print() {
        printf("A %d x %d Matrix", m, n);
        
        for(int i = 0; i < m; ++i) {
            printf("\n");
            for(int j=0; j < n; ++j) {
                printf("%lg\t", hElements[i][j]);
            }
            
        }
    }
};


typedef unsigned long long int uint64_t;

/******************************************************************************
*
******************************************************************************/

int main(int argc, const char * argv[]) {

    double arr[] = {1.,2.,3.,4.};
    Matrix m(2,2, arr);
    m.print();
    
    return 0;

}
