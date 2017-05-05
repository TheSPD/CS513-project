/*
* multNoShare.c
*
* Robert Hochberg
* January 24, 2012
*
* Based nearly entirely on the code from the CUDA C Programming Guide
*/
#include "matrix.cuh"


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
// Parallel multiplication of Matrices


// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row > A.height || col > B.width) return;
	for (int e = 0; e < A.width; ++e)
		Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
	C.elements[row * C.width + col] = Cvalue;
}

__device__ void MatMul(Matrix A, Matrix B, Matrix C){	
	
	for (int i = 0; i < A.height; ++i)
		for(int j = 0; j < B.width; ++j)
			C.elements[i*C.width + j] = 0;
			
	for (int i = 0; i < A.height; ++i)
		for(int j = 0; j < B.width; ++j)
			for(int k = 0; k < A.width; ++k)
				C.elements[i*B.width + j] += A.elements[i*A.width + k] * B.elements[k*B.width + j];
}

__global__ void ChainMatMulKernel(Matrix* Chain, int* Muls, Matrix* IntRes) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	
	int mulNum = Muls[threadId];	

	// Parallel Matmul code - Cannot run this on iLab machines due to incompatible hardware
	// dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// dim3 dimGrid((Chain[mulNum+1].width + dimBlock.x - 1) / dimBlock.x,
	// (Chain[mulNum].height + dimBlock.y - 1) / dimBlock.y);

	// MatMulKernel<<<dimGrid, dimBlock>>>(Chain[mulNum], Chain[mulNum + 1], IntRes[threadId]);
	
	// Sequential MatMul Code
	
	MatMul(Chain[mulNum], Chain[mulNum + 1], IntRes[threadId]);
	   
}


Matrix ChainMatMul(Matrix* Chain, int n) {

	Matrix Result;
	Matrix* h_Chain; // Elements on device
	Matrix* d_Chain; // Array fully on device
	Matrix* IntRes; // Array fully on host
	Matrix* h_IntRes; // Elements on device
	Matrix* d_IntRes; // Array fully on device

	h_Chain = (Matrix*)malloc(n*sizeof(Matrix));
	
	size_t size;
	cudaError_t err;
	
	// Transfer from Chain to h_Chain
	for(int i = 0; i < n;++i) {	        
		h_Chain[i].width = Chain[i].width;
		h_Chain[i].height = Chain[i].height;
		size = h_Chain[i].width * h_Chain[i].height * sizeof(float);
		err = cudaMalloc(&h_Chain[i].elements, size);
		printf("CUDA malloc Chain[%d].elements: %s\n", i, cudaGetErrorString(err));
		err = cudaMemcpy(h_Chain[i].elements, Chain[i].elements, size, cudaMemcpyHostToDevice);
		printf("Copy Chain[%d].elements to device: %s\n", i, cudaGetErrorString(err));
	}
		
	// Trasfer from h_Chain to d_Chain
	size = n * sizeof(Matrix);
	err = cudaMalloc(&d_Chain, size);
	printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_Chain, h_Chain, size, cudaMemcpyHostToDevice);
	printf("Copy Chain to device: %s\n", cudaGetErrorString(err));
	
	// Actual Multiplication
	dim3 dimGrid(2);
	int* h_muls; // Array on host
	int* d_muls; // Array on device
	h_muls = (int*)malloc(2 * sizeof(int));
	
	h_muls[0] = 0;
	h_muls[1] = 2;
	
	//Transfer muls on device
	err = cudaMalloc(&d_muls, 2 * sizeof(int));
	printf("CUDA malloc Muls: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_muls, h_muls, 2 * sizeof(int), cudaMemcpyHostToDevice);
	printf("Copy Muls to device: %s\n", cudaGetErrorString(err));

	// Hold intermediate results on host with elements on device
	h_IntRes = (Matrix*)malloc(2 * sizeof(Matrix));
	
	// Allocate memory on device for the elements of h_IntRes
	for(int i = 0; i < 2; ++i) {
		h_IntRes[i].height = h_Chain[h_muls[i]].height;
		h_IntRes[i].width = h_Chain[h_muls[i] + 1].width;
		size_t size = h_IntRes[i].width * h_IntRes[i].height * sizeof(float);
		err = cudaMalloc(&h_IntRes[i].elements, size);
		printf("CUDA malloc IntRes[%d]: %s\n", i, cudaGetErrorString(err));
	}
	
	// IntRes Fully on device
	size = 2 * sizeof(Matrix);
	err = cudaMalloc(&d_IntRes, size);
	printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_IntRes, h_IntRes, size, cudaMemcpyHostToDevice);
	printf("Copy Chain to device: %s\n", cudaGetErrorString(err));
	
	// Call to the kernel
	ChainMatMulKernel<<<1, dimGrid>>>(d_Chain, d_muls, d_IntRes); 
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// Update chain 
	for(int i = 0; i < 2;++i) {
		// Free device memory
		cudaFree(h_Chain[h_muls[i]].elements);
		cudaFree(h_Chain[h_muls[i] + 1].elements);
		
		// Update the chain
		h_Chain[h_muls[i]].height = h_IntRes[i].height;
		h_Chain[h_muls[i]].width = h_IntRes[i].width;
		h_Chain[h_muls[i]].elements = h_IntRes[i].elements;
	}
	
	// Reduce the size of the h_Chain array
	int i , j;
	for(i = 0, j = 0; i < n;) {
		while(h_muls[j] > i) {
			h_Chain[i].width = h_Chain[i + j].width;	
			h_Chain[i].height = h_Chain[i + j].height;
			h_Chain[i].elements = h_Chain[i + j].elements;
			i++;
		}
		i++;
		++j;
		--n;
		//Small memory leak here - should use Free h_Chain
	}

	// Refresh d_Chain
	cudaFree(d_Chain);
	
	size = n * sizeof(Matrix);
	err = cudaMalloc(&d_Chain, size);
	printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_Chain, h_Chain, size, cudaMemcpyHostToDevice);
	printf("Copy Chain to device: %s\n", cudaGetErrorString(err));
	
	//IntRes on host
	//IntRes = (Matrix *)malloc(2 * sizeof(Matrix));
	
	// Transferring to IntRes on host
	// for(int i = 0; i < 2;++i) {
	// 	IntRes[i].height = h_IntRes[i].height;
	// 	IntRes[i].width = h_IntRes[i].width;
	// 	size = IntRes[i].height * IntRes[i].width * sizeof(float);
	// 	// This bug took me two days 
	// 	IntRes[i].elements = (float *)malloc(size);
	// 	//
	// 	err = cudaMemcpy(IntRes[i].elements, h_IntRes[i].elements, size, cudaMemcpyDeviceToHost);
	// 	printf("Copy IntRes[%d] to host: %s\n", i, cudaGetErrorString(err));    
	// }

	// // Printing Intermediate results
	// for(int k = 0; k < 2; ++k) {
	// 	printf("\n IntRes[%d] : \n", k);
	// 	for(int i = 0; i < min(10, IntRes[k].height); i++){
	// 		for(int j = 0; j < min(10, IntRes[k].width); j++)
	// 			printf("%0.0f ", IntRes[k].elements[i*IntRes[k].width + j]);
	// 		printf("\n");
	// 	}
	// }

	// Free the Chain
	// for(int i = 0; i < n; ++i) {
	// 	free(Chain[i].elements);
	// }
	// free(Chain);

	// Chain = (Matrix*)malloc(n * sizeof(Matrix));

	// // Transferring to Chain on host
	// for(int i = 0; i < n;++i) {
	// 	Chain[i].height = h_Chain[i].height;
	// 	Chain[i].width = h_Chain[i].width;
	// 	size = Chain[i].height * Chain[i].width * sizeof(float);
	// 	// This bug took me two days 
	// 	Chain[i].elements = (float *)malloc(size);
	// 	//
	// 	err = cudaMemcpy(Chain[i].elements, h_Chain[i].elements, size, cudaMemcpyDeviceToHost);
	// 	printf("Copy Chain[%d] to host: %s\n", i, cudaGetErrorString(err));    
	// }

	// // Printing Chain
	// for(int k = 0; k < 2; ++k) {
	// 	printf("\n Chain[%d] : \n", k);
	// 	for(int i = 0; i < min(10, Chain[k].height); i++){
	// 		for(int j = 0; j < min(10, Chain[k].width); j++)
	// 			printf("%0.0f ", Chain[k].elements[i*Chain[k].width + j]);
	// 		printf("\n");
	// 	}
	// }
	
	// Free stuff
	free(h_muls);
	cudaFree(d_muls);
	for(int i = 0; i < 2; ++i) {
		cudaFree(h_IntRes[i].elements);
		// free(IntRes[i].elements);
	}
	free(h_IntRes);
	cudaFree(d_IntRes);
	// free(IntRes);

	// Next Cycle
	dimGrid.x = 1;
	
	h_muls = (int*)malloc(1 * sizeof(int));
	h_muls[0] = 0;
	
	err = cudaMalloc(&d_muls, 1 * sizeof(int));
	printf("CUDA malloc muls: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_muls, h_muls, 1 * sizeof(int), cudaMemcpyHostToDevice);
	printf("Copy muls to device: %s\n", cudaGetErrorString(err));
	
	h_IntRes = (Matrix*)malloc(1 * sizeof(Matrix));
	
	for(int i = 0; i < 1; ++i) {
		h_IntRes[i].height = h_Chain[h_muls[i]].height;
		h_IntRes[i].width = h_Chain[h_muls[i] + 1].width;
		size_t size = h_IntRes[i].width * h_IntRes[i].height * sizeof(float);
		err = cudaMalloc(&h_IntRes[i].elements, size);
		printf("CUDA malloc IntRes[%d]: %s\n", i, cudaGetErrorString(err));
	}
	
	size = 1 * sizeof(Matrix);
	err = cudaMalloc(&d_IntRes, size);
	printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_IntRes, h_IntRes, size, cudaMemcpyHostToDevice);
	printf("Copy Chain to device: %s\n", cudaGetErrorString(err));

	
	ChainMatMulKernel<<<1, 1>>>(d_Chain, d_muls, d_IntRes); 
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	for(int i = 0; i < 1;++i) {
		// Free device memory
		cudaFree(h_Chain[h_muls[i]].elements);
		cudaFree(h_Chain[h_muls[i] + 1].elements);
		
		h_Chain[h_muls[i]].height = h_IntRes[i].height;
		h_Chain[h_muls[i]].width = h_IntRes[i].width;
		h_Chain[h_muls[i]].elements = h_IntRes[i].elements;
	}
	
	for(int i = 0, j = 0; i < n;++i) {
		while(h_muls[j] != i++) {
			h_Chain[i].width = h_Chain[i + j].width;	
			h_Chain[i].height = h_Chain[i + j].height;
			h_Chain[i].elements = h_Chain[i + j].elements;		
		}
		++j;
		--n;
		//Small memory leak here as well
	}
	
	
	cudaFree(d_Chain);
	
	size = n * sizeof(Matrix);
	err = cudaMalloc(&d_Chain, size);
	printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_Chain, h_Chain, size, cudaMemcpyHostToDevice);
	printf("Copy Chain to device: %s\n", cudaGetErrorString(err));
	
	
	// Read Result from device memory
	Result.width = h_Chain[0].width;	
	Result.height = h_Chain[0].height;
	size = Result.width * Result.height * sizeof(float);
	Result.elements =  (float*)malloc(size);
	err = cudaMemcpy(Result.elements, h_Chain[0].elements, size, cudaMemcpyDeviceToHost);
	printf("Copy Result off of device: %s\n",cudaGetErrorString(err));
	
	
	// Free device memory
	for(int i = 0; i < n;++i) {        
		cudaFree(h_Chain[i].elements);
	}
	
	cudaFree(d_muls);
	cudaFree(d_Chain);
	free(h_muls);
	free(h_Chain);
	
	return Result;
}


// Usage: multNoShare a1 a2 b2
int main(int argc, char* argv[]){
	Matrix* Chain;
	Matrix Result;
	int* dims; 
	
	if(argc <= 3 || (argc != (atoi(argv[1]) + 3))) {
		printf("Please input in the following format\n multNoShare.out [#Matrices] [Mat1 height] [Mat1 width/Mat2 height] [Mat2 width/Mat3 height] .... [Matn width] \n");
		return 0;
	}
	
	//Read values from commandLine
	int n = atoi(argv[1]);
	
	dims = (int *) malloc((n+1)*sizeof(int));
	for(int i = 0; i <= n; ++i) {
		dims[i] = atoi(argv[i+2]);
	}
	
	Chain = (Matrix *) malloc(n*sizeof(Matrix));
	for(int i = 0; i < n; ++i) {
		Chain[i].height = dims[i];
		Chain[i].width = dims[i+1];
		Chain[i].elements = (float*)malloc(Chain[i].width * Chain[i].height * sizeof(float));
	}
	
	for(int k = 0; k < n; ++k)
		for(int i = 0; i < Chain[k].height; i++)
			for(int j = 0; j < Chain[k].width; j++)
				Chain[k].elements[i*Chain[k].width + j] = (float)(random() % 3);
		
	// Print up to a 10x10 portion of the three matrices
	for(int k = 0; k < n; ++k) {
		printf("\n Chain[%d] : \n", k);
		for(int i = 0; i < min(10, Chain[k].height); i++){
			for(int j = 0; j < min(10, Chain[k].width); j++)
				printf("%0.0f ", Chain[k].elements[i*Chain[k].width + j]);
			printf("\n");
		}
	}
	printf("\n");

	Result = ChainMatMul(Chain, n);
	
	// Print up to a 10x10 portion of the three matrices
	printf("\n Result : \n");
	for(int i = 0; i < min(10, Result.height); i++){
		for(int j = 0; j < min(10, Result.width); j++)
			printf("%0.0f ", Result.elements[i*Result.width + j]);
		printf("\n");
	}
	
}
