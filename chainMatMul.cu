/*
* multNoShare.c
*
*/
#include "matrix.cuh"
#include <stdio.h>


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
// Parallel multiplication of Matrices

__global__ void ChainMatMulKernel(Matrix* Chain, int* Muls, Matrix* IntRes) {
	int threadId = blockIdx.x;
	int row = blockIdx.y;
	int col = blockIdx.z;
	int Cvalue = 0;

	int mulNum = Muls[threadId];

	if(row >= Chain[mulNum].height || col >= Chain[mulNum + 1].width) return;
	for (int e = 0; e < Chain[mulNum].width; ++e){
		Cvalue += ((Chain[mulNum].elements[row * Chain[mulNum].width + e]) * (Chain[mulNum + 1].elements[e * Chain[mulNum + 1].width + col]));
		Cvalue = Cvalue	% 256;
	}
	IntRes[threadId].elements[row * IntRes[threadId].width + col] = Cvalue;	   
}

void SequentialSelectionSortDouble(int* array, int* arrayOrder, int n) {
	int position, swap, swapOrder, d, c;

	for(c = 0 ; c < ( n - 1 ) ; c++) {
		position = c;
 
		for ( d = c + 1 ; d < n ; d++ ) {
			if ( array[position] < array[d] )
				position = d;
		}
		if(position != c) {
			swap = array[c];
			swapOrder = arrayOrder[c];
		 
			array[c] = array[position];
			arrayOrder[c] = arrayOrder[position];

			array[position] = swap;
			arrayOrder[position] = swapOrder;
		}
	}
}

void SequentialSelectionSort(int* array, int n) {
	int position, swap, d, c;

	for(c = 0 ; c < ( n - 1 ) ; c++) {
		position = c;
 
		for ( d = c + 1 ; d < n ; d++ ) {
			if ( array[position] > array[d] )
				position = d;
		}
		if(position != c) {
			swap = array[c];
		 
			array[c] = array[position];

			array[position] = swap;
		}
	}
}

Matrix ChainMatMul(Matrix* Chain, int numMats) {

	int n = numMats;
	Matrix Result;
	Matrix* h_Chain; // Only elements on device
	Matrix* d_Chain; // Array fully on device
	Matrix* h_IntRes; // Only elements on device
	Matrix* d_IntRes; // Array fully on device
	int* ChainDims;
	int* ChainDimOrder;
	int numDims;
	int* h_muls; // Array on host
	int* d_muls; // Array on device
	int numMuls = 0;
	
	h_Chain = (Matrix*)malloc(n*sizeof(Matrix));
	
	size_t size;
	cudaError_t err;
	
	// Transfer from Chain to h_Chain
	for(int i = 0; i < n;++i) {	        
		h_Chain[i].width = Chain[i].width;
		h_Chain[i].height = Chain[i].height;
		size = h_Chain[i].width * h_Chain[i].height * sizeof(int);
		err = cudaMalloc(&h_Chain[i].elements, size);
		//printf("CUDA malloc Chain[%d].elements: %s\n", i, cudaGetErrorString(err));
		err = cudaMemcpy(h_Chain[i].elements, Chain[i].elements, size, cudaMemcpyHostToDevice);
		//printf("Copy Chain[%d].elements to device: %s\n", i, cudaGetErrorString(err));
	}
		
	// Trasfer from h_Chain to d_Chain
	size = n * sizeof(Matrix);
	err = cudaMalloc(&d_Chain, size);
	//printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_Chain, h_Chain, size, cudaMemcpyHostToDevice);
	

	while (n > 1) {
		// ************************** Find optimal multiplications ******************
		// Fill up ChainDims
		numDims = n - 1;
		numMuls = 0;
		ChainDims = (int*)malloc(numDims * sizeof(int));
		ChainDimOrder = (int*)malloc(numDims * sizeof(int));
		h_muls = (int*)malloc(numDims * sizeof(int));
		for(int i = 0; i < numDims; ++i) {
			ChainDims[i] = h_Chain[i].width;
			ChainDimOrder[i] = i;
		}

		// Sort ChainDims
		SequentialSelectionSortDouble(ChainDims, ChainDimOrder, numDims);

		// Select muls
		for(int i = 0, j = 0;i < numDims; ++i) {
			if(ChainDims[i] != 0 && (numMuls < 1024)) {
				h_muls[j] = ChainDimOrder[i];
				numMuls++;
				j++;
				for(int k = 0; k < numDims; k++){
					if(ChainDimOrder[k] == (ChainDimOrder[i] + 1) || ChainDimOrder[k] == (ChainDimOrder[i] - 1)) {
						ChainDims[k] = 0;
					}
				}
			}
		}
		free(ChainDims);
		free(ChainDimOrder);

		SequentialSelectionSort(h_muls, numMuls);
		printf("\nMultiplication choices : ");
		for(int i = 0; i < numMuls; ++i) {
			printf("Mat%d x Mat%d\t", h_muls[i], (h_muls[i]+1));
		}
		printf("\n");
		// **************************************************************************

		// ********************** Transfer stuff to Device **************************
		// Transfer muls on device
		err = cudaMalloc(&d_muls, numMuls * sizeof(int));
		//printf("CUDA malloc Muls: %s\n", cudaGetErrorString(err));
		err = cudaMemcpy(d_muls, h_muls, numMuls * sizeof(int), cudaMemcpyHostToDevice);
		//printf("Copy Muls to device: %s\n", cudaGetErrorString(err));

		// Hold intermediate results on host with elements on device
		h_IntRes = (Matrix*)malloc(numMuls * sizeof(Matrix));
		
		// Allocate memory on device for the elements of h_IntRes
		for(int i = 0; i < numMuls; ++i) {
			h_IntRes[i].height = h_Chain[h_muls[i]].height;
			h_IntRes[i].width = h_Chain[h_muls[i] + 1].width;
			size_t size = h_IntRes[i].width * h_IntRes[i].height * sizeof(int);
			err = cudaMalloc(&h_IntRes[i].elements, size);
			//printf("CUDA malloc IntRes[%d]: %s\n", i, cudaGetErrorString(err));
		}
		
		// IntRes Fully on device
		size = numMuls * sizeof(Matrix);
		err = cudaMalloc(&d_IntRes, size);
		//printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
		err = cudaMemcpy(d_IntRes, h_IntRes, size, cudaMemcpyHostToDevice);
		//printf("Copy Chain to device: %s\n", cudaGetErrorString(err));
		
		// **************************************************************************

		// *************************** Actual Multiplication ************************
		dim3 dimGrid(numMuls, 256, 256);

		// Call to the kernel
		ChainMatMulKernel<<<dimGrid, 1>>>(d_Chain, d_muls, d_IntRes); 
		err = cudaThreadSynchronize();
		//printf("Run kernel: %s\n", cudaGetErrorString(err));

		// **************************************************************************

		// ************************** Readying for next cycle ***********************
		// Update chain 
		for(int i = 0; i < numMuls;++i) {
			// Free device memory
			cudaFree(h_Chain[h_muls[i]].elements);
			cudaFree(h_Chain[h_muls[i] + 1].elements);
			
			// Update the chain
			h_Chain[h_muls[i]].height = h_IntRes[i].height;
			h_Chain[h_muls[i]].width = h_IntRes[i].width;
			h_Chain[h_muls[i]].elements = h_IntRes[i].elements;
		}
		
		// Reduce the size of the h_Chain array
		for(int i = 0; i < numMuls; ++i){
			h_Chain[h_muls[i]+1].width = 0;	
			h_Chain[h_muls[i]+1].height = 0;
		}

		for(int i = 0, j =0; i < n; ++i) {
			if(h_Chain[i+j].width == 0) {
				j++;
				n--;
			}
			h_Chain[i].width = h_Chain[i + j].width;	
			h_Chain[i].height = h_Chain[i + j].height;
			h_Chain[i].elements = h_Chain[i + j].elements;
		}
		// Small memory leak here - (but removing this is difficult)

		// Refresh d_Chain
		cudaFree(d_Chain);
		
		size = n * sizeof(Matrix);
		err = cudaMalloc(&d_Chain, size);
		//printf("CUDA malloc Chain: %s\n", cudaGetErrorString(err));
		err = cudaMemcpy(d_Chain, h_Chain, size, cudaMemcpyHostToDevice);
		//printf("Copy Chain to device: %s\n", cudaGetErrorString(err));
			
		// Free stuff
		free(h_muls);
		cudaFree(d_muls);
		free(h_IntRes);
		cudaFree(d_IntRes);

		// **************************************************************************
	}
	
	// Read Result from device memory
	Result.width = h_Chain[0].width;	
	Result.height = h_Chain[0].height;
	size = Result.width * Result.height * sizeof(int);
	Result.elements =  (int*)malloc(size);
	err = cudaMemcpy(Result.elements, h_Chain[0].elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy Result off of device: %s\n",cudaGetErrorString(err));
	
	cudaFree(h_Chain[0].elements);
	cudaFree(d_Chain);
	free(h_Chain);
	
	return Result;
}

// Usage: multNoShare a1 a2 b2
int main(int argc, char* argv[]){
	Matrix* Chain;
	Matrix Result;
	int* dims; 
	
	if(argc != 2) {
		printf("Please input in the following format\n multNoShare.out [#FileName] \n");
		return 0;
	}
		
	char const* const fileName = argv[1]; /* should check that argc > 1 */

	FILE* file = fopen(fileName, "r"); /* should check the result */
	char nLine[10];
	char line[256 * 5];
	int lineNums[256];

	// Read values from file
	int n = atoi(fgets(nLine, sizeof(nLine), file)) - 1;
	int sizeDim = sizeof(char) * (n+1) * 5;

	char* dimLine = (char *)malloc(sizeDim);
	fgets(dimLine, sizeof(char) * (n+1) * 5, file);
	char oneNum[5];
	dims = (int *) malloc((n+1)*sizeof(int));

	for(int i = 0, k = 0; dimLine[i] != '\0' ;++i) {
		int j = 0;
		for(; dimLine[i] != ' ' && dimLine[i]!= '\0'; ++j,++i){
			oneNum[j] = dimLine[i];
		}
		oneNum[j] = '\0';
		dims[k++] = atoi(oneNum);
	}

	Chain = (Matrix *) malloc(n*sizeof(Matrix));
	for(int i = 0; i < n; ++i) {
		Chain[i].height = dims[i];
		Chain[i].width = dims[i+1];
		Chain[i].elements = (int*)malloc(Chain[i].width * Chain[i].height * sizeof(int));
	}

	for(int k = 0; k < n; ++k)
		for(int i = 0; i < Chain[k].height; i++){
			fgets(line, sizeof(line), file);
			for(int p = 0, q = 0; line[p] != '\0' ;++p) {
				int r = 0;
				for(; line[p] != ' ' && line[p]!= '\0'; ++r,++p){
					oneNum[r] = line[p];
				}
				oneNum[r] = '\0';
				lineNums[q++] = atoi(oneNum);
			}	
			for(int j = 0; j < Chain[k].width; j++) {
				Chain[k].elements[i*Chain[k].width + j] = lineNums[j];
			}
		}


	fclose(file); 

	printf("Print up to a 10x10 portion of the matrices - to avoid clutter");
	for(int k = 0; k < n; ++k) {
		printf("\n Chain[%d] : %d x %d\n", k, Chain[k].height, Chain[k].width);
		for(int i = 0; i < min(10, Chain[k].height); i++){
			for(int j = 0; j < min(10, Chain[k].width); j++)
				printf("%d ", Chain[k].elements[i*Chain[k].width + j]);
			printf("\n");
		}
	}
	printf("\n");

	Result = ChainMatMul(Chain, n);
	
	// Print up to a 10x10 portion of the Result
	printf("\n Result : %d x %d\n", Result.height, Result.width);
	for(int i = 0; i < min(10, Result.height); i++){
		for(int j = 0; j < min(10, Result.width); j++)
			printf("%d ", Result.elements[i*Result.width + j]);
		printf("\n");
	}
	
}
