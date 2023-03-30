#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#include <omp.h>
#include <pthread.h>

#define ROWS 1500
#define COLS 1500
#define THREADS 8

// Global Declaration of 2D Matrices Used
int** A;
int** B;
int** C;

void* init(void* ID) {
	// pThread Runner Function
	int id = *(int*)ID;
	
	int start_row, end_row;
	start_row = id * ROWS / THREADS;
	end_row = (id + 1) * ROWS / THREADS;

	printf("pThread Thread %d Running (SR=%d ER=%d)...\n", id, start_row, end_row);
	for (int i = start_row; i < end_row; i++) {
		for (int j = 0; j < COLS; j++) {
			C[i][j] = 0;
			for (int k = 0; k < COLS; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	printf("pThread Thread %d Finished.\n", id);
}
void PTMM(int** A, int** B, int** C) {
	// Parallel Matrix Multiplication Method Using pThreads
	printf("pThreads - Multiplying Matrices...\n\n");
	pthread_t T[THREADS];
	int IDS[THREADS];
	for (int i = 0; i < THREADS; i++) {
		IDS[i] = i;
		pthread_create(&T[i], NULL, init, (void*)&IDS[i]);
	}
	for (int i = 0; i < THREADS; i++) {
		pthread_join(T[i], NULL);
	}
	printf("\n");
}
void OMPMM(int** A, int** B, int** C) {
	// Parallel Matrix Multiplication Method Using OMP
	printf("OpenMP - Multiplying Matrices...\n\n");
	omp_set_num_threads(THREADS);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int i;
		printf("OMP Thread %d Running...\n", id);
		#pragma omp for
		for (i = 0; i < ROWS; i++) {
			for (int j = 0; j < COLS; j++) {
				C[i][j] = 0;
				for (int k = 0; k < COLS; k++) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
		printf("OMP Thread %d Finished.\n", id);
	}
	printf("\n");
}
void SMM(int** A, int** B, int** C) {
	// Sequential Matrix Multiplication Method
	printf("Sequential - Multiplying Matrices...\n\n");
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			C[i][j] = 0;
			for (int k = 0; k < COLS; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}
void printMatrix(int** A) {
	// Matrix Printing Method
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < ROWS; j++) {
			printf("%d ", A[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
void main() {
	// Starter Code
	int choice = 0;
	printf("Matrix Multiplication\n\n");
	printf("Please Choose:\n1 - Sequential\n2 - Parallel (OpenMP)\n3 - Parallel (pThreads)\n");
	scanf_s("%d", &choice);
	printf("\n");
	if (choice > 3 || choice < 1) {
		printf("Invalid Choice\nExiting");
		exit(1);
	}

	// Memory Allocation and Error Handling
	A = (int**)malloc(ROWS * sizeof(int*));
	B = (int**)malloc(ROWS * sizeof(int*));
	C = (int**)malloc(ROWS * sizeof(int*));
	if (A == NULL || B == NULL || C == NULL) {
		exit(1);
	}
	for (int i = 0; i < ROWS; i++) {
		A[i] = (int*)malloc(COLS * sizeof(int));
		if (A[i] == NULL) {
			exit(1);
		}
	}
	for (int i = 0; i < ROWS; i++) {
		B[i] = (int*)malloc(COLS * sizeof(int));
		if (B[i] == NULL) {
			exit(1);
		}
	}
	for (int i = 0; i < ROWS; i++) {
		C[i] = (int*)malloc(COLS * sizeof(int));
		if (C[i] == NULL) {
			exit(1);
		}
	}

	// Filling Matrices with Values
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < ROWS; j++) {
			A[i][j] = rand() % 10;
			B[i][j] = rand() % 10;
			C[i][j] = 0;
		}
	}

	// Printing Matrices
	if (ROWS <= 20 && COLS <= 20) {
		printf("Input Matrix A:\n");
		printMatrix(A);
		printf("Input Matrix B:\n");
		printMatrix(B);
	}

	// Declaring Timer Variables
	clock_t start, end;
	double cpuTime;

	// Starting Timer and Executing Chosen MM
	start = clock();
	if (choice == 1) {
		SMM(A, B, C);
	}
	else if (choice == 2) {
		OMPMM(A, B, C);
	}
	else if (choice == 3) {
		PTMM(A, B, C);
	}
	
	end = clock();
	cpuTime = ((double)(end - start)) / CLOCKS_PER_SEC;

	// Printing Output Matrix and Time Taken
	if (ROWS<=20 && COLS<=20) {
		printf("Output Matrix C:\n");
		printMatrix(C);
	}
	printf("\nCPU Time Taken: %.6f\n\n", cpuTime);

	// Deallocating Used Memory
	for (int i = 0; i < ROWS; i++) {
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}
	free(A);
	free(B);
	free(C);
}