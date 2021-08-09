%%cu
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

const int N = 6;

int main()
{
	srand(time(NULL));
	int A[N];
	for(int i = 0; i < N; i++)
		A[i] = rand() % 100;

	printf("Unsorted Array: \n");
	for(int i = 0; i < 6; i++)
		printf("%d ", A[i]);

	thrust::sort(thrust::host, A, A + N);

	printf("\nSorted Array: \n");
	for(int i = 0; i < 6; i++)
		printf("%d ", A[i]);
}