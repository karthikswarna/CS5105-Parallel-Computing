%%cu
#include <stdio.h>
#define N 100

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line );
        exit(EXIT_FAILURE);
    }
}

__global__ void ArraySum(int *array, int *sum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(sum, array[tid]);
}

int main()
{
    int array[N];
    int sum = 0;
    
    srand(time(NULL));
    for(int i = 0; i < N; i++)
        array[i] = rand() % 100;

    // for(int i = 0; i < N; i++)
    //     printf("%d ", array[i]);
    // printf("\n");

    int  *d_array, *d_sum;
    HANDLE_ERROR ( cudaMalloc((void **)&d_array, N*sizeof(int) ) );
    HANDLE_ERROR ( cudaMemcpy(d_array, array, N*sizeof(int), cudaMemcpyHostToDevice) );

    HANDLE_ERROR ( cudaMalloc((void **)&d_sum, sizeof(int) ) );
    HANDLE_ERROR ( cudaMemcpy(d_sum, &sum, sizeof(int), cudaMemcpyHostToDevice) );

    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    ArraySum<<<blocksPerGrid, threadsPerBlock>>> (d_array, d_sum);

    HANDLE_ERROR (cudaMemcpy(&sum, d_sum, sizeof(int) , cudaMemcpyDeviceToHost));
    printf("Sum: %d", sum);

    cudaFree(d_array);
    cudaFree(d_sum);

    return 0;
}