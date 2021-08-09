%%cu
#include <stdio.h>

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line );
        exit(EXIT_FAILURE);
    }
}

const short N = 10;

__global__ void Vector_Addition (const int *dev_a, const int *dev_b, int *dev_c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) 
        dev_c [tid] = dev_a[tid] + dev_b[tid];
}

int main(void)
{
    int Host_a[N], Host_b[N], Host_c[N];
    int *dev_a , *dev_b, *dev_c ;

    HANDLE_ERROR ( cudaMalloc((void **)&dev_a , N*sizeof(int) ) );
    HANDLE_ERROR ( cudaMalloc((void **)&dev_b , N*sizeof(int) ) );
    HANDLE_ERROR ( cudaMalloc((void **)&dev_c , N*sizeof(int) ) );

    srand(time(NULL));
    for(int i = 0; i < N; i++)
    {
        Host_a[i] = rand() % 100;
        Host_b[i] = rand() % 100; 
    }

    HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , N*sizeof(int) , cudaMemcpyHostToDevice));
    HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , N*sizeof(int) , cudaMemcpyHostToDevice));

    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    Vector_Addition <<<blocksPerGrid, threadsPerBlock>>> (dev_a , dev_b , dev_c ) ;

    HANDLE_ERROR (cudaMemcpy(Host_c , dev_c , N*sizeof(int) , cudaMemcpyDeviceToHost));

    printf("Result array: \n");
    for (int i = 0; i < N; i++)
        printf("%d + %d = %d\n", Host_a[i], Host_b[i], Host_c[i]);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
