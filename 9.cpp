/*
 *  Author   : S. Karthik Chandra
 *  Roll     : CS17B026
 *  C[m][p] = A[m][n] *  B[n][p]
 *  Matrix multiply using OpenMP. Threads share row iterations according chunk size.
 */

#include <omp.h>
#include <random>
#include <chrono>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    if(argc < 5)
    {
        cerr << "Dimensions of the matrices, Number of threads are expected as arguments!\nFormat: ./a.out <m> <n> <p> <nthreads>" << endl;
        exit(0);
    }

    /* Getting dimensions and number of threads from arguments */
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);
    int nthreads = atoi(argv[4]);
    omp_set_num_threads(nthreads);
    
    /* Allocating memory for matrices */
    double** a = new double*[m];
    for(int i = 0; i < m; ++i)
        a[i] = new double[n];

    double** b = new double*[n];
    for(int i = 0; i < n; ++i)
        b[i] = new double[p];
    
    double** c = new double*[m];
    for(int i = 0; i < m; ++i)
        c[i] = new double[p];

    double** d = new double*[m];
    for(int i = 0; i < m; ++i)
        d[i] = new double[p];
    
    if(!a || !b || !c)
    {
        cerr << "Memory allocation for array failed" << endl;
        exit(0);
    }

    int tid, i, j, k, chunk;
    uniform_real_distribution<double> dist(0, 10);
    default_random_engine re;
    chunk = m / nthreads;       /* set loop iteration chunk size */

    /* Spawn a parallel region explicitly scoping all variables */
    #pragma omp parallel shared(a, b, c, chunk, dist, re) private(tid, i, j, k)
    {
        tid = omp_get_thread_num();

        /* Initialize matrices randomly */
        #pragma omp for schedule(static, chunk)
        for (i = 0; i < m; i++)
            for (j = 0; j < n; j++)
                a[i][j] = dist(re);
        #pragma omp for schedule(static, chunk)
        for (i = 0; i < n; i++)
            for (j = 0; j < p; j++)
                b[i][j] = dist(re);
        #pragma omp for schedule(static, chunk)
        for (i = 0; i < m; i++)
            for (j = 0; j < p; j++)
                c[i][j] = 0;

        /* Do matrix multiply sharing iterations on outer loop i.e rows of matrix C */
        // cout << "Thread " << tid << "starting matrix multiply..." << endl;
        #pragma omp for schedule(static, chunk)
        for (i = 0; i < m; i++)
        {
            // cout << "Thread=" << tid << " did row=" << i << endl;
            for (j = 0; j < p; j++)
                for (k = 0; k < n; k++)
                    c[i][j] += a[i][k] * b[k][j];
        }
    }

    /* Print results */
    cout << "******************************************************" << endl;
    cout << "Produt of given matrices:" << endl;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < p; j++)
            printf("%6.2f   ", c[i][j]);
        cout << endl;
    }
    cout << "******************************************************" << endl;
}