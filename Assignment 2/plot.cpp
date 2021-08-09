#include <iostream>
#include <limits>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

// for i in {10..1000..100} do; ./a.out $i 0.01 done >> data.txt
// 

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cerr << "Size of the matrix and the threshold are expected as arguments!\nFormat: ./a.out <size> <threshold>" << endl;
        exit(0);
    }

    int N {atoi(argv[1])};
    double threshold = 0.01;
    // omp_set_num_threads(sqrt(N));

    /* Dynamically allocating memory for 2D matrices */
    double **input = new double*[N];
    for(int i = 0; i < N; i++)
        input[i] = new double[N];

    double **output = new double*[N];
    for(int i = 0; i < N; i++)
        output[i] = new double[N];


    /* Assigning random values to the matrix */
    uniform_real_distribution<double> dist(0, 10);
    default_random_engine re;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            input[i][j] = dist(re);

    auto start = chrono::high_resolution_clock::now();

    /* Stencil looping until the threshold is met */
    double maxdiff = numeric_limits<double>::max();
    int iterations = 0, nthreads;
    while(maxdiff >= threshold)
    {
        nthreads = omp_get_max_threads();
        maxdiff = 0;
        // #pragma omp parallel for shared(input, output, maxdiff, threshold, iterations)
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                output[i][j] = (input[i][j] + 
                                ( j > 0 ? input[i][j-1] : 0) + 
                                ( i > 0 ? input[i-1][j] : 0) + 
                                ( i < N-1 ? input[i+1][j] : 0) + 
                                ( j < N-1 ? input[i][j+1] : 0))/5;

                if(fabs(output[i][j] - input[i][j]) > maxdiff)
                    maxdiff = fabs(output[i][j] - input[i][j]);
            }
        }

        swap(input, output);
        ++iterations;
    }

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << N << ' ' << duration.count() << endl;

    return 0;
}