/*
 *  Author   : S. Karthik Chandra 
 *  Roll     : CS17B026
 */


#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cerr << "Size of the array is expected as argument!\nFormat: ./a.out <size>" << endl;
        exit(0);
    }

    int length = atoi(argv[1]);
    int *array = new(nothrow) int[length];
    if(!array)
    {
        cout << "Memory allocation for array failed" << endl;
        exit(0);
    }

    /* Randomly generate an array */
    uniform_real_distribution<double> dist(0, 10);
    default_random_engine re;
    for(int i = 0; i < length; i++)
        array[i] = dist(re);

    int total_sum = 0;
    /* Sum calculation using OpenMP reduction */
    #pragma omp parallel for reduction(+: total_sum)
        for(int i = 0; i < length; i++)
            total_sum += array[i];

    cout << "Total Sum: " << total_sum << endl;
    delete[] array;

    return 0;
}