/*
 *  Author   : S. Karthik Chandra 
 *  Roll     : CS17B026
 *  Reference: Code explained in class[CS5105].
 */

#include <iostream>
#include <thread>
#include <vector>
#include <random>

using namespace std;

int nthreads, ndata;
int *sum;

void compute_sum(int, const vector<int> &);

int main(int argc, char **argv)
{
    if(argc < 3)
    {
        cerr << "Number of threads, and size of the array are expected arguments!" << endl;
        exit(0);
    }

    nthreads = atoi(argv[1]);
    ndata = atoi(argv[2]);
    sum = new int[nthreads];

    /* Creating a vector with random data */
    uniform_real_distribution<double> dist(0, 100);
    default_random_engine re;
    vector<int> data;
    for (int i = 0; i < ndata; i++)
        data.push_back(dist(re));

    /* Creating and launching a vector of threads */
    vector<thread> threads;
    for (int i = 0; i < nthreads; i++)
        threads.push_back(thread(compute_sum, i, cref(data)));

    /* Wait for all the threads to complete */
    for (auto &th : threads)
        th.join();

    /* Compute the total sum */
    int tsum = 0;
    for (int i = 0; i < nthreads; i++)
        tsum += sum[i];

    cout << "Total sum: " << tsum << endl;

    delete[] sum;
    return 0;
}

void compute_sum(int tid, const vector<int> &d)
{
    int st = floor(ndata / nthreads) * tid;
    int en = floor(ndata / nthreads) * (tid + 1);

    /* Adding selected elements */
    for(int i = st; i < en; i++)
        sum[tid] += d[i];

    /* Adding leftover elements */
    if(((floor(ndata / nthreads) * nthreads) + tid) < ndata)
        sum[tid] += d[(floor(ndata / nthreads) * nthreads) + tid];
}