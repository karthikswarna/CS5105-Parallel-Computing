/*
 *  Author: S. Karthik Chandra
 *  Roll  : CS17B026
 */

#include <mpi.h>
#include <thread>
#include <vector>
#include <iostream>
#include <unistd.h>

using namespace std;

int buffer = 0, tcount = 0;
pthread_mutex_t lock;

void iittp_barrier(int, int, int, int);
void thread_func(int, int, int, int);

int main(int argc, char **argv)
{
    int rank, np, nthreads;

    if(argc == 2)
        nthreads = atoi(argv[1]);
    else
        nthreads = 2;

    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    /* Launching threads on thread_func function which calls a barrier */
    vector<thread> threads;
    for (int tid = 0; tid < nthreads; tid++)
        threads.push_back(thread(thread_func, rank, np, tid, nthreads));

    for (auto &th : threads)
        th.join();

    MPI_Finalize();
    return 0;
}

void thread_func(int rank, int np, int tid, int nthreads)
{
    cout << rank << " " << tid << endl;
    sleep(10);              // Sleep for sometime so show that barrier is working and printing is done

    iittp_barrier(rank, np, tid, nthreads);

    sleep(10);              // Sleep for sometime so show that barrier is working and printing is done
    cout << rank << " " << tid << endl;
}

void iittp_barrier(int rank, int np, int tid, int nthreads)
{
    /* Thread with id=0 handles the send and receive in all processes */
    if(tid == 0)
    {    
        /* MASTER PROCESS */
        if (rank == 0)
        {
            int pcount = 0;
            for (int i = 1; i < np; i++)
            {
                /* Receives from all other processes that they have reached the barrier */
                MPI_Recv(&buffer, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pcount++;
            }

            if (pcount == np - 1)
                for (int i = 1; i < np; i++)
                    /* Sends to allow processes to continue */
                    MPI_Send(&buffer, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
        }
        /* SLAVE PROCESS */
        else
        {
            MPI_Send(&buffer, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&buffer, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    /* Each thread increases the thread counter to signal that they have reached this point */
    pthread_mutex_lock(&lock);
    tcount++;
    pthread_mutex_unlock(&lock);
    /* When thread 0 increases the counter(after sending and receiving messages across processes), it equals to the nthreads */
    while (tcount != nthreads);
}