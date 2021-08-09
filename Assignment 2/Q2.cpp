#include <iostream>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;

void nQueen(char **, int, int, int);
bool isSafe(char **, int, int);

mutex m;
static int N;
int solutionCount = 0;

/* Extended thread class with own private board */
struct Thread
{
    thread worker;
    char **board;
    int tid;

    Thread(void(*func)(char**, int, int, int), int tid)
        : tid(tid)
    {
        board = new char*[N];
        for(int i = 0; i < N; i++)
        {
            board[i] = new char[N];
            memset(board[i], '-', N*sizeof(char));
        }

        worker = thread(func, board, 0, tid, tid);
    }

    void join()
    {
        this->worker.join();
    }
};

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cerr << "Integer is expected as an argument!\nFormat: ./a.out <N>" << endl;
        exit(0);
    }
    N = atoi(argv[1]);

    // auto start = chrono::high_resolution_clock::now();
  
    /* Creating a vector of threads and launching on nQueens function */
    vector<Thread> threads;
    for(int i = 0; i < (N+1)/2; i++)
        threads.push_back(Thread(nQueen, i));

    /* Joining the thread objects */
    for(auto &th: threads)
        th.join();

    // auto stop = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    // std::cout << duration.count() << std::endl;

    cout << "Number of solutions: " << solutionCount << endl;

    return 0;
}

void nQueen(char **board, int r, int c, int tid)
{
    if (isSafe(board, r, c))
    {
        if(r == N - 1)
        {
            // The number of solutions on the left side of the tree will be the same as those on the right.(Solutions are mirror images)
            // If the thread is in the middle, just increment the solution counter. If the thread is not in the middle, increment twice.
            if(N%2 == 1 && N/2 == tid)
            {
                m.lock();
                solutionCount++;
                m.unlock();
            }
            else
            {
                m.lock();
                solutionCount += 2;
                m.unlock();
            }
            return;
        }

        board[r][c] = 'Q';

        /* For the next row, place a queen in each of the columns and recur */
        for (int i = 0; i < N; i++)
            nQueen(board, r + 1, i, tid);

        board[r][c] = '-';
    }
}

bool isSafe(char **board, int r, int c)
{
    /* return false if two queens share the same column */
    for (int i = 0; i < r; i++)
        if (board[i][c] == 'Q')
            return false;

    /* return false if two queens share the same \ diagonal */
    for (int i = r, j = c; i >= 0 && j >= 0; i--, j--)
        if (board[i][j] == 'Q')
            return false;

    /* return false if two queens share the same / diagonal */
    for (int i = r, j = c; i >= 0 && j < N; i--, j++)
        if (board[i][j] == 'Q')
            return false;

    return true;
}