#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        cerr << "Size of the array is expected as an argument!" << endl;
        exit(0);
    }

    int size = atoi(argv[1]);
    ofstream fout ("array.txt");
    if(fout.is_open())
    {
        for(int i = 0; i < size; i++)
            fout << i << " " ;
        fout.close();
    }
    else
        cout << "Unable to open file" << endl;

    return 0;
}