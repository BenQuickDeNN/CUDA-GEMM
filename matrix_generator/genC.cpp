#include "CSV.cxx"

#include <cstdio>
#include <vector>
#include <fstream>

using namespace std;

int main()
{
    vector<vector<float>> A = readCSVSP("A.csv");
    vector<vector<float>> B = readCSVSP("B.csv");
    if (A.empty() || B.empty())
    {
        fprintf(stderr, "matrixs are empty!\r\n");
        return -1;
    }
    if (A[0].size() != B.size())
    {
        fprintf(stderr, "sizes of A, B are not matched!\r\n");
        return -2;
    }
    float C[A.size()][B[0].size()];
    for (int row = 0; row < A.size(); row++)
        for (int col = 0; col < B[0].size(); col++)
        {
            C[row][col] = 0.0;
            for (int i = 0; i < B.size(); i++)
                C[row][col] += A[row][i] * B[i][col];
        }
    ofstream csvfile;
    csvfile.open("C.csv");
    char buffer[32];
    for (int row = 0; row < A.size(); row++)
    {
        for (int col = 0; col < B[0].size(); col++)
        {
            gcvt(C[row][col], 8, buffer);
            if (col != B[0].size() - 1)
                csvfile<<buffer<<',';
            else
                csvfile<<buffer;
        }
        if (row != A.size() - 1)
            csvfile<<endl;
    }
    csvfile.close();
}