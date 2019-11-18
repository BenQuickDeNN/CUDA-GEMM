#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace std;

void genMatrix(const int& height, const int& width);
int main(int argc, char** argv)
{
    if (argc == 2)
        genMatrix(atoi(argv[1]), atoi(argv[1]));
    else if (argc == 3)
        genMatrix(atoi(argv[1]), atoi(argv[2]));
    else
    {
        /* error */
        fprintf(stderr, "parameter doesnot matched!\r\n");
        return -1;
    }
}
void genMatrix(const int& height, const int& width)
{
    ofstream csvfile;
    csvfile.open("A.csv");
    char buffer[32];
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            gcvt(10.0 * ((float) (rand() % 10) / 9), 8, buffer);
            if (col != width - 1)
                csvfile<<buffer<<',';
            else
                csvfile<<buffer;
        }
        if (row != height - 1)
            csvfile<<endl;
    }
    csvfile.close();
}