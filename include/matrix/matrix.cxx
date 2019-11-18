#include <cstdio>
#include <cstdlib>

namespace gemm
{
    class Matrix
    {
    protected:
        unsigned int _height;
        unsigned int _width;
    };
    class MatrixSP : public Matrix
    {
    public:
        float* _element;
        /**
         * @brief matrix mul C = A * B
         * @param C result
         * @param A matrix A
         * @pramm B matrix B
        */
        static void mul(MatrixSP& C, MatrixSP& A, MatrixSP& B);
        /**
         * @brief check if this matrix is empty
         * @return true->empty, false->not empty
        */
        bool empty();
        /**
         * @brief get height of this matrix
         * @return height
        */
        unsigned int height();
        /**
         * @brief get width of this matrix
         * @return width
        */
        unsigned int width();
        /**
         * @brief get element
         * @return element pointer
        */
        float* element();
        /**
         * @brief parentheses overload
         * @param row row index of matrix
         * @param col column index of matrix
         * @return element in row & column
        */
        float& operator()(const unsigned int& row, const unsigned int& col);
        /**
         * @brief constructor - memory allocation
        */
        MatrixSP(const unsigned int& _height, const unsigned int& _width);
        /**
         * @brief deconstructor - memory free
        */
        ~MatrixSP();
    };
};
void gemm::MatrixSP::mul(MatrixSP& C, MatrixSP& A, MatrixSP& B)
{
    if (A.empty() || B.empty() || C.empty())
    {
        std::fprintf(stderr, "mul error, matrix is empty!\r\n");
        return;
    }
    if (A.width() != B.height() || A.height() != C.height() || B.width() != C.width())
    {
        std::fprintf(stderr, "mul error, matrix doesnot match!\r\n");
        return;
    }
    for (unsigned int row = 0; row < C.height(); row++)
        for (unsigned int col = 0; col < C.width(); col++)
        {
            C(row, col) = 0.0;
            for (unsigned int i = 0; i < A.width(); i++)
                C(row, col) += A(row, i) * B(i, col);
        }
}
bool gemm::MatrixSP::empty()
{
    return _element == NULL;
}
unsigned int gemm::MatrixSP::height()
{
    return _height;
}
unsigned int gemm::MatrixSP::width()
{
    return _width;
}
float* gemm::MatrixSP::element()
{
    return  _element;
}
float& gemm::MatrixSP::operator()(const unsigned int& row, const unsigned int& col)
{
    return _element[row * _width + col];
}
gemm::MatrixSP::MatrixSP(const unsigned int& _height, const unsigned int& _width)
{
    this->_height = _height;
    this->_width = _width;
    /* allocate memory */
    _element = (float*) std::malloc(sizeof(float) * _height * _width);
    if (_element == NULL)
        std::fprintf(stderr, "memory allocation fail!\r\n");
}
gemm::MatrixSP::~MatrixSP()
{
    /* free memory */
    if (_element != NULL)
        std::free(_element);
}