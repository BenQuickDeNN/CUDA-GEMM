
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

std::vector<std::vector<float>> readCSVSP(const std::string filename)
{
    std::vector<std::vector<float>> strArray;
    std::ifstream inFile(filename.c_str(), std::ios::in);
    if (inFile.fail())
    {
        std::fprintf(stderr, "cannot open file %s\r\n", filename.c_str());
        return strArray;
    }
    std::string lineStr;
    while (std::getline(inFile, lineStr))  
    {
        // 存成二维表结构  
        std::stringstream ss(lineStr);  
        std::string str;  
        std::vector<float> lineArray;  
        // 按照逗号分隔  
        while (std::getline(ss, str, ','))  
            lineArray.push_back(std::atof(str.c_str()));
        strArray.push_back(lineArray);  
    }
    inFile.close();
    return strArray;
}