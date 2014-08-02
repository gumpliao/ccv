#ifndef SHARED_HPP
#define SHARED_HPP

#include <vector>
#include <fstream>
#include <cassert>
#include <string>
using namespace std;

void discard(ifstream &f, char* temp);

char** getModelsFromFile(char* filename, int& no_file_models);

void saveToFile(char* filename, Matrix<float> &X, vector<string> *paths);

float* getDataFromPtrVector(vector<float*> v, int m, int n);

float* getDictionaryData(char* filename, int &K);

vector<float*> parseModelFiles(char** file_models, int no_file_models,
                               int start, int argc, char** argv, int &m, int &n,
                               vector<string> *paths);

void cleanup(float* prX, float* prD, char** file_models, int no_file_models);

vector<float*> parseModelFiles(char** file_models, int no_file_models,
                               int start, int argc, char** argv, int &m, int &n,
                               vector<string> *paths)
{
    vector<float*> part_v;
    bool size_set = false, file_parsed = false;
    int c = file_models? 0 : start;
    bool condition = file_models? c < no_file_models : c < argc;
    char** now_parsing = file_models? file_models : argv;
    while (condition)
    {
        ifstream f;
        f.open(now_parsing[c]);
        if (paths) paths->push_back(now_parsing[c]);

        char flag = f.get();
        assert(flag == '.');

        int comps, rows, cols, parts;
        f >> comps;
        for (int k = 0; k < comps; k++)
        {
            f >> rows;
            f >> cols;

            char* temp = new char[cols*31];
            for (int j = 0; j < rows * cols * 31 + 4; j++)
                f >> temp;
            f >> parts;
            discard(f, temp);

            f >> rows;
            f >> cols;
            assert(rows == cols);
            f >> temp;

            int weights_per_part = rows*cols*31;
            if (!size_set)
            {
                m = weights_per_part;
                size_set = true;
            }
            else
                assert(weights_per_part == m);

            for (int j = 0; j < parts; j++)
            {
                float* part_filter = new float[weights_per_part];

                for (int i = 0; i < weights_per_part; i++)
                {
                    f >> temp;
                    part_filter[i] = atof(temp);
                }
                if (j < parts-1)
                {
                    discard(f, temp);
                    f >> temp;
                    assert(atoi(temp) == rows);
                    f >> temp;
                    assert(atoi(temp) == cols);
                    f >> temp;
                }
                part_v.push_back(part_filter);
            }
            delete[] temp;
        }
        f.close();
        if (file_models && !file_parsed)
            if (c == no_file_models-1)
            {
                file_parsed = true;
                c = start-1;
                now_parsing = argv;
            }
        condition = file_models? (file_parsed? ++c < argc : ++c < no_file_models) : ++c < argc;
    }
    n = part_v.size();
    assert(m > 0 && n > 0);

    return part_v;
}

char** getModelsFromFile(char* filename, int& no_file_models)
{
    ifstream f1;
    f1.open(filename);
    string temp;

    int maxLen = 0;
    while (f1 >> temp)
    {
        no_file_models++;
        if ((int)temp.length() > maxLen)
            maxLen = temp.length();
    }
    f1.close();

    ifstream f2;
    f2.open(filename);
    char** models = new char*[no_file_models];
    for (int i = 0; i < no_file_models; i++)
    {
        models[i] = new char[maxLen];
        f2 >> models[i];
    }

    return no_file_models? models : NULL;
}

void saveToFile(char* filename, Matrix<float> &X, vector<string> *paths)
{
    ofstream o;
    o.open(filename);
    int cols = X.m(), rows = X.n();

    if (paths)
    {
        o << paths->size() << "\n";
        for (unsigned int i = 0; i < paths->size(); i++)
            o << paths->at(i) << "\n";
    }
    else
        o << sqrt(cols/31) << " " << sqrt(cols/31) << " ";

    o << rows;
    if (paths)
        o << " " << cols;
    o << "\n";
    for (int j = 0; j < rows; j++)
        for (int i = 0; i < cols; i++)
            o << X(i,j) << (i < (cols-1)? " " : "\n");
    o.close();

    delete paths;
}

float* getDataFromPtrVector(vector<float*> v, int m, int n)
{
    float* prX = new float[m*n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            prX[i*m+j] = v[i][j];

    for (unsigned int i = 0; i < v.size(); i++)
        delete[] v[i];

    return prX;
}

float* getDictionaryData(char* filename, int m, int &K)
{
    vector<float*> dict_v;
    ifstream f;
    f.open(filename);
    /* get rid of first 3 numbers */
    int temp;
    f >> temp; f >> temp; f >> temp;
    while (!f.fail() && !f.eof())
    {
        float* dict_filter = new float[m];
        int i = 0;
        for (; i < m; i++)
            if (!(f >> dict_filter[i]))
                break;
        if (i == m)
            dict_v.push_back(dict_filter);
    }
    f.close();
    K = dict_v.size();
    assert(K > 0);

    return getDataFromPtrVector(dict_v, m, K);
}

void discard(ifstream &f, char* temp)
{
    for (int i = 0; i < 13; i++)
        f >> temp;
}

void cleanup(float* prX, float* prD, char** file_models, int no_file_models)
{
    delete[] prX;
    delete[] prD;
    for (int i = 0; i < no_file_models; i++)
        delete[] file_models[i];
}

#endif
