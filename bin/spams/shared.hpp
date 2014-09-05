#ifndef SHARED_HPP
#define SHARED_HPP

#include <vector>
#include <fstream>
#include <cassert>
#include <string>
using namespace std;

/*** DECLARATIONS ***/

void discard(ifstream &f, char* temp);

char** getModelsFromFile(char* filename, int& no_file_models);

void saveToFile(char* filename, Matrix<double> &X, vector<string> *paths, bool append);

double* getDataFromPtrVector(vector<double*> v, int m, int n);

double* getDictionaryData(char* filename, int &K);

vector<double*> parseModelFiles(char** file_models, int no_file_models,
                               int start, int argc, char** argv, int &m, int &n,
                               vector<string> *paths);

void cleanup(double* prX, double* prD, char** file_models, int no_file_models);

/*** DEFINITIONS ***/

/* grab part filters data from model files */
vector<double*> parseModelFiles(char** file_models, int no_file_models,
                               int start, int argc, char** argv, int &m, int &n,
                               vector<string> *paths)
{
    vector<double*> part_v;
    bool size_set = false, file_parsed = false;
    int c = file_models? 0 : start;
    /* set the loop condition to parse models from 2 sources:
       1) list of model files provided through input file and stored into file_models;
       2) paths directly listed as command-line arguments
    */
    bool condition = file_models? c < no_file_models : c < argc;
    char** now_parsing = file_models? file_models : argv;
    while (condition)
    {
        ifstream f;
        f.open(now_parsing[c]);
        /* if we want to store model paths */
        if (paths) paths->push_back(now_parsing[c]);

        /* check the "final model" flag */
        char flag = f.get();
        assert(flag == '.');

        int comps, rows, cols, parts;
        f >> comps;
        for (int k = 0; k < comps; k++)
        {
            /* root info (read and ignored) */
            f >> rows;
            f >> cols;

            char* temp = new char[cols*31];
            for (int j = 0; j < rows * cols * 31 + 4; j++)
                f >> temp;
            f >> parts;
            discard(f, temp);

            /* part info (used and stored) */
            f >> rows;
            f >> cols;
            /* must be square filter */
            assert(rows == cols);
            f >> temp;

            int weights_per_part = rows*cols*31;
            /* set or check feature size */
            if (!size_set)
            {
                m = weights_per_part;
                size_set = true;
            }
            else
                assert(weights_per_part == m);

            for (int j = 0; j < parts; j++)
            {
                double* part_filter = new double[weights_per_part];

                /* store part weights */
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
        /* if we have parsed completely one source, change the loop condition and parse the other (if provided) */
        if (file_models && !file_parsed)
            if (c == no_file_models-1)
            {
                file_parsed = true;
                c = start-1;
                now_parsing = argv;
            }
        condition = file_models? (file_parsed? ++c < argc : ++c < no_file_models) : ++c < argc;
    }
    n = part_v.size(); /* total number of part filters */
    assert(m > 0 && n > 0);

    return part_v;
}

/* grab model list from file */
char** getModelsFromFile(char* filename, int& no_file_models)
{
    ifstream f1;
    f1.open(filename);
    string temp;

    int maxLen = 0;
    /* set max path length as the longest path among those provided */
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

/* save a data matrix to file */
void saveToFile(char* filename, Matrix<double> &X, vector<string> *paths, bool append)
{
    ofstream o;
    /* set open mode */
    if (append)
        o.open(filename, ios_base::app);
    else
        o.open(filename);
    int cols = X.m(), rows = X.n();

    /* write file paths (if provided) */
    if (paths)
    {
        o << paths->size() << "\n";
        for (unsigned int i = 0; i < paths->size(); i++)
            o << paths->at(i) << "\n";
    }
    else /* otherwise, write filter size */
        o << sqrt(cols/31) << " " << sqrt(cols/31) << " ";

    /* write number of rows - i.e. dictionary size (K) if the input matrix is the learnt dictionary */
    o << rows;
    /* if computing the alpha matrix, write also the number of columns */
    if (paths)
        o << " " << cols;
    o << "\n";
    /* write the matrix data */
    for (int j = 0; j < rows; j++)
        for (int i = 0; i < cols; i++)
            o << X(i,j) << (i < (cols-1)? " " : "\n");
    o.close();

    delete paths;
}

/* convert from vector<double*> to double* array (the format required by SPAMS Matrix constructor) */
double* getDataFromPtrVector(vector<double*> v, int m, int n)
{
    double* prX = new double[m*n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            prX[i*m+j] = v[i][j];

    for (unsigned int i = 0; i < v.size(); i++)
        delete[] v[i];

    return prX;
}

/* grab data from the dictionary file */
double* getDictionaryData(char* filename, int m, int &K)
{
    vector<double*> dict_v;
    ifstream f;
    f.open(filename);
    /* get rid of first 3 numbers */
    int temp;
    f >> temp; f >> temp; f >> temp;
    /* store sparse (dict) filters into temporary vector<double*> */
    while (!f.fail() && !f.eof())
    {
        double* dict_filter = new double[m];
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

    /* employ the getDataFromPtrVector to convert vector<double*> to double* array */
    return getDataFromPtrVector(dict_v, m, K);
}

/* used in the context of parseModelFiles */
void discard(ifstream &f, char* temp)
{
    for (int i = 0; i < 13; i++)
        f >> temp;
}

/* free memory */
void cleanup(double* prX, double* prD, char** file_models, int no_file_models)
{
    delete[] prX;
    delete[] prD;
    for (int i = 0; i < no_file_models; i++)
        delete[] file_models[i];
}

#endif
