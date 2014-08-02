#include <getopt.h>
#include <cassert>
#include <iostream>
#include <string>
using namespace std;

#include "spams/spams.h"
#include "spams/shared.hpp"

void exit_with_help()
{
    cout << "TODO" << endl;
    exit(-1);
}

int main(int argc, char** argv)
{
    static struct option param_options[] = {
        /* help */
        {"help", 0, 0, 0},
        /* required parameters */
        {"dict", 1, 0, 0},
        /* optional parameters */
        {"L", 1, 0, 0},
        {"eps", 1, 0, 0},
        {"lambda", 1, 0, 0},
        {"numThreads", 1, 0, 0},
        {"file", 1, 0, 0},
        {0, 0, 0, 0}
    };

    int sizeL(1), sizeE(1), sizeLambda(1), L(0),
        NUM_THREADS(-1), dict_ind(0), ind(0);
    float eps(0), lambda(0);
    bool FILE_PROVIDED(false),
         LIST_PROVIDED(false),
         DICT_PROVIDED(false);

    int p(0), param_count(0);
    while (getopt_long_only(argc, argv, "", param_options, &p) != -1)
    {
        switch (p)
        {
            case 0:
                exit_with_help();
            case 1:
                dict_ind = optind-1;
                param_count++;
                break;
            case 2:
                L = atoi(optarg);
                param_count++;
                break;
            case 3:
                eps = atof(optarg);
                param_count++;
                break;
            case 4:
                lambda = atof(optarg);
                param_count++;
                break;
            case 5:
                NUM_THREADS = atoi(optarg);
                param_count++;
                break;
            case 6:
                FILE_PROVIDED = true;
                ind = optind-1;
                param_count++;
                break;
        }
    }
    cout << "L = " << L << "\neps = " << eps << "\nlambda = " << lambda << endl;

    DICT_PROVIDED = dict_ind;
    assert(DICT_PROVIDED);
    LIST_PROVIDED = argc > param_count*2+1;
    assert(FILE_PROVIDED || LIST_PROVIDED);

    int no_file_models = 0;
    char** file_models = FILE_PROVIDED? getModelsFromFile(argv[ind], no_file_models) : NULL;

    int m = 0, n = 0;

    vector<string> *paths = new vector<string>();
    vector<float*> part_v = parseModelFiles(file_models, no_file_models, param_count*2+1, argc, argv, m, n, paths);

    float *prX = getDataFromPtrVector(part_v, m, n);

    int K = 0;

    float* prD = getDictionaryData(argv[dict_ind], m, K);

    Matrix<float> X(prX,m,n), D(prD,m,K);

    if (L == 0)
        L=MIN(n,K);

    int *pL = &L;
    float* pE=&eps;
    float* pLambda=&lambda;

    SpMatrix<float> alpha;

    OMP<float>(X, D, alpha, pL, pE, pLambda, sizeL, sizeE, sizeLambda, NUM_THREADS);

    Matrix<float> alphaDense;

    alpha.toFull(alphaDense);

    char filename[] = "alpha.dict";
    saveToFile(filename, alphaDense, paths);

    cleanup(prX, prD, file_models, no_file_models);

    return 0;
}
