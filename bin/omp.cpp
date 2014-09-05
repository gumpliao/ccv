#include <getopt.h>
#include <cassert>
#include <iostream>
#include <string>
using namespace std;

#include "spams/spams.h"
#include "spams/shared.hpp"

void exit_with_help()
{
    cout << "\n  \033[1mUSAGE\033[0m\n\n    omp [OPTION...]\n\n"
    "  \033[1mREQUIRED OPTIONS\033[0m\n\n"
    "    --dict : path to the dictionary data file, created using traindl\n\n"
    "  \033[1mOTHER OPTIONS\033[0m\n\n"
    "    --L : maximum number of elements in each decomposition [DEFAULT TO min(M,K)]\n"
    "    --eps : threshold on the squared L2-norm of the residual [DEFAULT TO 0]\n"
    "    --lambda : penalty parameter [DEFAULT TO 0]\n"
    "    --numThreads : number of threads to use on enabled multi-core / multi-cpu machines [DEFAULT TO -1 (all available)]\n"
    "    --file : optional path to text file containing a list of model files [DEFAULT TO null]\n\n";
    exit(-1);
}

int main(int argc, char** argv)
{
    /* if no parameters are provided, print help to stdout */
    if (argc == 1)
        exit_with_help();

    /* structure of input parameters */
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

    /* initialise default values for optional parameters */
    int sizeL(1), sizeE(1), sizeLambda(1), L(0),
        NUM_THREADS(-1), dict_ind(0), ind(0);
    double eps(0), lambda(0);
    bool FILE_PROVIDED(false),
         LIST_PROVIDED(false),
         DICT_PROVIDED(false);

    /* parse input arguments */
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

    /* check if the required parameters were provided */
    DICT_PROVIDED = dict_ind;
    assert(DICT_PROVIDED);
    LIST_PROVIDED = argc > param_count*2+1;
    assert(FILE_PROVIDED || LIST_PROVIDED);

    /* grab model list from file (if provided) */
    int no_file_models = 0;
    char** file_models = FILE_PROVIDED? getModelsFromFile(argv[ind], no_file_models) : NULL;

    int m = 0, n = 0;

    /* grab part filters data from model files and store their paths */
    vector<string> *paths = new vector<string>();
    vector<double*> part_v = parseModelFiles(file_models, no_file_models, param_count*2+1, argc, argv, m, n, paths);

    /* convert data from vector to array (the format required by SPAMS Matrix constructor) */
    double *prX = getDataFromPtrVector(part_v, m, n);

    int K = 0;

    /* grab data from the file where the previously trained dictionary is stored */
    double* prD = getDictionaryData(argv[dict_ind], m, K);

    /* initialise input data matrix X and dictionary matrix D */
    Matrix<double> X(prX,m,n), D(prD,m,K);

    if (L == 0)
        L=MIN(m,K);

    /* convert input parameters to the format required by omp function in SPAMS */
    int *pL = &L;
    double* pE=&eps;
    double* pLambda=&lambda;

    /* initialise and compute sparse matrix of coefficient values */
    SpMatrix<double> alpha;

    OMP<double>(X, D, alpha, pL, pE, pLambda, sizeL, sizeE, sizeLambda, NUM_THREADS);

    /* convert to dense and save to file */
    Matrix<double> alphaDense;

    alpha.toFull(alphaDense);

    saveToFile(argv[dict_ind], alphaDense, paths, true);

    /* free memory */
    cleanup(prX, prD, file_models, no_file_models);

    cout << "The matrix of coefficient values (alpha) was successfully computed and added to: " << argv[dict_ind] << endl;
    return 0;
}
