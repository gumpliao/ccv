#include <getopt.h>
#include <cassert>
#include <iostream>
using namespace std;

#include "spams/spams.h"
#include "spams/shared.hpp"

void exit_with_help()
{
    cout << "\n  \033[1mUSAGE\033[0m\n\n    traindl [OPTION...]\n\n"
    "  \033[1mREQUIRED OPTIONS\033[0m\n\n"
    "    --K : dictionary size\n"
    "    --lambda : sparsity level\n"
    "    --iter : number of iterations\n\n"
    "  \033[1mOTHER OPTIONS\033[0m\n\n"
    "    --lambda2 : optional parameter [DEFAULT TO 10e-10]\n"
    "    --mode : defines the dictionary learning problem; in range [0-6] [DEFAULT TO 2]\n"
    "    --posAlpha : adds positivity constraints on the coefficients; incompatible with mode = 3; 4 [DEFAULT TO false]\n"
    "    --modeD : constraint type for the dictionary; in range [0-3] [DEFAULT TO 0]\n"
    "    --posD : adds positivity constraints on the dictionary; incompatible with modeD = 2 [DEFAULT TO false]\n"
    "    --gamma1 : parameter for modeD >= 1 [DEFAULT TO 0]\n"
    "    --gamma2 : parameter for modeD = 1 [DEFAULT TO 0]\n"
    "    --numThreads : number of threads to use on enabled multi-core / multi-cpu machines [DEFAULT TO -1 (all available)]\n"
    "    --batchsize : minibatch size [DEFAULT TO 256*(numThreads+1)]\n"
    "    --iter updateD : number of BCD iterations for the dictionary update step [DEFAULT TO 1]\n"
    "    --modeParam : optimisation mode; in range [0-3] [DEFAULT TO 0]\n"
    "    --rho : optional tuning parameter for modeParam >= 1 [DEFAULT TO 1.0]\n"
    "    --t0 : optional tuning parameter for modeParam >= 1 [DEFAULT TO 1e-5]\n"
    "    --clean : cleans unused elements from dictionary [DEFAULT TO true]\n"
    "    --verbose : displays more information during execution [DEFAULT TO true]\n"
    "    --file : optional path to text file containing a list of model files [DEFAULT TO null]\n"
    "    --outfile : optional output file to store the computed dictionary [DEFAULT TO sparse.dict]\n\n";
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
        {"K", 1, 0, 0},
        {"lambda", 1, 0, 0},
        {"iter", 1, 0, 0},
        /* optional parameters */
        {"lambda2", 1, 0, 0},
        {"mode", 1, 0, 0},
        {"posAlpha", 1, 0, 0},
        {"modeD", 1, 0, 0},
        {"posD", 1, 0, 0},
        {"gamma1", 1, 0, 0},
        {"gamma2", 1, 0, 0},
        {"batchsize", 1, 0, 0},
        {"iter_updateD", 1, 0, 0},
        {"modeParam", 1, 0, 0},
        {"rho", 1, 0, 0},
        {"t0", 1, 0, 0},
        {"clean", 1, 0, 0},
        {"verbose", 1, 0, 0},
        {"numThreads", 1, 0, 0},
        {"file", 1, 0, 0},
        {"outfile", 1, 0, 0},
        {0, 0, 0, 0}
    };

    /* initialise default values for optional parameters */
    int K(0), iter(0), iter_updateD(1), ind(0),
        NUM_THREADS(-1), batchsize = 256*(NUM_THREADS+1);
    double lambda(0), lambda2(10e-10),
          gamma1(0), gamma2(0),
          t0(1e-5), rho(double(1.0));
    bool posAlpha(false), posD(false),
         clean(true), verbose(true),
         FILE_PROVIDED(false), LIST_PROVIDED(false);
    constraint_type mode(PENALTY);
    constraint_type_D modeD(L2);
    mode_compute modeParam(static_cast<mode_compute>(0));
    char* outfile = 0;

    /* parse input arguments */
    int p(0), param_count(0);
    while (getopt_long_only(argc, argv, "", param_options, &p) != -1)
    {
        switch (p)
        {
            case 0:
                exit_with_help();
            case 1:
                K = atoi(optarg);
                param_count++;
                break;
            case 2:
                lambda = atof(optarg);
                param_count++;
                break;
            case 3:
                iter = atoi(optarg);
                param_count++;
                break;
            case 4:
                lambda2 = atof(optarg);
                param_count++;
                break;
            case 5:
                mode = static_cast<constraint_type>(atoi(optarg));
                param_count++;
                break;
            case 6:
                posAlpha = !!atoi(optarg);
                param_count++;
                break;
            case 7:
                modeD = static_cast<constraint_type_D>(atoi(optarg));
                param_count++;
                break;
            case 8:
                posD = !!atoi(optarg);
                param_count++;
                break;
            case 9:
                gamma1 = atof(optarg);
                param_count++;
                break;
            case 10:
                gamma2 = atof(optarg);
                param_count++;
                break;
            case 11:
                batchsize = atoi(optarg);
                param_count++;
                break;
            case 12:
                iter_updateD = atoi(optarg);
                param_count++;
                break;
            case 13:
                modeParam = static_cast<mode_compute>(atoi(optarg));
                param_count++;
                break;
            case 14:
                rho = atof(optarg);
                param_count++;
                break;
            case 15:
                t0 = atof(optarg);
                param_count++;
                break;
            case 16:
                clean = !!atoi(optarg);
                param_count++;
                break;
            case 17:
                verbose = !!atoi(optarg);
                param_count++;
                break;
            case 18:
                NUM_THREADS = atoi(optarg);
                param_count++;
                break;
            case 19:
                FILE_PROVIDED = true;
                ind = optind-1;
                param_count++;
                break;
            case 20:
                outfile = optarg;
                param_count++;
                break;
        }
    }

    /* check if the required parameters were provided */
    LIST_PROVIDED = argc > param_count*2+1;
    assert(FILE_PROVIDED || LIST_PROVIDED);
    assert(K != 0 && iter != 0 && lambda != 0 && NUM_THREADS != 0);

    /* initialise param structure for dictionary learning */
    ParamDictLearn<double> param;
    param.lambda = lambda;
    param.lambda2 = lambda2;
    param.iter = iter;
    param.t0 = t0;
    param.mode = mode;
    param.posAlpha = posAlpha;
    param.posD = posD;
    param.expand = false;
    param.modeD = modeD;
    param.whiten = false;
    param.clean = clean;
    param.verbose = verbose;
    param.gamma1 = gamma1;
    param.gamma2 = gamma2;
    param.rho = rho;
    param.stochastic = false;
    param.modeParam = modeParam;
    param.batch = false;
    param.iter_updateD = iter_updateD;
    param.log = false;

    /* grab model list from file (if provided) */
    int no_file_models = 0;
    char** file_models = FILE_PROVIDED? getModelsFromFile(argv[ind], no_file_models) : NULL;

    int m = 0, n = 0;

    /* grab part filters data from model files */
    vector<double*> part_v = parseModelFiles(file_models, no_file_models, param_count*2+1, argc, argv, m, n, NULL);

    /* convert data from vector to array (the format required by SPAMS Matrix constructor) */
    double *prX = getDataFromPtrVector(part_v, m, n);

    /* construct Matrix of input data X */
    Matrix<double> X(prX,m,n), D;

    /* train dictionary on input data X with the provided parameters stored in param */
    trainDL<double>(X, D, param, K, batchsize, NUM_THREADS);

    /* save results to specified filename or default one */
    if (outfile)
        saveToFile(outfile, D, NULL, false);
    else
    {
        char filename[] = "sparse.dict";
        saveToFile(filename, D, NULL, false);
    }

    /* free memory */
    cleanup(prX, NULL, file_models, no_file_models);

    cout << "The dictionary was successfully trained and saved to: " << (outfile? outfile : "sparse.dict") << endl;
    return 0;
}
