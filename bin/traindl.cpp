#include <getopt.h>
#include <cassert>
#include <iostream>
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
        {0, 0, 0, 0}
    };

    int K(0), iter(0), iter_updateD(1), ind(0),
        NUM_THREADS(-1), batchsize = 256*(NUM_THREADS+1);
    float lambda(0), lambda2(10e-10),
          gamma1(0), gamma2(0),
          t0(1e-5), rho(float(1.0));
    bool posAlpha(false), posD(false),
         clean(true), verbose(true),
         FILE_PROVIDED(false), LIST_PROVIDED(false);
    constraint_type mode(PENALTY);
    constraint_type_D modeD(L2);
    mode_compute modeParam(static_cast<mode_compute>(0));

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
        }
    }
    cout << "K = " << K << "\nlambda = " << lambda << "\niter = " << iter <<
    "\nmode = " << mode << "\nmodeD = " << modeD << "\nmodeParam = " << modeParam << endl;

    LIST_PROVIDED = argc > param_count*2+1;
    assert(FILE_PROVIDED || LIST_PROVIDED);
    assert(K != 0 && iter != 0 && lambda != 0 && NUM_THREADS != 0);

    ParamDictLearn<float> param;
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

    int no_file_models = 0;
    char** file_models = FILE_PROVIDED? getModelsFromFile(argv[ind], no_file_models) : NULL;

    int m = 0, n = 0;

    vector<float*> part_v = parseModelFiles(file_models, no_file_models, param_count*2+1, argc, argv, m, n, NULL);

    float *prX = getDataFromPtrVector(part_v, m, n);

    Matrix<float> X(prX,m,n), D;

    trainDL<float>(X, D, param, K, batchsize, NUM_THREADS);

    char filename[] = "sparse.dict";
    saveToFile(filename, D, NULL);

    cleanup(prX, NULL, file_models, no_file_models);

    return 0;
}
