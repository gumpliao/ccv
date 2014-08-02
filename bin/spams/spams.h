
/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SPAMS_H
#define SPAMS_H

#include "dicts.h"
#include "decomp.h"

template <typename T>
   inline void trainDL(Matrix<T> &X, Matrix<T> &D, ParamDictLearn<T> param,
                       int K, int batchsize, int NUM_THREADS);

template <typename T>
   inline void OMP(Matrix<T> &X, Matrix<T> &D, SpMatrix<T> &alpha,
                   int* pL, T* pE, T* pLambda, int sizeL, int sizeE, int sizeLambda, int NUM_THREADS);

template <typename T>
   inline void trainDL(Matrix<T> &X, Matrix<T> &D, ParamDictLearn<T> param,
                       int K, int batchsize, int NUM_THREADS) {
#ifdef _OPENMP
      NUM_THREADS = NUM_THREADS == -1 ? omp_get_num_procs() : NUM_THREADS;
#else
      NUM_THREADS=1;
#endif
      Trainer<T> trainer(K,batchsize,NUM_THREADS);

      trainer.train(X,param);

      trainer.getD(D);
   }

template <typename T>
   inline void OMP(Matrix<T> &X, Matrix<T> &D, SpMatrix<T> &alpha,
                   int* pL, T* pE, T* pLambda, int sizeL, int sizeE, int sizeLambda, int NUM_THREADS) {
#ifdef _OPENMP
      NUM_THREADS = NUM_THREADS == -1 ? omp_get_num_procs() : NUM_THREADS;
#else
      NUM_THREADS=1;
#endif
      Matrix<T>* prPath=NULL;

      omp<T>(X,D,alpha,pL,pE,pLambda,sizeL > 1,sizeE > 1,sizeLambda > 1,
            NUM_THREADS,prPath);
   }

#endif
