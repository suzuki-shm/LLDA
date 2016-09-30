#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-02-17

import os
import ctypes
import gensim
import sys
import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

class LLDAClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                maxiter=100,
                alpha=0.1,
                beta=0.1,
                n_particle=100,
                ess=10,
                rejuvenation=10,
                threshold=0.1,
                tmp="/tmp/labeled_lda"):
        if type(rejuvenation) is not int:
            raise ValueError("rejuvenation should be integer value.")
        if type(ess) is not int:
            raise ValueError("ess(effective sample size) should be integer value.")
        if os.path.exists(tmp) is False:
            warnings.warn("{0} is not exist, make this directory".format(tmp), UserWarning)
            os.makedirs(tmp)

        self.maxiter=maxiter
        self.alpha=alpha
        self.beta=beta
        self.n_particle=n_particle
        self.ess=ess
        self.rejuvenation=rejuvenation
        self.threshold = threshold
        self.tmp = tmp
        self.program_dir = os.path.dirname(os.path.abspath(__file__))

    def fit(self, X, y):
        y = self._validate_targets(y)
        self.__class_num = y.shape[1]

        self._convert_svmlight(X, "train")
        self._convert_low(y, "train")
        llda = ctypes.CDLL(os.path.join(self.program_dir, "lib", "Labeled-LDA", "llda.so"))
        llda.calculate.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        llda.calculate(self.maxiter,
                       self.alpha,
                       self.beta,
                       os.path.join(self.tmp, "train_x.svmlight").encode("UTF-8"),
                       os.path.join(self.tmp, "train_y.low").encode("UTF-8"),
                       os.path.join(self.tmp, "fit").encode("UTF-8")
                      )

    def predict_proba(self, X):
        if (os.path.exists(os.path.join(self.tmp, "fit.lik")) == False) or\
                (os.path.exists(os.path.join(self.tmp, "fit.theta")) == False) or\
                (os.path.exists(os.path.join(self.tmp, "fit.n_mz")) == False) or\
                (os.path.exists(os.path.join(self.tmp, "fit.n_wz")) == False) or\
                (os.path.exists(os.path.join(self.tmp, "fit.phi")) == False):
            print("Anything output of L-LDA is missing")
            sys.exit(1)
        self._convert_svmlight(X, "test")
        ldapf = ctypes.CDLL(os.path.join(self.program_dir, "lib", "OnlineLDA_ParticleFilter", "ldapf.so"))
        ldapf.calculate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p]
        ldapf.calculate(self.n_particle,
                        self.ess,
                        self.rejuvenation,
                        self.alpha,
                        self.beta,
                        os.path.join(self.tmp, "test_x.svmlight").encode("UTF-8"),
                        os.path.join(self.tmp, "fit").encode("UTF-8")
                       )
        result = np.loadtxt(os.path.join(self.tmp, "test_x.svmlight.theta"))
        while result.shape[1] < self.__class_num:
            result = np.c_[result, np.zeros(result.shape[0])]
        return result

    def predict(self, X):
        probability = self.predict_proba(X)
        return self._assignment(probability)

    def get_params(self, deep=True):
        return {"maxiter":self.maxiter,
                "alpha":self.alpha,
                "beta":self.beta,
                "n_particle":self.n_particle,
                "ess":self.ess,
                "rejuvenation":self.rejuvenation,
                "threshold":self.threshold,
                "tmp":self.tmp
                }

    def set_prarams(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def _validate_targets(self, y):
        y = check_array(y)
        if y.shape[0]==1 or y.dtype!=np.int:
            raise ValueError("Label input must be sparse matrix.")
        return y

    def _assignment(self, y):
        return (y>self.threshold).astype(np.int)

    def _convert_svmlight(self, X, porpose):
        np_X = np.array(X)
        if len(np_X.shape) == 2:
            np_X = gensim.matutils.Dense2Corpus(np_X.T)
        elif len(np_X.shape) == 1:
            if not gensim.utils.is_corpus(np_X)[0]:
                print("Invalid input: X should be 2d-dense matrix or gensim corpus format")
                sys.exit(1)
        else:
            print("Invalid input: X should be 2d-dense matrix or gensim corpus format")
            sys.exit(1)
        with open(os.path.join(self.tmp, "{0}_x.svmlight".format(porpose)), "w") as f:
            for doc in np_X:
                i = 0
                for key, value in doc:
                    if i != 0:
                        f.write(" ")
                    f.write("{0}:{1}".format(str(key+1), str(int(value))))
                    i += 1 
                f.write("\n")

    def _convert_low(self, y, porpose):
        np_y= np.array(y)
        with open(os.path.join(self.tmp, "{0}_y.low".format(porpose)), "w") as f:
            for doc in np_y:
                i = 0
                for index in np.where(doc==1)[0]:
                    if i !=0:
                        f.write(" ")
                    f.write(str(index+1))
                    i += 1
                f.write("\n")
