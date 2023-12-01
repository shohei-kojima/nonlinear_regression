#!/usr/bin/env python

"""
Author: Shohei Kojima @ RIKEN
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.stats as st
from numba import jit


class BetaApproximation():
    def __init__(self):
        self.X = None
        self.Y = None
        self.n_var = None
        self.n_permut = None
        self.reshaped_X = None
        self.actual_dof = None
        self.nominal_p = None
        self.min_ps = None
        self.a = None
        self.b = None
        self.empirical_p = None
    
    def reshape_X(self):
        Xs = []
        invXTXs = []
        invXTXaXTs = []
        for i in range(self.n_var):
            X = self.X[:,i:i+1]
            invXTX = np.linalg.inv(X.T @ X)
            invXTXaXT = invXTX @ X.T
            Xs.append(X)
            invXTXs.append(invXTX)
            invXTXaXTs.append(invXTXaXT)
        t = (np.array(Xs), np.squeeze(np.array(invXTXs)), np.array(invXTXaXTs))
        self.reshaped_X = t
    
    @staticmethod
    @jit('f8[:](i8,i8,f8[:,:,:],f8[:],f8[:,:,:],f8[:,:],i8,i8)', nopython = True)
    def ols_core(n_var, n_permut, X, invXTX, invXTXaXT, Y, dof, n_chunk):
        # X.shape = (n_var, n_indiv, 1)
        # invXTX.shape = (n_var)
        # invXTXaXT.shape = (n_var, 1, n_indiv)
        F = np.zeros((n_chunk + 1, n_permut))
        for i in range(n_var):
            mod = i % n_chunk
            popt = invXTXaXT[i] @ Y
            sigmasq = np.square(X[i] @ popt - Y).sum(0) / dof
            F[mod+1,:] = np.square(popt[0,:] / np.sqrt(sigmasq * invXTX[i]))
            if mod == 0:
                for j in range(n_permut):
                    F[0,j] = np.max(F[:,j])
        for j in range(n_permut):
            F[0,j] = np.max(F[:,j])
        return F[0]
    
    # Func specific for Y ~ X without intercept,
    # where X.shape = (n_var, n_indiv, 1); Y.shape = (n_indiv, n_permut).
    # Returns minimum P across variants.
    def ols_f(self, Y, n_chunk = 5000):
        '''
        Func ols_core() is the same as:
            popt = np.tensordot(invXTXaXT, Y, axes = (2, 0))  # shape (n_var, 1, n_permut)
            sigmasq = np.square(X @ popt - Y).sum(1) / self.actual_dof  # shape (n_var, n_permut)
            pcov = np.array(sigmasq).T * np.squeeze(invXTX)  # shape (n_permut, n_var)
            F = np.max((popt / np.sqrt(pcov.T.reshape(*popt.shape))) ** 2, axis = 0)
        but can process with lower memory
        '''
        F = self.ols_core(self.n_var, Y.shape[1], *self.reshaped_X, Y, self.actual_dof, n_chunk)
        P = st.f.sf(F, 1, self.actual_dof)
        return P
    
    def residual(self, C, Y):
        C = np.hstack((np.ones((C.shape[0], 1)).reshape(-1, 1), C))
        invXTX = np.linalg.inv(C.T @ C)
        popt = invXTX @ C.T @ Y
        return Y - C @ popt
    
    def fit_beta(self):
        '''
        Negative log-likelihood is:
            (1.0-a)*np.sum(np.log(x)) + (1.0-b)*np.sum(np.log(1.0-x)) + len(x)*logbeta,
            where logbeta = loggamma(a) + loggamma(b) - loggamma(a+b).
        See: https://github.com/broadinstitute/tensorqtl/blob/9857a3c6b15d2e2eed40d9aefd1af4c678b21edd/tensorqtl/core.py#L332
        '''
        # prior of parameters (a, b) will be calculated internally in the fit() func.
        res = st.beta.fit(self.min_ps, floc = 0, fscale = 1, method = 'MLE')
        self.a = res[0]
        self.b = res[1]
    
    def calc_empirical_p(self):
        self.empirical_p = st.beta.cdf(self.nominal_p, self.a, self.b)
    
    def calc_nominal_p(self):
        self.nominal_p = self.ols_f(self.Y)[0]
    
    def calc_permut_p(self):
        index = np.arange(self.Y.shape[0])
        permut_Ys = []
        for _ in range(self.n_permut):
            permut_Y = self.Y[np.random.permutation(index)]
            permut_Ys.append(permut_Y)
        permut_Ys = np.hstack(permut_Ys)
        self.min_ps = self.ols_f(permut_Ys)
    
    def print_result(self):
        text = ''
        text += 'n_var = %d, n_indiv = %d\n' % (self.n_var, self.Y.shape[0])
        text += 'Params: a = %.4f, b = %.4f\n' % (self.a, self.b)
        text += 'Nominal   P: %.10f (non-corrected)\n' % self.nominal_p
        text += 'Nominal   P: %.10f (Bonferroni corrected)\n' % (min(self.nominal_p * self.n_var, 1.0))
        text += 'Empirical P: %.10f (Beta approximation)\n' % self.empirical_p
        print(text)
    
    # X: Genotypes, ranging from 0 to 2; shape = (n_indiv, n_var)
    # Y: Phenotype, shape = (n_indiv, 1)
    # C: Covariates, shape = (n_indiv, n_cov)
    def beta_approximation(self, X, Y, C, n_permut = 10000):
        self.n_permut = n_permut
        X = st.zscore(X, axis = 0)
        Y = st.zscore(self.residual(C, Y))
        self.X = X
        self.Y = Y
        self.n_var = self.X.shape[1]
        self.actual_dof = X.shape[0] - C.shape[1] - 1
        self.reshape_X()
        self.calc_nominal_p()
        self.calc_permut_p()
        self.fit_beta()
        self.calc_empirical_p()



if __name__ == '__main__':
    # 11000 SNVs, AF = 0.5, n_sample = 40
    X = []
    for _ in range(10000):
        X.append([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ] * 3)
    for _ in range(10000):
        X.append([
            0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0,
            1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0,
            0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0,
            1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0,
        ] * 3)
    X = np.array(X).T
    
    # phenotype
    # genotype 0: np.random.normal(30, 5, 10)
    # genotype 1: np.random.normal(35, 5, 20)
    # genotype 2: np.random.normal(40, 5, 10)
    Y = [
        32.14112632, 32.28103633, 21.5878028 , 23.74998249, 22.55591413,
        29.99099136, 30.09312012, 28.90669731, 35.12866656, 21.1896814 ,
        40.5692329 , 39.57345167, 34.02592732, 35.6348316 , 42.53487115,
        38.79425237, 37.56578456, 39.75582345, 45.58662454, 36.71478966,
        27.57432544, 38.53914044, 41.74004083, 35.71251688, 42.86528668,
        38.67088324, 33.84130313, 32.0166378 , 35.56389366, 44.34117487,
        43.95291086, 40.36335334, 31.26123478, 32.83810509, 40.30713615,
        37.40358288, 41.94864552, 46.45253714, 29.86020523, 44.65659903,
    ] * 3
    Y = np.array(Y).reshape(-1, 1)

    C = np.random.normal(0, 0.1, X.shape[0]).reshape(-1, 1)

    model = BetaApproximation()
    model.beta_approximation(X, Y, C)
    model.print_result()
