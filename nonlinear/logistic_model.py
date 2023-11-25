#!/usr/bin/env python

"""
Author: Shohei Kojima @ RIKEN
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


class LogisticModel():
    def __init__(self):
        self.X = None
        self.Y = None
        self.popt = None
        self.pcov = None
        self.dof = None
        self.pred_ci_lower = None
        self.pred_ci_upper = None
        self.obs_ci_lower = None
        self.obs_ci_upper = None

    @staticmethod
    def logistic_func(X, a, b, c, d):
        return a / (1.0 + np.exp(-b * (X + c))) + d
    
    @staticmethod
    def logistic_func_deriv_x(X, a, b, c, d):
        # sym.diff(a / (1.0 + sym.exp(-b * (x + c))) + d, x)
        return a*b*np.exp(-b*(c + x))/(1.0 + np.exp(-b*(c + x)))**2
    
    @staticmethod
    def logistic_func_deriv_a(X, a, b, c, d):
        # sym.diff(a / (1.0 + sym.exp(-b * (x + c))) + d, a)
        return 1/(1.0 + np.exp(-b*(c + X)))
    
    @staticmethod
    def logistic_func_deriv_b(X, a, b, c, d):
        # sym.diff(a / (1.0 + sym.exp(-b * (x + c))) + d, b)
        return -a*(-c - X)*np.exp(-b*(c + X))/(1.0 + np.exp(-b*(c + X)))**2
    
    @staticmethod
    def logistic_func_deriv_c(X, a, b, c, d):
        # sym.diff(a / (1.0 + sym.exp(-b * (x + c))) + d, c)
        return a*b*np.exp(-b*(c + X))/(1.0 + np.exp(-b*(c + X)))**2
    
    @staticmethod
    def logistic_func_deriv_d(X, a, b, c, d):
        # sym.diff(a / (1.0 + sym.exp(-b * (x + c))) + d, d)
        return np.ones(X.shape)
    
    def calc_dYdP(self, X, params):
        dYda = self.logistic_func_deriv_a(X, *params)
        dYdb = self.logistic_func_deriv_b(X, *params)
        dYdc = self.logistic_func_deriv_c(X, *params)
        dYdd = self.logistic_func_deriv_d(X, *params)
        dYdP = np.hstack([dYda, dYdb, dYdc, dYdd])
        return dYdP
    
    def sigma_bse(self):
        return np.diag(self.pcov)
    
    def bse(self):
        return np.sqrt(self.sigma_bse())
    
    def Ypred(self, X = None):
        if X is None:
            X = self.X
        return self.logistic_func(X, *self.popt)
    
    def pred_ci(self, X = None, alpha = 0.05):
        cache = False
        if X is None:
            X = self.X
            cache = True
        Ypred = self.Ypred(X)
        dYdP = self.calc_dYdP(X, self.popt)
        var_Ypred = (dYdP @ self.pcov * dYdP).sum(1)
        se_Ypred = np.sqrt(var_Ypred)
        q = st.t.ppf(1 - alpha / 2, self.dof)
        if cache:
            self.pred_ci_lower = Ypred.T - q * se_Ypred
            self.pred_ci_upper = Ypred.T + q * se_Ypred
        else:
            return Ypred.T - q * se_Ypred, Ypred.T + q * se_Ypred

    def obs_ci(self, X = None, alpha = 0.05):
        cache = False
        if X is None:
            X = self.X
            cache = True
        Ypred = self.Ypred(X)
        sigmasq = np.square(self.Ypred() - self.Y).sum() / self.dof
        q = st.t.ppf(1 - alpha / 2, self.dof)
        dYdP = self.calc_dYdP(X, self.popt)
        var_Ypred = (dYdP @ self.pcov * dYdP).sum(1)
        se = np.sqrt(var_Ypred + sigmasq)
        if cache:
            self.obs_ci_lower = Ypred.T - q * se
            self.obs_ci_upper = Ypred.T + q * se
        else:
            return Ypred.T - q * se, Ypred.T + q * se
    
    def sum_sq_err(self, params):
        pred = self.logistic_func(self.X, *params)
        return ((self.Y - pred) ** 2.0).sum()
    
    def make_init_params(self, maxiter = 1000):
        # bounds for a, b, c, d
        param_bounds = []
        param_bounds.append([0.0, 1000.0])
        param_bounds.append([0.5, 5.0])
        param_bounds.append([0.0, 50.0])
        param_bounds.append([0.0, 500.0])
        res = opt.direct(
            self.sum_sq_err,
            param_bounds,
            maxiter = maxiter,
        )
        print('\nInit params:\n', res)
        return res.x
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        init_params = self.make_init_params()
        popt, pcov = opt.curve_fit(self.logistic_func, X.T[0], Y.T[0], init_params)
        print('\nModel after fit:  %.3f / (1.0 + np.exp(-1 * %.3f * (t + %.3f))) + %.3f' % tuple(popt))
        self.popt = popt
        self.pcov = pcov
        self.dof = X.shape[0] - popt.shape[0]
    
    def to_df(self):
        df = pd.DataFrame(index = range(self.X.shape[0]))
        df['X'] = self.X
        df['Y'] = self.Y
        df['Ypred'] = self.Ypred()
        if self.pred_ci_lower is None:
            self.pred_ci()
        if self.obs_ci_lower is None:
            self.obs_ci()
        df['pred_ci_lower'] = self.pred_ci_lower.T
        df['pred_ci_upper'] = self.pred_ci_upper.T
        df['obs_ci_lower'] = self.obs_ci_lower.T
        df['obs_ci_upper'] = self.obs_ci_upper.T
        return df
    
    def plot(self, outfilename, n_grid = 100):
        df = self.to_df()
        gridX = np.linspace(self.X.min(), self.X.max(), n_grid)
        gridY = self.logistic_func(gridX, *self.popt)
        pred_ci_lower, pred_ci_upper = self.pred_ci(np.array([gridX]).T)
        obs_ci_lower, obs_ci_upper = self.obs_ci(np.array([gridX]).T)
        plt.fill_between(
            gridX, pred_ci_lower[0], pred_ci_upper[0],
            alpha=.2, label='Pred_CI', fc = 'Orange',
        )
        plt.fill_between(
            gridX, obs_ci_lower[0],  obs_ci_upper[0],
            alpha=.1, label='Obs_CI', fc = 'tab:blue',
        )
        plt.plot(gridX, gridY, color = 'orange', alpha = 0.5)
        sns.scatterplot(
            data = df,
            x = 'X',
            y = 'Y',
            alpha = 0.5,
            color = 'tab:blue',
            edgecolor = None,
            linewidth = 0,
        )
        plt.savefig(outfilename)
        plt.close()


if __name__ == '__main__':
    f = 'test_data.logistic_model.tsv'
    df = pd.read_table(f).iloc[:100]
    X = np.array([df['T'].to_numpy()]).T
    Y = np.array([df['Y'].to_numpy()]).T
    model = LogisticModel()
    model.fit(X, Y)
    model.plot('plot.fit_logistic.pdf')
