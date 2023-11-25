#!/usr/bin/env python

"""
Author: Shohei Kojima @ RIKEN
"""

import numpy as np
import scipy.stats as st


# just a memo
def lm_statsmodels(X, Y):
    import statsmodels.api as sm
    res = sm.OLS(Y, X).fit()
    pred = res.get_prediction().summary_frame(0.05)
    print(res.summary())
    print(pred)

# just a memo
def lm_linalg(X, Y):
    '''
    #### when X.shape[1] == 2 ####
    yhat = a + b * X
    N = len(X) = len(Y)
    least_square E = np.sum((Y - yhat) ** 2)
      = np.sum((Y - (a + b * X)) ** 2)
      = np.sum(Y ** 2 + (a + b * X) ** 2 - 2 * Y * (a + b * X))
      = np.sum(Y ** 2 + a ** 2 * N + (b * X) ** 2 + 2 * a * b * X - 2 * a * Y - 2 * b * X * Y)
    This expression can be differentiated into simultaneous equation below
    dE / da = np.sum(2 * a * N + 2 * b * X - 2 * Y) = 0
      = 2 * np.sum(a * N + b * X - Y) = 0
    dE / db = np.sum(2 * X ** 2 * b + 2 * a * X - 2 X * Y) = 0
      = 2 * np.sum(X ** 2 * b + a * X - X * Y) = 0
    '''
    X = X[:,1]
    N = X.shape[0]
    sumX = X.sum()
    sumY = Y.sum()
    sumsqX = np.square(X).sum()
    XY = (X @ Y)[0]
    # solve simultaneous equation by numpy
    x = np.array([[N, sumX], [sumX, sumsqX]])
    y = np.array([sumY, XY])
    popt = np.linalg.solve(x, y)
    print(popt)
    '''
    # alternatively, can use sympy
    import sympy as sym
    a, b = sym.symbols('a, b')
    eq1 = sym.Eq(0, a * N + b * sumX - sumY)
    eq2 = sym.Eq(0, a * sumX + b * sumsqX - XY)
    print(sym.solve([eq1, eq2]))
    '''


class OLS():
    def __init__(self):
        self.X = None
        self.Y = None
        self.popt = None
        self.pcov = None
        self.dof = None
    
    def pred_ci(self, X = None, alpha = 0.05):
        if X is None:
            X = self.X
        Ypred = (X @ self.popt).T
        var_Ypred = (X @ self.pcov * X).sum(1)
        se_Ypred = np.sqrt(var_Ypred)
        q = st.t.ppf(1 - alpha / 2, self.dof)
        ci_lower = Ypred - q * se_Ypred
        ci_upper = Ypred + q * se_Ypred
        return ci_lower, ci_upper

    def obs_ci(self, X = None, alpha = 0.05):
        if X is None:
            X = self.X
        Ypred = (X @ self.popt).T
        q = st.t.ppf(1 - alpha / 2, self.dof)
        var_Ypred = (X @ self.pcov * X).sum(1)
        sigmasq = np.square(self.X @ self.popt - self.Y).sum() / self.dof
        se = np.sqrt(var_Ypred + sigmasq)
        ci_lower = Ypred - q * se
        ci_upper = Ypred + q * se
        return ci_lower, ci_upper
    
    def ols(self):
        X = self.X
        Y = self.Y
        invXTX = np.linalg.inv(X.T @ X)
        popt = invXTX @ X.T @ Y
        dof = X.shape[0] - X.shape[1]
        sigmasq = np.square(X @ popt - Y).sum() / dof
        self.popt = popt
        self.pcov = sigmasq * invXTX
        self.dof = dof
    
    def qr_decomposition(self):
        X = self.X
        Y = self.Y
        Q, R = np.linalg.qr(X)
        pinvX = np.linalg.pinv(X)
        popt = np.linalg.solve(R, Q.T @ Y)
        normed_pcov = np.linalg.inv(R.T @ R)  # same as invXTX
        dof = X.shape[0] - X.shape[1]
        scale = np.square(X @ popt - Y).sum() / dof  # same as (sigma ** 2)
        self.popt = popt
        self.pcov = scale * normed_pcov
        self.dof = dof
    
    def bse(self):
        return np.sqrt(np.diag(self.pcov))
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.ols()  # alternatively, self.qr_decomposition()
    
    # when only X.shape[1] == 2
    def plot(self, outfilename, n_grid = 100):
        import seaborn as sns
        import matplotlib
        import matplotlib.pyplot as plt
        if X.shape[1] != 2:
            return
        gridX = np.linspace(self.X[:,1].min(), self.X[:,1].max(), n_grid)
        gridX = np.hstack((np.ones((n_grid, 1)), np.array([gridX]).T))
        pred_ci_lower, pred_ci_upper = self.pred_ci(gridX)
        obs_ci_lower, obs_ci_upper = self.obs_ci(gridX)
        plt.fill_between(
            gridX[:,1], pred_ci_lower[0], pred_ci_upper[0],
            alpha=.2, label='Pred_CI', fc = 'Orange',
        )
        plt.fill_between(
            gridX[:,1], obs_ci_lower[0],  obs_ci_upper[0],
            alpha=.1, label='Obs_CI', fc = 'tab:blue',
        )
        plt.plot(gridX[:,1], gridX @ self.popt, color = 'orange', alpha = 0.5)
        sns.scatterplot(
            x = X[:,1].T,
            y = Y.T[0],
            alpha = 0.5,
            color = 'tab:blue',
            edgecolor = None,
            linewidth = 0,
        )
        '''
        # if want to check concordance with statsmodels
        import statsmodels.api as sm
        res = sm.OLS(self.Y, self.X).fit()
        pred = res.get_prediction().summary_frame(0.05)
        plt.plot(self.X[:,1], pred['obs_ci_lower'], color = 'k')
        plt.plot(self.X[:,1], pred['obs_ci_upper'], color = 'k')
        plt.plot(self.X[:,1], pred['mean_ci_lower'], color = 'k')
        plt.plot(self.X[:,1], pred['mean_ci_upper'], color = 'k')
        '''
        plt.savefig(outfilename)
        plt.close()


if __name__ == '__main__':
    X = np.array([[1, 1, 1, 1], [2, 3, 5, 6], [4, 6, 3, 5]]).T
    Y = np.array([[4, 7, 8, 10]]).T
    model = OLS()
    model.fit(X, Y)
    print(model.popt)
    print(model.pcov)
    
    X = np.array([[1, 1, 1, 1], [2, 3, 5, 6]]).T
    Y = np.array([[4, 7, 8, 10]]).T
    model = OLS()
    model.fit(X, Y)
    model.plot('plot.OLS.pdf')
    print(model.popt)
    print(model.pcov)
