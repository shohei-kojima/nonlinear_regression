#!/usr/bin/env python

"""
Author: Shohei Kojima @ RIKEN
"""

class OLS():
    def __init__(self):
        self.X = None
        self.Y = None
        self.popt = None
        self.pcov = None
        self.dof = None
        self.F = None
        self.W = None
        self.P = None
    
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
    
    def bse(self):
        return np.sqrt(np.diag(self.pcov))
    
    def f_test(self):
        F = ((self.popt.T / self.bse())[0]) ** 2
        P = st.f.sf(F, 1, self.dof)
        self.F = F
        self.P = P
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.ols()
    
    
class BetaApproximation():
    def __init__(self):
        self.X_orig = None
        self.Y_orig = None
        self.X = None
        self.Y = None
        self.n_var = None
        self.actual_dof = None
        self.nominal_p = None
        self.a = None
        self.b = None
        self.beta_approx_p = None
    
    def ols_f(self, X, Y):
        invXTX = np.linalg.inv(X.T @ X)
        popt = invXTX @ X.T @ Y
        sigmasq = np.square(X @ popt - Y).sum() / self.actual_dof
        pcov = sigmasq * invXTX
        F = ((popt.T / np.sqrt(np.diag(pcov)))[0]) ** 2
        P = st.f.sf(F, 1, self.actual_dof)
        return P
    
    def residual(self, C, Y):
        C = np.hstack((np.ones((C.shape[0], 1)).reshape(-1, 1), C))
        model = OLS()
        model.fit(C, Y)
        return Y - C @ model.popt
    
    @staticmethod
    def beta_log_likelihood(x, a, b):
        """negative log-likelihood of beta distribution"""
        #https://github.com/broadinstitute/tensorqtl/blob/9857a3c6b15d2e2eed40d9aefd1af4c678b21edd/tensorqtl/core.py#L332
        logbeta = loggamma(a) + loggamma(b) - loggamma(a+b)
        return (1.0-a)*np.sum(np.log(x)) + (1.0-b)*np.sum(np.log(1.0-x)) + len(x)*logbeta
    
    def fit_beta(self):
        mean = np.mean(self.min_ps)
        var =  np.var(self.min_ps)
        a = mean * (mean * (1 - mean) / var - 1)
        b = a * (1/mean - 1)
        res = opt.minimize(
            lambda s: self.beta_log_likelihood(self.min_ps, s[0], s[1]), [a, b],
            method = 'Nelder-Mead',
        )
        self.a = res.x[0]
        self.b = res.x[1]
    
    def calc_beta_approx_p(self):
        self.beta_approx_p = st.beta.cdf(self.nominal_p, self.a, self.b)
    
    def calc_nominal_p(self):
        ps = []
        for i in range(self.n_var):
            ps.append(self.ols_f(self.X[:,i:i+1], self.Y))
        self.nominal_p = min(ps)
    
    def calc_permut_p(self, n_permut = 10000):
        min_ps = []
        index = np.arange(self.Y.shape[0])
        for _ in range(n_permut):
            permut_Y = self.Y[np.random.permutation(index)]
            ps = []
            for i in range(self.n_var):
                ps.append(self.ols_f(self.X[:,i:i+1], permut_Y))
            min_ps.append(min(ps))
        self.min_ps = np.array(min_ps)
    
    def beta_approximation(self, X, Y, C):
        self.X_orig = X
        self.Y_orig = Y
        X = st.zscore(X, axis = 0)
        Y = st.zscore(self.residual(C, Y))
        self.X = X
        self.Y = Y
        self.n_var = self.X.shape[1]
        self.actual_dof = X.shape[0] - C.shape[1] - 1
        self.calc_nominal_p()
        self.calc_permut_p()
        self.fit_beta()
        self.calc_beta_approx_p()




if __name__ == '__main__':
    # 10 SNVs, AF = 0.5, n_sample = 40
    X = []
    for _ in range(3):
        X.append([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ])
    for _ in range(8):
        X.append([
            0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0,
            1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0,
            0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0,
            1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0,
        ])
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
    ]
    Y = np.array(Y).reshape(-1, 1)

    C = np.random.normal(0, 0.1, X.shape[0]).reshape(-1, 1)

    model = BetaApproximation()
    model.beta_approximation(X, Y, C)

    print(model.a)
    print(model.b)
    print(model.nominal_p, model.n_var)
    print(model.beta_approx_p)

