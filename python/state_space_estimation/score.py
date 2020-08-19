import numpy as np
from sklearn.linear_model import LinearRegression
import math

def sigma_sq(Y, Y_hat):
    n = Y.shape[0]
    sigma_sq = (1/n)*np.mean(np.sum((Y - Y_hat)**2))
    return sigma_sq


def llik(Y, Y_hat):
    n = Y.shape[0]
    sigma = sigma_sq(Y, Y_hat)
    ll = (-1 * (n/2) * np.log(2*math.pi) - 
            (n/2) * np.log(sigma) - 
            (1/(2*sigma)) * np.sum((Y - Y_hat)**2))
    return ll
    
    
def loglik(roles, data):
    ll = 0

    t_controls = np.append(roles.controls_idx, roles.endo_states_idx)
    n = data.shape[0]
    X_hat = np.empty((n,0))
    if len(roles.exo_states) > 0:
        for lag_exo_state, exo_state in zip(roles.exo_states_idx, roles.lag_exo_states_idx):
            X = data[:,lag_exo_state].reshape(-1,1)
            Y = data[:,exo_state].reshape(-1,1)
            model = LinearRegression(fit_intercept=False, normalize=False)
            model.fit(X, Y)
            Y_hat = model.predict(X)
            
            ll += llik(Y, Y_hat)
            X_hat = np.append(X_hat.reshape(n,-1), model.predict(X).reshape(n,-1), axis=1)
        
    X_hat = np.append(X_hat.reshape(n,-1), data[:,roles.lag_endo_states_idx], axis=1)
    # X_hat = np.append(data[:,roles.exo_states_idx], data[:,roles.lag_endo_states_idx], axis=1)
    
    if X_hat.shape[1] > 0:
        for control in t_controls:
            Y = data[:,control]
            model = LinearRegression(fit_intercept=False, normalize=False)
            model.fit(X_hat, Y)
            Y_hat = model.predict(X_hat)
            ll += llik(Y, Y_hat)

    else: # No states in model
        for control in t_controls:
            ll += llik(data[:,control], np.zeros((n,1)))
        
    return ll
        

def aic(L, roles, data):
    k = 2*(len(roles.exo_states) + 
        2*(len(roles.exo_states) + len(roles.endo_states))*
        (len(roles.controls) + len(roles.endo_states)))
    return 2*k - 2*L

    
def bic(L, roles, data):
    k = 2*(len(roles.exo_states) + 
        2*(len(roles.exo_states) + len(roles.endo_states))*
        (len(roles.controls) + len(roles.endo_states)))
    n = data.shape[0]
    return k * np.log(n) - 2 * L


def mse(roles, data):
    t_controls = np.append(roles.controls_idx, roles.endo_states_idx)
    n = data.shape[0]
    X_hat = np.empty((n,0))
    if len(roles.exo_states) > 0:
        for lag_exo_state, exo_state in zip(roles.exo_states_idx, roles.lag_exo_states_idx):
            X = data[:,lag_exo_state].reshape(-1,1)
            Y = data[:,exo_state].reshape(-1,1)
#             model = sm.OLS(Y, X)
#             model_fit = model.fit()
            model = LinearRegression(fit_intercept=False, normalize=False)
            model.fit(X, Y)
            X_hat = np.append(X_hat.reshape(n,-1), model.predict(X).reshape(n,-1), axis=1) 
    # X_hat = np.append(X_hat.reshape(n,-1), data[:,roles.lag_endo_states_idx], axis=1)
    X_hat = np.append(data[:,roles.exo_states_idx], data[:,roles.lag_endo_states_idx], axis=1)
    if X_hat.shape[1] > 0:
        X = np.append(data[:,roles.exo_states_idx], data[:,roles.lag_endo_states_idx], axis=1)
        Y = data[:,t_controls]
#         model = sm.OLS(Y, X_hat)
#         model_fit = model.fit()
#         Y_hat = model_fit.predict(X)
        model = LinearRegression(fit_intercept=False, normalize=False)
        model.fit(X, Y)
        Y_hat = model.predict(X)
        mse = np.mean(np.sum((Y - Y_hat)**2, axis=0))
    else:
        mse = (1/n)*np.sum((data[:,t_controls])**2)
    
    return mse


def score_tests(roles, data):
    L = loglik(roles, data.values)
    b = bic(L, roles, data.values) 
    a = aic(L, roles, data.values)
    m = mse(roles, data.values)
    tests = {
        'loglik': L,
        'bic': b,
        'aic': a,
        'mse': m
    }  
    return tests