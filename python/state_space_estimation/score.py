import numpy as np
from sklearn.linear_model import LinearRegression
import math

def sigma_sq(Y, Y_hat):
    '''
    Inputs:
        Y: array-like
        Y_hat: array-like
    Returns:
        Mean squared error between Y and Y_hat
    '''
    sigma_sq = np.mean((Y - Y_hat)**2)
    return sigma_sq


def llik(Y, Y_hat):
    '''
    Inputs:
        Y: array-like
        Y_hat: array-like
    Returns:
        Log-likelihood of Y_hat assuming Y is the true value and is normally distributed
    '''
    n = Y.shape[0]
    sigma = sigma_sq(Y, Y_hat)
    ll = (-1 * (n/2) * np.log(2*math.pi) - 
               (n/2) * np.log(sigma) - 
               (n/2)) # The mse in the last term cancels out after we plug in the ML value of sigma_squared
    return ll
    
    
def loglik(roles, data):
    '''
    Inputs:
        roles: state_space_estimation.roles
        data: pd.DataFrame
    Performs:
        Calculate the loglikelihood of the model specified 
        by roles relative to data
    Returns
        ll: float
    '''
    ll = 0

    t_controls = np.append(roles.controls_idx, roles.endo_states_idx)
    n = data.shape[0]
    if len(roles.exo_states) > 0:
        for lag_exo_state, exo_state in zip(roles.exo_states_idx, roles.lag_exo_states_idx):
            X = data[:,lag_exo_state].reshape(-1,1)
            Y = data[:,exo_state].reshape(-1,1)
            model = LinearRegression(fit_intercept=False, normalize=False)
            model.fit(X, Y)
            Y_hat = model.predict(X)
            
            ll += llik(Y, Y_hat)
        
    X_hat = np.append(data[:,roles.exo_states_idx], data[:,roles.lag_endo_states_idx], axis=1)
    
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
        

def aic(roles, data):
    '''
    Inputs:
        roles: state_space_estimation.roles
        data: pd.DataFrame
    Performs:
        Calculate the Alaike Information Criterion
    Returns
        float
    '''
    k = 2*(len(roles.exo_states) + 
        2*(len(roles.exo_states) + len(roles.endo_states))*
        (len(roles.controls) + len(roles.endo_states)))
    L = loglik(roles, data)
    return 2*k - 2*L

    
def bic(roles, data):
    '''
    Inputs:
        roles: state_space_estimation.roles
        data: pd.DataFrame
    Performs:
        Calculate the Bayesian Information Criterion
    Returns
        float
    '''
    k = 2*(len(roles.exo_states) + 
        2*(len(roles.exo_states) + len(roles.endo_states))*
        (len(roles.controls) + len(roles.endo_states)))
    L = loglik(roles, data)
    n = data.shape[0]
    return k * np.log(n) - 2 * L


def mse(roles, data):
    '''
    Inputs:
        roles: state_space_estimation.roles
        data: pd.DataFrame 
    Performs:
        Computes the MSE of the model specified by roles relative to data
    Returns:
        mse: float
    '''
    mse = 0
    t_controls = np.append(roles.controls_idx, roles.endo_states_idx)
    n = data.shape[0]
    if len(roles.exo_states) > 0:
        for lag_exo_state, exo_state in zip(roles.exo_states_idx, roles.lag_exo_states_idx):
            X = data[:,lag_exo_state].reshape(-1,1)
            Y = data[:,exo_state].reshape(-1,1)
            model = LinearRegression(fit_intercept=False, normalize=False)
            model.fit(X, Y)
            Y_hat = model.predict(X)
            mse += sigma_sq(Y, Y_hat)
    X_hat = np.append(data[:,roles.exo_states_idx], data[:,roles.lag_endo_states_idx], axis=1)
    if X_hat.shape[1] > 0:
        X = np.append(data[:,roles.exo_states_idx], data[:,roles.lag_endo_states_idx], axis=1)
        Y = data[:,t_controls]
        model = LinearRegression(fit_intercept=False, normalize=False)
        model.fit(X, Y)
        Y_hat = model.predict(X)
        mse += sigma_sq(Y, Y_hat)
    else:
        mse += np.mean((data[:,t_controls])**2)
    
    return mse


def score_tests(roles, data):
    '''
    Inputs:
        roles: state_space_estimation.roles
        data: pd.DataFrame 
    Performs:
        Calculate all score tests for model specified by roles relative to data
    Returns:
        tests: dict
    '''
    L = loglik(roles, data.values)
    b = bic(roles, data.values) 
    a = aic(roles, data.values)
    m = mse(roles, data.values)
    tests = {
        'loglik': L,
        'bic': b,
        'aic': a,
        'mse': m
    }  
    return tests