import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings


def srivastava(data):
    '''
    Inputs:
        data: np.ndarray
            Residual correlation matrix
    Performs:
        Perform test T3* from Srivastava (2005) to test wheter
        the corrleation matrix is diagonal
    Returns:
        float
    '''
    n = data.shape[0]
    p = data.shape[1]
    if p > 1:
        S = np.cov(data.T)
        a2_hat = np.sum(np.square(np.diag(S)))
        a4_hat = np.sum(np.power(np.diag(S), 4))
        a20_hat = (n/(p*(n+2)))*a2_hat
        a40_hat = (1/p)*a4_hat
        g3_hat = (n/(n-1))*((np.trace(np.dot(S,S))-(1/n)*(np.trace(S))**2)/(np.sum(np.square(np.diag(S)))))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            T3_hat = (n/2)*((g3_hat-1)/np.sqrt(1-(1/p)*(a40_hat/(a20_hat**2))))
            if np.isnan(T3_hat): # Could have sqrt of negative, replace as in Wang et al.
                T3_hat = (n/2)*((g3_hat-1)/np.sqrt(1-(a4_hat/(a2_hat**2))))

        return T3_hat
    else: # Test isn't meaningful, so do not exclude the model on this basis
        return 0        


def schott(data):
    '''
    Inputs:
        data: np.ndarray
            Residual correlation matrix
    Performs:
        Perform test from Schott (2005) to test wheter
        the corrleation matrix is diagonal
    Returns:
        float
    '''
    n = data.shape[0]
    m = data.shape[1]
    if m > 1:
        R = np.corrcoef(data.T)
        t_nm = np.sum(np.square(np.triu(R, k=1))) - ((m*(m-1))/(2*n))
        s_nm = (m*(m-1)*(n-1))/((n**2)*(n+2))

        return t_nm/np.sqrt(s_nm)
    
    else: # Test isn't meaningful, so do not exclude the model on this basis
        return 0


def get_resids(roles, data):
    '''
    Inputs:
        roles: state_space_estimation.roles
        data: pd.DataFrame
    Performs:
        Collect linear regression residuals from the model specified in roles
    Returns:
        (np.ndarray, np.ndarray)
    '''
    # Use numpy indexing instead of pandas for large performance increase
    # (At the expense of some increased code complexity)
    data = data.values
    
    # Conditioning sets
    cset = np.append(roles.lag_2_endo_states_idx, roles.lag_exo_states_idx)
    
    # Targets 
    tar = np.append(np.append(np.append(roles.lag_endo_states_idx, 
                                        roles.lag_controls_idx), 
                              roles.exo_states_idx), 
                    roles.lag_2_exo_states_idx)
    
    if cset.shape[0] > 0:
        lm = LinearRegression(fit_intercept=True, normalize=False)
        lm.fit(data[:,cset], data[:,tar])
        resid = data[:,tar] - lm.predict(data[:,cset]) 
    else:
        resid = data[:,tar]
    
    return resid


def constraint_tests(roles, data, method='srivastava', alpha=0.05, tol=1e-20):
    '''
    Inputs:
        roles: state_space_estimation.roles
            Model upon which to conduct constraint tests
        data: pd.DataFrame
        method: one of ('srivastava', 'schott')
            Testing strategy to use
        alpha: float
            Significance level
        tol: float
            Tolerence, used for detecting near zero residuals 
            which make testing unstable
    Performs:
        Conduct constraint-based (partial correlation) tests on data
        given the state-space model specified by roles and return all
        tests in a dictionary (two test statistics, two p-values, and
        overall decision 'valid')
    Returns:
        tests: dict
    '''
    valid = True
    resid = get_resids(roles, data)
    f = len(roles.controls) + len(roles.endo_states)
    if method == 'srivastava':
        t = srivastava(resid)
        crit_val = stats.norm.ppf(1-(alpha)) # One-sided test
        p = 1 - stats.norm.cdf(t)
        if t > crit_val:
            valid = False

    elif method == 'schott':
        t = schott(resid)
        crit_val = stats.norm.ppf(1-(alpha/2)) # Two-sided test
        p = 2*(1 - stats.norm.cdf(np.abs(t)))
        if np.abs(t) > crit_val:
            valid = False
        
    else:
        raise ValueError('method {} not found'.format(method))

    return {'t': t, 'p': p, 'valid': valid}
