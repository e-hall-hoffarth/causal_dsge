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
    

def custom(data, l, tol=1e-5):
    '''
    Inputs:
        data: np.ndarray
            Residual correlation matrix
    Performs:
        Perform a test modified from Schott (2005) to test the
        hypothesis that all of the elements in the upper right
        lxm right trapazoid of the correlation matrix are zero
    Returns:
        float
    '''
    n = data.shape[0]   # sample size
    m = data.shape[1]   # total number of variables
    k = (l/2)*(2*m-l-1) # number of elements of R considered
    if m > 1 and k > 0:
        R = np.corrcoef(data.T)      
        t_nm = (np.sum(np.square(np.triu(R, k=1))) 
                - np.sum(np.square(np.triu(R[:(m-l),:(m-l)], k=1)))
                - k/n)
        s_nm = (2*k*(n-1))/((n**2)*(n+2))
        return t_nm/np.sqrt(s_nm)
    
    else: # Test isn't meaningful, so do not exclude the model on this basis
        return 0


def get_resids(roles, data, ntests=4):
    '''
    Inputs:
        roles: state_space_estimation.roles
        data: pd.DataFrame
        ntests: int in (2, 3, 4)
            Number of tests (from paper) that can be tested with the 
            returned residuals
    Performs:
        Collect linear regression residuals for the specified number
        of tests from the model specified in roles
    Returns:
        (np.ndarray, np.ndarray)
    '''
    # Use numpy indexing instead of pandas for large performance increase
    # (At the expense of some increased code complexity)
    data = data.values
    
    # Conditioning sets
    cset1 = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
    cset2 = roles.lag_exo_states_idx
    
    # Targets 
    
    # All 4 tests
    if ntests == 4:
        tar1 = np.append(roles.lag_exo_states_idx, np.append(roles.endo_states_idx, roles.controls_idx))
        tar2 = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
        
    # Exclude (7) --- causes trouble with simulated data
    elif ntests == 3:
        tar1 = np.append(roles.endo_states_idx, roles.controls_idx)
        tar2 = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
    
    # Only 2 tests involving (strictly) diagonal matrices
    elif ntests == 2:
        tar1 = np.append(roles.endo_states_idx, roles.controls_idx)
        tar2 = roles.exo_states_idx
    
    else:
        raise ValueError("Valid numbers of tests are 2, 3 and 4")
    
    if cset1.shape[0] > 0:
        lm1 = LinearRegression(fit_intercept=True, normalize=False)
        lm1.fit(data[:,cset1], data[:,tar1])
        resid1 = data[:,tar1] - lm1.predict(data[:,cset1])
        
    else:
        resid1 = data[:,tar1]
        
    if cset2.shape[0] > 0:
        lm2 = LinearRegression(fit_intercept=True, normalize=False)
        lm2.fit(data[:,cset2], data[:,tar2])
        resid2 = data[:,tar2] - lm2.predict(data[:,cset2])
    else:
        resid2 = data[:,tar2]
    
    return resid1, resid2


def constraint_tests(roles, data, method='custom_3', alpha=0.05, tol=1e-20):
    '''
    Inputs:
        roles: state_space_estimation.roles
            Model upon which to conduct constraint tests
        data: pd.DataFrame
        method: one of ('srivastava', 'schott', 'custom_3', 'custom_4')
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
    if method == 'srivastava':
        resid1, resid2 = get_resids(roles, data, ntests=2)
        if np.var(resid1.flatten()) < tol:
            t1 = 0
            t2 = srivastava(resid2)
        elif np.var(resid2.flatten()) < tol:
            t1 = srivastava(resid1)
            t2 = 0
        else:
            t1 = srivastava(resid1)
            t2 = srivastava(resid2)

        crit_val = stats.norm.ppf(1-(alpha/2)) # One-sided test w/ Bonferroni correction
        p1 = 1 - stats.norm.cdf(t1)
        p2 = 1 - stats.norm.cdf(t2)
        if t1 > crit_val or t2 > crit_val:
            valid = False

    elif method == 'schott':
        resid1, resid2 = get_resids(roles, data, ntests=2)
        if np.var(resid1.flatten()) < tol:
            t1 = 0
            t2 = schott(resid2)
        elif np.var(resid2.flatten()) < tol:
            t1 = schott(resid1)
            t2 = 0
        else:
            t1 = schott(resid1)
            t2 = schott(resid2)
        
        crit_val = stats.norm.ppf(1-(alpha/4)) # Two-sided test w/ Bonferroni correction
        p1 = 2*(1 - stats.norm.cdf(np.abs(t1)))
        p2 = 2*(1 - stats.norm.cdf(np.abs(t2)))
        if np.abs(t1) > crit_val or np.abs(t2) > crit_val:
            valid = False
        
    elif method == 'custom_3':
        resid1, resid2 = get_resids(roles, data, ntests=3)

        l1 = len(roles.controls) + len(roles.endo_states)
        l2 = len(roles.exo_states)
        
        if np.var(resid1.flatten()) < tol:
            t1 = 0
            t2 = custom(resid2, l2)
        elif np.var(resid2.flatten()) < tol:
            t1 = custom(resid1, l1)
            t2 = 0
        else:
            t1 = custom(resid1, l1)
            t2 = custom(resid2, l2)
        
        crit_val = stats.norm.ppf(1-(alpha/4)) # Two-sided test w/ Bonferroni correction
        p1 = 2*(1 - stats.norm.cdf(np.abs(t1)))
        p2 = 2*(1 - stats.norm.cdf(np.abs(t2)))
        if np.abs(t1) > crit_val or np.abs(t2) > crit_val:
            valid = False


    elif method == 'custom_4':
        resid1, resid2 = get_resids(roles, data, ntests=4)

        l1 = len(roles.controls) + len(roles.endo_states)
        l2 = len(roles.exo_states)
        
        if np.var(resid1.flatten()) < tol:
            t1 = 0
            t2 = custom(resid2, l2)
        elif np.var(resid2.flatten()) < tol:
            t1 = custom(resid1, l1)
            t2 = 0
        else:
            t1 = custom(resid1, l1)
            t2 = custom(resid2, l2)
        
        crit_val = stats.norm.ppf(1-(alpha/4)) # Two-sided test w/ Bonferroni correction
        p1 = 2*(1 - stats.norm.cdf(np.abs(t1)))
        p2 = 2*(1 - stats.norm.cdf(np.abs(t2)))
        if np.abs(t1) > crit_val or np.abs(t2) > crit_val:
            valid = False
                
    else:
        raise ValueError('method {} not found'.format(method))

    return {'t1': t1, 't2': t2, 'p1': p1, 'p2': p2, 'valid': valid}