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


def get_resids_multiple(roles, data):
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
    tar = np.append(np.append(roles.lag_endo_states_idx, 
                              roles.lag_controls_idx), 
                    roles.exo_states_idx) 

    if cset.shape[0] > 0:
        lm = LinearRegression(fit_intercept=True, normalize=False)
        lm.fit(data[:,cset], data[:,tar])
        resid = data[:,tar] - lm.predict(data[:,cset]) 
    else:
        resid = data[:,tar]
    
    return resid


def get_resids(Y, X):
    model = LinearRegression(fit_intercept=True, normalize=False)
    model.fit(X, Y)
    residuals = Y - model.predict(X)
    return residuals, model.score(X, Y)


def partial_correlation(y, x, z=None, tol=1e-5):
    '''
    Arguments:
        y: array-like
        x: array-like
        z: array-like or None
        tol: float
    Performs:
        Calculate the partial correlation between y and x conditional on z,
        by first regressing y and x on z, and then the correlation between
        the residuals.
    Returns:
        tuple(partial correlation, p-value): (float, float)
    '''
    if z is None or z.shape[1] == 0:
        return stats.pearsonr(y, x)

    resid_y, score_y = get_resids(y, z)
    resid_x, score_x = get_resids(x, z)

    # If there is evidence that x or y are _totally explained_ by z
    # then the residuals are a constant --- zero
    # any constant is independent of any RV or any other constant
    if score_y > 1-tol or score_x > 1-tol:
        return 0, 1

    return stats.pearsonr(resid_y, resid_x) # (t, p)


def arrange_results(pcorr, pval, names, x, y, z):
    return {'x': names[x],
            'y': names[y],
            'z': names[z] if len(z) > 0 else [],
            'pcorr': pcorr,
            'pval': pval}


def constraint_tests(roles, data, method='srivastava', alpha=0.05, tol=1e-20):
    '''
    Inputs:
        roles: state_space_estimation.roles
            Model upon which to conduct constraint tests
        data: pd.DataFrame
        method: one of ('srivastava', 'multiple')
            Testing strategy to use
        alpha: float
            Significance level
        tol: float
            Tolerence, used for detecting near zero residuals 
            which make testing unstable
    Performs:
        Conduct constraint-based (partial correlation) tests on data
        given the state-space model specified by roles and return all
        tests in a dictionary
    Returns:
        tests: dict
    '''
    if method == 'srivastava':
        resid = get_resids_multiple(roles, data)
        t = srivastava(resid)
        crit_val = stats.norm.ppf(1-(alpha/2)) # two-sided test
        p = 2*(1 - stats.norm.cdf(np.abs(t)))
        if np.abs(t) > crit_val:
            valid = False
        else:
            valid = True

        return {'t': t, 'p': p, 'valid': valid, method: 'srivastava'}

    elif method == 'multiple':
        data = data.values
        tests = []

        # Test that controls and endogenous states are conditionally independent
        # Conditional on lag of endo states and exo states
        for (x, y) in combinations(np.append(roles.controls_idx, roles.endo_states_idx), 2):
            z = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
            pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
            tests.append(arrange_results(pcorr, pval, roles.names, x, y, z))
        
        # Test that controls and endo_states are independent of lagged exo_states
        # and lagged controls conditional on current exo_states and lagged
        # endo_states
        for y in np.append(roles.controls_idx, roles.endo_states_idx):
            for x in np.append(roles.lag_controls_idx, roles.lag_exo_states_idx):
                z = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
                pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
                tests.append(arrange_results(pcorr, pval, roles.names, x, y, z))

        # Test that current exo_states and lag endo_states are independent
        # Conditional on lag exo_states
        for y in roles.exo_states_idx:
            for x in roles.lag_endo_states_idx:
                z = roles.lag_exo_states_idx
                pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
                tests.append(arrange_results(pcorr, pval, roles.names, x, y, z))

        # Test that exogenous states are (marginally) independent
        # This is a somewhat optional / restrictive assumption
        if len(roles.exo_states) > 1:
            for (x, y) in combinations(roles.exo_states_idx, 2):
                z = roles.lag_exo_states_idx
                pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
                res = arrange_results(pcorr, pval, roles.names, x, y, z)
                tests.append(res)

        crit_p = alpha/len(tests)
        valid = not any([test['pval'] < crit_p for test in tests])
        min_p = min([test['pval'] for test in tests])
        return {min_p: 'min_p', 'valid': valid, method: 'multiple'}
        
    else:
        raise ValueError('method {} not found'.format(method))
