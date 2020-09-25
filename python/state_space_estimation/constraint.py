import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy import stats
from joblib import Memory
import tempfile
import warnings

cachedir = tempfile.gettempdir()
mem = Memory(cachedir)
#  TODO: make sure that this cache is cleared-up on the drive after running - I think it is
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
    residual_cache = mem.cache(get_residuals, verbose=False)
    if z is None or z.shape[1] == 0:
        return stats.pearsonr(y, x)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        resid_y, score_y = residual_cache(y, z)
        resid_x, score_x = residual_cache(x, z)

    # If there is evidence that x or y are _totally explained_ by z
    # then the residuals are a constant --- zero
    # any constant is independent of any RV or any other constant
    if score_y > 1-tol or score_x > 1-tol:
        return 0, 1

    return stats.pearsonr(resid_y, resid_x) # return p-value


def get_residuals(Y, X):
    model = LinearRegression(fit_intercept=False, normalize=False)
    model.fit(X, Y)
    residuals = Y - model.predict(X)
    return residuals, model.score(X, Y)


def arrange_results(pcorr, pval, names, x, y, z):
    return {'x': names[x],
            'y': names[y],
            'z': names[z] if len(z) > 0 else [],
            'pcorr': pcorr,
            'pval': pval}


def constraint_tests(roles, names, data):
    '''
    Inputs:
        roles: state_space_estimation.roles
        names: array-like 
        data: pd.DataFrame
    Performs:
        Conduct constraint-based (partial correlation) tests on data
        given the state-space model specified by roles and return all
        tests in a dictionary
    Returns:
        tests: dict
    '''
    # TODO: We can calculate partial correlations from a single covariance matrix 
    #       instead which will greatly speed up this code
    data = data.values
    tests = []
    # Test that controls and endogenous states are conditionally independent
    # Conditional on lag of endo states and exo states
    for (x, y) in combinations(np.append(roles.controls_idx, roles.endo_states_idx), 2):
        z = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
        pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
        tests.append(arrange_results(pcorr, pval, names, x, y, z))
    
    # Test that controls and endo_states are independent of lagged exo_states
    # and lagged controls conditional on current exo_states and lagged
    # endo_states
    for y in np.append(roles.controls_idx, roles.endo_states_idx):
        for x in np.append(roles.lag_controls_idx, roles.lag_exo_states_idx):
            z = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
            pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
            tests.append(arrange_results(pcorr, pval, names, x, y, z))

    # Test that current exo_states and lag endo_states are independent
    # Conditional on lag exo_states
    for y in roles.exo_states_idx:
        for x in roles.lag_endo_states_idx:
            z = roles.lag_exo_states_idx
            pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
            tests.append(arrange_results(pcorr, pval, names, x, y, z))

    # Test that exogenous states are (marginally) independent
    # This is a somewhat optional / restrictive assumption
    if len(roles.exo_states) > 1:
        for (x, y) in combinations(roles.exo_states_idx, 2):
            z = roles.lag_exo_states_idx
            pcorr, pval = partial_correlation(data[:, x], data[:, y], data[:, z])
            res = arrange_results(pcorr, pval, names, x, y, z)
            tests.append(res)

    return tests