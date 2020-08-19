import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy import stats

def partial_correlation(y, x, z=None, tol=1e-2):
    if z is None or z.shape[1] == 0:
        pcorr = stats.pearsonr(y, x)
    else:
        model_y = LinearRegression(fit_intercept=False, normalize=False)
        model_x = LinearRegression(fit_intercept=False, normalize=False)
        model_y.fit(z, y)
        model_x.fit(z, x)
        resid_y = y - model_y.predict(z)
        resid_x = x - model_x.predict(z)
        
        pcorr = stats.pearsonr(resid_y, resid_x) # return p-value
        
        # If there is evidence that x or y are _totally explained_ by z
        # then the residuals are a constant --- zero
        # any constant is independent of any RV or any other constant
        if model_y.score(z, y) > 1-tol or model_x.score(z, x) > 1-tol:
            pcorr = (0, 1)
    
    return pcorr


def constraint_tests(roles, names, data):
    data = data.values
    tests = []
    # Test that controls and endogenous states are conditionally independent
    # Conditional on lag of endo states and exo states
    for (x, y) in combinations(np.append(roles.controls_idx, roles.endo_states_idx), 2):
        X = data[:,x]
        Y = data[:,y]
        z = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
        
        Z = data[:,z]
        
        p = partial_correlation(X, Y, Z)
        pcorr = p[0]
        pval = p[1]
        tests.append({'x': names[x],
                      'y': names[y],
                      'z': names[z] if len(z) > 0 else [],
                      'pcorr': pcorr,
                      'pval': pval})
    
    # Test that controls and endo_states are indendent of lagged exo_states
    # and lagged controls conditional on current exo_states and lagged
    # endo_states
    for y in np.append(roles.controls_idx, roles.endo_states_idx):
        for x in np.append(roles.lag_controls_idx, roles.lag_exo_states_idx):
            X = data[:,x]
            Y = data[:,y]
            z = np.append(roles.lag_endo_states_idx, roles.exo_states_idx)
            
            Z = data[:,z]

            p = partial_correlation(X, Y, Z)
            pcorr = p[0]
            pval = p[1]
            tests.append({'x': names[x],
                          'y': names[y],
                          'z': names[z] if len(z) > 0 else [],
                          'pcorr': pcorr,
                          'pval': pval})
    
    # Test that current exo_states and lag endo_states are independent
    # Conditional on lag exo_states
    for y in roles.exo_states_idx:
        for x in roles.lag_endo_states_idx:
            X = data[:,x]
            Y = data[:,y]
            z = roles.lag_exo_states_idx
            
            Z = data[:,z]

            p = partial_correlation(X, Y, Z)
            pcorr = p[0]
            pval = p[1]
            tests.append({'x': names[x],
                          'y': names[y],
                          'z': names[z] if len(z) > 0 else [],
                          'pcorr': pcorr,
                          'pval': pval})
    
    # Test that exogenous states are (marginally) independent
    # This is a somewhat optional / restrictive assumption
    if len(roles.exo_states) > 1:
        for (x, y) in combinations(roles.exo_states_idx, 2):
            X = data[:,x]
            Y = data[:,y]
            p = partial_correlation(X, Y)
            pcorr = p[0]
            pval = p[1]
            tests.append({'x': names[x],
                          'y': names[y],
                          'z': [],
                          'pcorr': pcorr,
                          'pval': pval})

    return tests