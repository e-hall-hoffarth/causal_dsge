import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class roles():
    '''
    This class primarily exists to keep track of the indexes of each variable type in a 
    state space model because numpy integer indexing is much faster than pandas string
    indexing
    '''
    def __init__(self, exo_states, endo_states, controls, names):
        '''
        Inputs:
            exo_states: array-like
            endo_states: array-like
            controls: array-like
            names: array-like
                Combined, exo_states, endo_states, and controls specify the roles of each 
                variable in a state space model. These lists should be mutual exclusive, 
                and their union should be names.
        '''
        self.exo_states = list(exo_states)
        self.endo_states = list(endo_states)
        self.controls = list(controls)
        self.lag_exo_states = [x + '_1' for x in self.exo_states]
        self.lag_endo_states = [x + '_1' for x in self.endo_states]
        self.lag_controls = [x + '_1' for x in self.controls]

        self.names = names
        self.t_names = np.array([name for name in names if '_1' not in name])
        self.lag_names = np.array([name for name in names if '_1' in name])
        
        self.exo_states_idx = np.where([x in self.exo_states for x in self.names])[0]
        self.endo_states_idx = np.where([x in self.endo_states for x in self.names])[0]
        self.controls_idx = np.where([x in self.controls for x in self.names])[0]
        self.lag_exo_states_idx = np.where([x in self.lag_exo_states for x in self.names])[0]
        self.lag_endo_states_idx = np.where([x in self.lag_endo_states for x in self.names])[0]
        self.lag_controls_idx = np.where([x in self.lag_controls for x in self.names])[0]

    
    def make_adjacency(self, data):
        '''
        Inputs:
            data: pd.DataFrame
                Used for estimating parameters
        Performs:
            converts the model implied by this roles to a weighted adjacency matrix
        Returns:
            result: pd.DataFrame
        '''
        data_names = data.columns.values.tolist()
        data = data.values
        data_current = [name for name in data_names if '_1' not in name]
        implied_names = self.exo_states + self.endo_states + self.controls
        
        # Make sure roles and data make sense
        for name in implied_names:
            if name not in data_names:
                raise ValueError('Name {} missing from data'.format(name))
            if str(name) + '_1' not in data_names:
                raise ValueError('Lag of name {} missing from data'.format(name))
            if any([name not in data_names for name in implied_names]) or any([name not in implied_names for name in data_current]):
                print(data_current)
                print(implied_names)
                raise ValueError('Implied names and data do not align')
        
        d = len(data_names)
        result = pd.DataFrame(np.zeros((d, d)), columns=data_names, index=data_names)
        
        for (exo_state, lag_exo_state) in zip(self.exo_states_idx, self.lag_exo_states_idx):
            model = LinearRegression(fit_intercept=False)
            model.fit(data[:,[lag_exo_state]], data[:,[exo_state]])
            result.iloc[lag_exo_state, exo_state] = model.coef_[0]
            
        for endo in np.append(self.endo_states_idx, self.controls_idx):
            outcome_name = self.names[endo]
            regressors = np.append(self.lag_endo_states_idx, self.exo_states_idx)
            regressor_names = self.names[regressors]
            model = LinearRegression(fit_intercept=False)
            model.fit(data[:,regressors], data[:,endo])
            coefs = {}
            for i in range(len(regressors)):
                coefs[regressor_names[i]] = model.coef_[i]
            for x in regressor_names:
                result.loc[x, outcome_name] = coefs[x]
        
        return result