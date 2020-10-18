import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from itertools import chain, combinations, product
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from .roles import roles
from .dag import dag
from .constraint import constraint_tests
from .score import score_tests

def nCr(n,r):
    '''
    Inputs:
        n: int
        r: int
    Returns:
        n choose r: int
    '''
    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)


class estimation():
    def __init__(self, data):
        self.data = data
        self.results = None


    def potential_states(self, n_states):
        '''
        Inputs:
            n_states: int
        Performs: 
            Create a generator containing a state_space_estimation.roles objects
            for every possible state space model given this data and n_states
        Returns:
            generator
        '''
        variables = self.data.columns.values[:int(len(self.data.columns.values)/2)]
        limit = len(variables)-1 if n_states is None else n_states
        exo_states = chain.from_iterable(combinations(variables, r) for r in range(limit+1))
        for exo in exo_states:
            y = [z for z in variables if z not in exo]
            endo_states = combinations(y, limit-len(exo))
            for endo in endo_states:
                controls = [z for z in variables if z not in endo and z not in exo]
                yield roles(exo, endo, controls, self.data.columns.values)
        return None


    def evaluate_states(self, roles, tests=('score', 'constraint'), 
                        method='custom_3', alpha=0.05, verbose=False):
        '''
        Inputs:
            roles: state_space_estimation.roles
            tests: tuple(('score'), ('constraint'))
                Which types of tests to perform
            method: one of ('srivastava', 'schott', 'custom_3', 'custom_4')
                Testing strategy to use (for constrain tests)
            alpha: float in (0, 1)
                The significance level to apply in constraint testing
            return_tests:
                If true results contains all constraint tests that were performed
                (memory intensive)
            verbose: bool
                If true print progress
        Performs:
            Perform tests on model given in roles. Perform tests in score.py 
            if 'score' is in tests and tests in constraint.py if 'constraint'
             in tests. Return results in a dictionary.
        Outputs:
            results: dict
        '''
        if verbose: 
            print('Evaluating states {}'.format(list(roles.exo_states) + [es + '_1' for es in roles.endo_states]))
        results = {}
        results['exo_states'] = roles.exo_states
        results['endo_states'] = roles.endo_states
        results['controls'] = roles.controls
        if 'constraint' in tests:
            ct = constraint_tests(roles, self.data, method=method, alpha=alpha) 
            results = {**results, **ct}
        if 'score' in tests: 
            st = score_tests(roles, self.data)             
            results = {**results, **st}
        results['nstates'] = len(roles.endo_states) + len(roles.exo_states)
        results['nexo'] = len(roles.exo_states)
        results['nendo'] = len(roles.endo_states)
        return results


    def choose_states(self, n_states, tests=['score', 'constraint'], 
                      method='custom_3', alpha=0.05, verbose=False):
        '''
        Inputs:
            n_states: int
                the number of states in models to consider
                Which types of tests to perform
            tests: tuple(('score'), ('constraint'))
            method: one of ('srivastava', 'schott', 'custom_3', 'custom_4')
                Testing strategy to use (for constrain tests)
            alpha: float in (0, 1)
                The significance level to apply in constraint testing
            verbose: bool
                If true print progress
        Performs:
            Evaluate all possible modes with n_states given the observed variables in this
            estimator's data, and return results from the specified types of tests in
            a pandas dataframe
        Returns:
            results: pd.DataFrame
        '''
        results = pd.DataFrame()
        variables = self.data.columns.values[:int(len(self.data.columns.values)/2)]
        ps = tqdm(self.potential_states(n_states=n_states),
                total=(nCr(len(variables), n_states) * (2 ** (n_states))))
        for states in ps:
            result = self.evaluate_states(states, tests, method=method, 
                                          alpha=alpha, verbose=verbose)
            results = results.append(result, ignore_index=True)
            if 'valid' in results.columns:
                results['valid'] = results['valid'].astype(bool)
        return results

            
    def choose_states_parallel(self, n_states, tests=['score', 'constraint'], 
                               method='custom_3', alpha=0.05, verbose=False):
        '''
        See self.choose_states; implements same functionality with a parallel backend.
        '''
        variables = self.data.columns.values[:np.int64(len(self.data.columns.values)/2)]
        states = self.potential_states(n_states=n_states)
        results = Parallel(n_jobs=cpu_count())(delayed(self.evaluate_states)(state, tests, method, alpha, verbose) 
                                            for state in tqdm(states, 
                                                                total=(nCr(len(variables), n_states) * (2 ** (n_states)))))
        return pd.DataFrame(results)