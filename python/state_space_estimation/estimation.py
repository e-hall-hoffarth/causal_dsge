import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from itertools import chain, combinations
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from .roles import roles
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


    def evaluate_states(self, roles,
                        tests=('score', 'constraint'),
                        alpha=0.05,
                        return_tests=True,
                        verbose=False):
        '''
        Inputs:
            roles: state_space_estimation.roles
            tests: tuple(('score'), ('constraint'))
                Which types of tests to perform
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
            print('\nEvaluating states {}'.format(list(roles.exo_states) + [es + '_1' for es in roles.endo_states]))
        names = roles.names
        results = {}
        results['exo_states'] = roles.exo_states
        results['endo_states'] = roles.endo_states
        results['controls'] = roles.controls
        if 'constraint' in tests:
            ct = constraint_tests(roles, names, self.data) 
            m = len(ct)
            valid = (len(ct) > 0) and all([test['pval'] > (alpha/2)/m for test in ct]) 
            if valid and verbose:
                print('Valid states found: {}'.format(np.append(roles.exo_states, roles.endo_states)))
            results['valid'] = valid
            results['mean_pval'] = np.mean([test['pval'] for test in ct]) if len(ct) > 0 else None
            results['max_pval'] = max([test['pval'] for test in ct]) if len(ct) > 0 else None
            results['min_pval'] = min([test['pval'] for test in ct]) if len(ct) > 0 else None
            if return_tests:
                results['constraint_tests'] = ct
        if 'score' in tests: 
            st = score_tests(roles, self.data)             
            results = {**results, **st}
        results['nstates'] = len(roles.endo_states) + len(roles.exo_states)
        results['nexo'] = len(roles.exo_states)
        results['nendo'] = len(roles.endo_states)
        return results


    def choose_states(self, n_states, alpha=0.05, tests=['score', 'constraint'], return_tests=True, verbose=False):
        '''
        Inputs:
            n_states: int
                the number of states in models to consider
            alpha: float in (0, 1)
                The significance level to apply in constraint testing
            tests: tuple(('score'), ('constraint'))
                Which types of tests to perform
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
            result = self.evaluate_states(states, tests, alpha=alpha, return_tests=True, verbose=verbose)
            results = results.append(result, ignore_index=True)
            if 'valid' in results.columns:
                results['valid'] = results['valid'].astype(bool)
        return results

    def choose_states_parallel(self, n_states,
                               alpha=0.05,
                               tests=['score', 'constraint'],
                               return_tests=True,
                               serial=False,
                               verbose=False):
        """
        Args:
            n_states:
            serial: bool: whether to perform the iterations more slowly, not in parallel

        Returns:
             See self.choose_states; implements same functionality with a parallel backend.
        """
        opts = {
            'tests': tests,
            'alpha': alpha,
            'return_tests': return_tests,
            'verbose': verbose
        }

        states = self.potential_states(n_states=n_states)

        # TODO: can the expression below be simpler?
        len_variables = len(self.data.columns.values[:np.int64(len(self.data.columns.values)/2)])
        tqdm_total = nCr(len_variables, n_states) * (2 ** n_states)

        if serial:
            tq = tqdm(states, total=tqdm_total)
            return pd.DataFrame([self.evaluate_states(state, **opts) for state in tq])

        par = Parallel(n_jobs=cpu_count())
        tq = tqdm(states, total=tqdm_total)
        results = par(delayed(self.evaluate_states)(state, **opts) for state in tq)
        return pd.DataFrame(results)
