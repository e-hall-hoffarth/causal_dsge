import numpy as np
import pandas as pd
import math
from tqdm.notebook import tqdm
from itertools import chain, combinations, product
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from .roles import roles
from .dag import dag
from .constraint import constraint_tests
from .score import score_tests

def nCr(n,r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)


class estimation():
    def __init__(self, data):
        self.data = data
        self.results = None


    def potential_states(self, n_states=None):
        x = self.data.columns.values[:int(len(self.data.columns.values)/2)]
        limit = len(x)-1 if n_states is None else n_states
        exo_states = chain.from_iterable(combinations(x, r) for r in range(limit+1))
        for exo in exo_states:
            y = [z for z in x if z not in exo]
            endo_states = combinations(y, limit-len(exo))
            for endo in endo_states:
                controls = [z for z in x if z not in endo and z not in exo]
                yield roles(exo, endo, controls, x)
        return None


    def evaluate_states(self, roles, tests=['score', 'constraint'], alpha=0.05, verbose=False):
        if verbose: 
            print('Evaluating states {}'.format(list(roles.exo_states) + [es + '_1' for es in roles.endo_states]))
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
            results['constraint_tests'] = ct
        if 'score' in tests: 
            st = score_tests(roles, self.data)             
            results = {**results, **st}
        results['nstates'] = len(roles.endo_states) + len(roles.exo_states)
        results['nexo'] = len(roles.exo_states)
        results['nendo'] = len(roles.endo_states)
        results['roles'] = roles
        return results


    def choose_states(self, n_states, alpha=0.05, tests=['score', 'constraint'], max_states=None, verbose=False):
        results = pd.DataFrame()
        variables = self.data.columns.values[:int(len(self.data.columns.values)/2)]
        ps = tqdm(self.potential_states(n_states=n_states),
                total=(nCr(len(variables), n_states) * (2 ** (n_states))))
        for states in ps:
            result = self.evaluate_states(states, tests, alpha=alpha, verbose=verbose)
            results = results.append(result, ignore_index=True)
            if 'valid' in results.columns:
                results['valid'] = results['valid'].astype(bool)
        return results

            
    def choose_states_parallel(self, n_states, alpha=0.05, tests=['score', 'constraint'], verbose=False):
        variables = self.data.columns.values[:np.int64(len(self.data.columns.values)/2)]
        states = self.potential_states(n_states=n_states)
        results = Parallel(n_jobs=cpu_count())(delayed(self.evaluate_states)(state, tests, alpha, verbose) 
                                            for state in tqdm(states, 
                                                                total=(nCr(len(variables), n_states) * (2 ** (n_states)))))
        return pd.DataFrame(results)