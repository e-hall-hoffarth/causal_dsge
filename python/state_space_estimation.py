from state_space_estimation.dag import dag
from state_space_estimation.roles import roles
from state_space_estimation.estimation import estimation
import pandas as pd
import numpy as np
import gc
import sys
import argparse as ap
from sklearn.linear_model import LinearRegression

desc = '''
Execute the state space estimation algorithm on the indicated data source
The algorithm involves a "brute-force" search over all possible models,
starting with models with no states (or min_states if specified), and
then continues until a model with x states that is valid relative to
the constraint tests is found. The algorithm then terminates after having
estimated all models with x states. Print to a csv file in data the valid 
model(s) found.
'''

parser = ap.ArgumentParser(description=desc, formatter_class = ap.RawTextHelpFormatter)
parser.add_argument('source', help='''one of: 
"rbc": use data from data/rbc.csv
"nk": use data from data/gali.csv
"sw": use data from data/sw.csv
"real": use data from data/real_data.csv''')
parser.add_argument('-t', '--test', required=False, help='Testing strategy to employ, one of (score, srivastava, multiple)')
parser.add_argument('-a', '--alpha', required=False, help='Nominal significance level of constraint tests (default 0.05)')
parser.add_argument('-m', '--min_states', required=False, help='consider only models with > min_states')
parser.add_argument('-M', '--max_states', required=False, help='consider only models with < max_states')
parser.add_argument('-n', '--sample_size', required=False, help='max sample size')
parser.add_argument('-r', '--random_state', required=False, help='random state for subsampling')
parser.add_argument('-c', '--repeat', required=False, help='how many times to repeat test')
parser.add_argument('-s', '--save', required=False, help='If argument is present output will be saved in data')

args = parser.parse_args()
source = args.source

if args.test:
    assert args.test in ['srivastava', 'multiple', 'score']
    method = args.test
else:
    method = 'srivastava'

if args.alpha:
    alpha = np.float64(args.alpha)
else:
    alpha = 0.05

if args.min_states:
    min_states = int(args.min_states)
else:
    min_states = 1

if args.sample_size:
    n = int(args.sample_size)
else:
    n = False

if args.random_state:
    random_state = int(args.random_state)
else:
    random_state = None

if args.repeat:
    repeat = int(args.repeat)
else:
    repeat = False

if args.save:
    save = True
else:
    save = False

if source == 'rbc':
    data = pd.read_csv('../data/rbc.csv')

elif source == 'nk':
    data = pd.read_csv('../data/nk.csv')

elif source == 'sw':
    data = pd.read_csv('../data/sw.csv')
    
    shift_vars = data.columns.values
    shift = data.loc[:,shift_vars].shift()
    shift.columns = [str(col) + '_1' for col in shift.columns]
    data = pd.concat([data, shift], axis=1)
    data = data.iloc[1:,:]

elif source == 'real':
    data = pd.read_csv('../data/real_data.csv', index_col='DATE')

else:
    raise ValueError("Source data not supported")

if args.max_states:
    max_states = min(int(args.max_states) + 1, int(len(data.columns.values)/3) - 1)
else:
    max_states = int(len(data.columns.values)/3) - 1

def test(data):
    est = estimation(data)
    if method == 'score':
        results = pd.DataFrame()
        for i in range(min_states, max_states):
            print('Evaluating models with {} states'.format(i))
            results = results.append(est.choose_states(i, method=method, alpha=alpha, tests=['score']), ignore_index=True)
        results = results.sort_values(by='bic', ascending=False)
        return results
    else:
        for i in range(min_states, max_states):
            print('Evaluating models with {} states'.format(i))
            results = est.choose_states(i, method=method, alpha=alpha)
            if results[results['valid']].shape[0] > 0:
                return results[results['valid']].sort_values(by=['nexo','loglik'], ascending=[True,False])
            else:
                del results
                gc.collect()
        return None # No valid models found

if repeat:
    wins = pd.DataFrame(index=pd.MultiIndex.from_frame(pd.DataFrame(columns=['exo_states','endo_states'])), columns=['wins','valid'])  
    results = pd.DataFrame()
    for i in range(repeat):
        print('Iteration {}'.format(i+1))
        if n:
            sample = data.sample(n, random_state=random_state*i, replace=True)
        else:
            sample = data
        result = test(sample)
        if result is not None:
            true_valid = False
            true_index = -1
            total_valid = result.shape[0]
            for i in range(result.shape[0]):
                row = result.iloc[i,:]
                if source == 'rbc' and set(row['exo_states']) == set(['z', 'g']) and set(row['endo_states']) == set(['k']):
                    true_valid = True
                    true_index = i
                elif source == 'nk' and set(row['exo_states']) == set(['nu', 'a', 'z']) and set(row['endo_states']) == set(['p']):
                    true_valid = True
                    true_index = i
                if wins.index.isin([('_'.join(sorted(row['exo_states'])), '_'.join(sorted(row['endo_states'])))]).any():
                    wins.loc[('_'.join(sorted(row['exo_states'])), '_'.join(sorted(row['endo_states']))),'wins'] += 1 if i == 0 else 0
                    wins.loc[('_'.join(sorted(row['exo_states'])), '_'.join(sorted(row['endo_states']))),'valid'] += 1
                else:
                    wins.loc[('_'.join(sorted(row['exo_states'])), '_'.join(sorted(row['endo_states']))),'wins'] = 1 if i == 0 else 0
                    wins.loc[('_'.join(sorted(row['exo_states'])), '_'.join(sorted(row['endo_states']))),'valid'] = 1

            result = result.iloc[0,:]
            result['true_valid'] = true_valid
            result['true_index'] = true_index
            result['total_valid'] = total_valid
            results = results.append(result, ignore_index=True)
            print('Result: exo_states: {}, endo_states: {}, true model valid: {}, true index: {}, total valid: {}'.format(
                result['exo_states'], result['endo_states'], result['true_valid'], result['true_index'], result['total_valid']
            ))
        else:
            print('No valid model found')
        wins.sort_values(by='wins', ascending=False, inplace=True)
    if save:
        results.to_csv('../data/{}_{}_{}_{}_{}iter_results.csv'.format(source, str(n), str(method), str(alpha), str(repeat)))
        wins.to_csv('../data/{}_{}_{}_{}_{}iter_wins.csv'.format(source, str(n), str(method), str(alpha), str(repeat)))

else:
    if n:
        sample = data.sample(n, random_state=random_state, replace=True)
        results = test(sample)
    else:
        results = test(data)
    if results is not None:
        print('Found valid model with {} states'.format(results['nstates'].iloc[0]))
        if save:
            results.to_csv('../data/{}_{}_{}_{}_results.csv'.format(source, str(n), str(method), str(alpha)))
        for result in results.iterrows():
            print('exo_states: {} || endo_states: {} || controls: {} || log-likelihood:{}'.format(
                result[1]['exo_states'], result[1]['endo_states'], result[1]['controls'], result[1]['loglik']
            ))
    else:
        print('No valid model found')

