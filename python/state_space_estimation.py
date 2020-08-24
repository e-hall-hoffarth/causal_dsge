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
parser.add_argument('-n', '--sample_size', required=False, help='max sample size')
parser.add_argument('-m', '--min_states', required=False, help='consider only models with > min_states')

args = parser.parse_args()
source = args.source
if args.sample_size:
    n = int(args.sample_size)
else:
    n = 1000

if args.min_states:
    min_states = int(args.min_states)
else:
    min_states = 0

if source == 'rbc':
    data = pd.read_csv('../data/rbc_100k.csv')
    data = data.drop(['Unnamed: 0', 'eps_g', 'eps_z'], axis=1)
    data.columns = [col.replace(" ", "") for col in data.columns]

    shift_vars = data.columns.values
    shift = data.loc[:,shift_vars].shift()
    shift.columns = [str(col) + '_1' for col in shift.columns]
    data = pd.concat([data, shift], axis=1)
    data = data.iloc[1:,:]

    data = data.iloc[:n,:]
    data = data.apply(lambda x: x - x.mean(), axis=0)

elif source == 'nk':
    data = pd.read_csv('../data/gali.csv')
    data.columns = [col.replace(" ", "") for col in data.columns]
    data = data.drop(['Unnamed:0', 
                      'eps_a', 'eps_z', 'eps_nu',
                      'pi_ann', 'r_nat_ann', 'r_real_ann', 'm_growth_ann', 'i_ann',
                      'y_gap', 'mu_hat', 'yhat',
                      'm_nominal'], 
                axis=1)

    shift_vars = data.columns.values
    shift = data.loc[:,shift_vars].shift()
    shift.columns = [str(col) + '_1' for col in shift.columns]
    data = pd.concat([data, shift], axis=1)
    data = data.iloc[1:,:]

    data = data.iloc[:n,:]
    data = data.apply(lambda x: x - x.mean(), axis=0)

elif source == 'sw':
    data = pd.read_csv('../data/sw.csv')
    data.columns = [col.replace(" ", "") for col in data.columns]
    data = data.drop(['Unnamed:0', 
                      'robs', 'labobs', 'pinfobs',
                      'dy', 'dc', 'dw', 'dinve',                    
                      'ea', 'eb', 'eg', 'eqs', 'em', 'epinf', 'ew',
                      'ewma', 'epinfma'], 
                axis=1)

    shift_vars = data.columns.values
    shift = data.loc[:,shift_vars].shift()
    shift.columns = [str(col) + '_1' for col in shift.columns]
    data = pd.concat([data, shift], axis=1)
    data = data.iloc[1:,:]

    data = data.iloc[:n,:]
    data = data.apply(lambda x: x - x.mean(), axis=0)

elif source == 'real':
    data = pd.read_csv('../data/real_data.csv', index_col='DATE')
    data = data.iloc[:n,:]

else:
    raise ValueError("Source data not supported")

est = estimation(data)
for i in range(min_states, int(len(data.columns.values)/2) - 1):
    print('Evaluating models with {} states'.format(i))
    results = est.choose_states_parallel(i)
    if results[results['valid']].shape[0] > 0:
        print('Found valid model with {} states'.format(i))
        results[results['valid']].to_csv('../data/{}_{}_results.csv'.format(source, n))
        for result in results[results['valid']].iterrows():
            print('exo_states: {} || endo_states: {} || controls: {}'.format(
                result[1]['exo_states'], result[1]['endo_states'], result[1]['controls']
            ))
        break
    else:
        del results
        gc.collect()
