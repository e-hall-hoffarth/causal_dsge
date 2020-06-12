#!/bin/python

# Arguments 
# 1: mod file
# 2: destination of output

from pynare import pynare
import numpy as np
import pandas as pd
import sys, os

try:
    sim = pynare(sys.argv[1])
except Exception as e:
    os._exit(1)

endo_names = sim.workspace['M_']['endo_names']
exo_names = sim.workspace['M_']['exo_names']
names = np.append(endo_names, [exo_names])

endo = sim.oo_.endo_simul.T
exo = sim.oo_.exo_simul
data = pd.DataFrame(np.append(endo, exo, axis=1), columns=names)

data.to_csv(sys.argv[2])
os._exit(0)