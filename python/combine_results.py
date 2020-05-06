#!/bin/python

# Arguments
# 1: endogenous variables
# 2: exogenous variables
# 3: names (one per line)
# 4: destination of output

import pandas as pd
import csv
import sys

endo = pd.read_csv(sys.argv[1], header=None).T
exo = pd.read_csv(sys.argv[2], header=None)

reader = csv.reader(open(sys.argv[3]))
names = []
for name in reader:
  names.append(name[0])
out = pd.concat([endo, exo], axis = 1)
out.columns = names

out.to_csv(sys.argv[4])

